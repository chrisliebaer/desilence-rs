//! Streaming pipeline for selective output of audible segments

use std::collections::HashMap;

use ffmpeg_next::{
	self as ffmpeg,
	codec,
	encoder,
	format,
	frame,
	media,
	packet::Mut as PacketMut,
	software,
	Packet,
	Rational,
};
use tracing::{
	debug,
	info,
	warn,
};

use crate::{
	error::{
		DesilenceError,
		Result,
	},
	segment::SegmentList,
};

/// Stream mapping information
#[derive(Debug)]
struct StreamMapping {
	/// Input stream index
	input_index: usize,
	/// Output stream index
	output_index: usize,
	/// Input time base
	input_time_base: Rational,
	/// Output time base
	output_time_base: Rational,
	/// Stream type
	media_type: media::Type,
	/// Last processed linear PTS for this stream (for strict monotonicity)
	last_pts: Option<i64>,
}

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
	/// Audio stream index used for silence detection
	pub detection_stream_index: usize,
	/// Whether to output all streams or only video + detection audio
	pub output_all_streams: bool,
}

/// Mapping of an input segment to its time offset in the output
#[derive(Debug)]
struct SegmentOffset {
	start: f64,
	end: f64, // Use concrete end for easier lookup (last segment uses Infinity if needed)
	/// Amount to subtract from input timestamp to get output timestamp
	/// (Total silence duration removed prior to this segment)
	time_offset: f64,
}

/// Run the streaming pipeline, outputting audible segments to stdout.
///
/// This decodes the input file and selectively outputs frames that fall
/// within audible segments, writing to stdout in NUT format.
pub fn run_pipeline<P: AsRef<std::path::Path>>(
	input_path: P,
	output_path: Option<&std::path::Path>,
	segments: &SegmentList,
	config: &PipelineConfig,
) -> Result<()> {
	let input_path = input_path.as_ref();

	info!(
			path = %input_path.display(),
			output = ?output_path,
			audible_segments = segments.audible_count(),
			"Starting streaming pipeline"
	);

	ffmpeg::init()?;

	let mut ictx = format::input(input_path)?;

	// Create output context
	let mut octx = if let Some(path) = output_path {
		format::output(path)?
	} else {
		create_stdout_output()?
	};

	// Build stream mappings
	let mut stream_mappings: Vec<StreamMapping> = Vec::new();
	let mut video_decoder: Option<(usize, codec::decoder::Video)> = None;
	let mut audio_decoders: HashMap<usize, codec::decoder::Audio> = HashMap::new();
	let mut audio_encoders: HashMap<usize, encoder::Audio> = HashMap::new();
	let mut audio_resamplers: HashMap<usize, software::resampling::Context> = HashMap::new();

	let mut ost_index = 0usize;

	for ist in ictx.streams() {
		let ist_index = ist.index();
		let params = ist.parameters();
		let medium = params.medium();

		// Determine if we should include this stream
		let include = match medium {
			media::Type::Video => true,
			media::Type::Audio => config.output_all_streams || ist_index == config.detection_stream_index,
			media::Type::Subtitle => config.output_all_streams,
			_ => false,
		};

		if !include {
			debug!(stream = ist_index, medium = ?medium, "Skipping stream");
			continue;
		}

		match medium {
			media::Type::Video => {
				// For video, we'll decode to raw and output as rawvideo
				let context = codec::context::Context::from_parameters(params)?;
				let decoder = context.decoder().video()?;

				// Add rawvideo stream to output
				let codec = encoder::find(codec::Id::RAWVIDEO).ok_or_else(|| DesilenceError::Output {
					message: "rawvideo codec not found".to_string(),
				})?;

				let mut ost = octx.add_stream(codec)?;

				// Configure output stream to match input
				let mut encoder_ctx = codec::context::Context::new_with_codec(codec).encoder().video()?;

				encoder_ctx.set_width(decoder.width());
				encoder_ctx.set_height(decoder.height());
				encoder_ctx.set_format(decoder.format());

				// Determine output time base from stream metadata.
				// 1. avg_frame_rate: Primary source (container metadata), verified to establish correct 30fps.
				// 2. r_frame_rate: Secondary source (nominal rate), also verified to establish correct 30fps.
				let output_time_base = if ist.avg_frame_rate().numerator() > 0 {
					let fps = ist.avg_frame_rate();
					Rational(fps.denominator(), fps.numerator())
				} else if ist.rate().numerator() > 0 {
					let fps = ist.rate();
					Rational(fps.denominator(), fps.numerator())
				} else {
					ist.time_base()
				};

				encoder_ctx.set_time_base(output_time_base);

				// Explicitly set the encoder frame rate derived from the verified time base.
				// This ensures the stream metadata is populated correctly.
				if output_time_base.numerator() > 0 {
					let fps = Rational(output_time_base.denominator(), output_time_base.numerator());
					encoder_ctx.set_frame_rate(Some(fps));
				}

				let opened = encoder_ctx.open()?;
				ost.set_parameters(&opened);
				ost.set_time_base(output_time_base);

				// Explicitly set stream-level frame rate metadata (avg_frame_rate).
				// set_parameters does not copy this property from the encoder context.
				if output_time_base.numerator() > 0 {
					let fps = Rational(output_time_base.denominator(), output_time_base.numerator());
					ost.set_avg_frame_rate(fps);
				}

				debug!(
						input = ist_index,
						output = ost_index,
						width = decoder.width(),
						height = decoder.height(),
						format = ?decoder.format(),
						"Added video stream"
				);

				video_decoder = Some((ist_index, decoder));

				stream_mappings.push(StreamMapping {
					input_index: ist_index,
					output_index: ost_index,
					input_time_base: ist.time_base(),
					output_time_base: ist.time_base(), // Will update after write_header
					media_type: medium,
					last_pts: None,
				});

				ost_index += 1;
			},
			media::Type::Audio => {
				// For audio, decode and output as PCM matching input format
				let context = codec::context::Context::from_parameters(params)?;
				let decoder = context.decoder().audio()?;

				// Choose PCM codec based on input format
				// Note: PCM codecs require packed (non-planar) formats
				let (pcm_codec_id, output_format) = match decoder.format() {
					format::Sample::F32(_) | format::Sample::F64(_) => {
						(codec::Id::PCM_F32LE, format::Sample::F32(format::sample::Type::Packed))
					},
					format::Sample::I32(_) => (codec::Id::PCM_S32LE, format::Sample::I32(format::sample::Type::Packed)),
					format::Sample::I16(_) => (codec::Id::PCM_S16LE, format::Sample::I16(format::sample::Type::Packed)),
					_ => (codec::Id::PCM_S16LE, format::Sample::I16(format::sample::Type::Packed)),
				};

				let codec = encoder::find(pcm_codec_id).ok_or_else(|| DesilenceError::Output {
					message: format!("PCM codec {:?} not found", pcm_codec_id),
				})?;

				let mut ost = octx.add_stream(codec)?;

				// Configure output stream
				let mut encoder_ctx = codec::context::Context::new_with_codec(codec).encoder().audio()?;

				encoder_ctx.set_rate(decoder.rate() as i32);
				encoder_ctx.set_channel_layout(decoder.channel_layout());
				encoder_ctx.set_format(output_format);
				encoder_ctx.set_time_base(Rational(1, decoder.rate() as i32));

				let opened = encoder_ctx.open()?;
				ost.set_parameters(&opened);

				debug!(
						input = ist_index,
						output = ost_index,
						rate = decoder.rate(),
						channels = decoder.channels(),
						input_format = ?decoder.format(),
						output_format = ?output_format,
						pcm_codec = ?pcm_codec_id,
						"Added audio stream"
				);

				// Create resampler to convert from input format to output (packed) format
				let resampler = software::resampling::Context::get(
					decoder.format(),
					decoder.channel_layout(),
					decoder.rate(),
					output_format,
					decoder.channel_layout(),
					decoder.rate(),
				)?;

				audio_decoders.insert(ist_index, decoder);
				audio_encoders.insert(ist_index, opened);
				audio_resamplers.insert(ist_index, resampler);

				stream_mappings.push(StreamMapping {
					input_index: ist_index,
					output_index: ost_index,
					input_time_base: ist.time_base(),
					output_time_base: Rational(1, 48000), // Will update after write_header
					media_type: medium,
					last_pts: None,
				});

				ost_index += 1;
			},
			media::Type::Subtitle => {
				// Copy subtitles as-is
				let mut ost = octx.add_stream(encoder::find(codec::Id::None))?;
				ost.set_parameters(params);

				// Clear codec tag for compatibility
				// SAFETY: ffmpeg-next's Parameters wrapper does not expose a setter for codec_tag,
				// so we must set it via the raw pointer to ensure compatibility.
				unsafe {
					(*ost.parameters().as_mut_ptr()).codec_tag = 0;
				}

				debug!(input = ist_index, output = ost_index, "Added subtitle stream (copy)");

				stream_mappings.push(StreamMapping {
					input_index: ist_index,
					output_index: ost_index,
					input_time_base: ist.time_base(),
					output_time_base: ist.time_base(),
					media_type: medium,
					last_pts: None,
				});

				ost_index += 1;
			},
			_ => {},
		}
	}

	// Validate we have required streams
	if video_decoder.is_none() {
		return Err(DesilenceError::NoVideoStream);
	}

	if audio_decoders.is_empty() {
		return Err(DesilenceError::NoAudioStream);
	}

	// Write output header
	octx.write_header()?;

	// Update output time bases after header is written
	for mapping in &mut stream_mappings {
		mapping.output_time_base = octx.stream(mapping.output_index).unwrap().time_base();
	}

	// Prepare Segment Offsets
	// This maps "Input Time" -> "Output Offset"
	// Output_Time = Input_Time - Offset
	let mut segment_offsets: Vec<SegmentOffset> = Vec::new();

	// Re-calculate offsets correctly by iterating ALL segments to accumulate silence
	let mut total_silence_removed = 0.0;
	for seg in segments.all_segments() {
		if seg.is_audible() {
			segment_offsets.push(SegmentOffset {
				start: seg.start,
				end: seg.end.unwrap_or(f64::INFINITY),
				time_offset: total_silence_removed,
			});
		} else if let Some(dur) = seg.duration() {
			total_silence_removed += dur;
		}
	}

	info!(
		streams = stream_mappings.len(),
		audible_segments = segment_offsets.len(),
		"Starting frame processing"
	);

	let (video_ist_index, mut video_dec) = video_decoder.unwrap();

	// Unified Global Offset (in Microseconds: AV_TIME_BASE)
	// Shared across all streams to maintain lock-step sync.
	let mut global_offset_micros: i64 = 0;

	/// Linearize timestamps using the global offset, correcting for glitches.
	///
	/// This function maintains a global synchronization by shifting the timeline whenever a
	/// non-monotonic timestamp is detected. It ensures all streams remain in lock-step relative
	/// to the shift.
	fn linearize_timestamp(mapping: &mut StreamMapping, global_offset_micros: &mut i64, raw_pts: i64, duration: i64) -> i64 {
		let offset_in_tb = unsafe {
			ffmpeg::ffi::av_rescale_q(
				*global_offset_micros,
				ffmpeg::ffi::AV_TIME_BASE_Q,
				mapping.input_time_base.into(),
			)
		};
		let mut linear_pts = raw_pts + offset_in_tb;

		if let Some(last) = mapping.last_pts {
			if linear_pts <= last {
				// Glitch detected: Shift global timeline to enforce monotonicity
				let target_pts = last + duration;
				let shift_needed_in_tb = target_pts - linear_pts;

				let shift_micros = unsafe {
					ffmpeg::ffi::av_rescale_q(
						shift_needed_in_tb,
						mapping.input_time_base.into(),
						ffmpeg::ffi::AV_TIME_BASE_Q,
					)
				};

				*global_offset_micros += shift_micros;

				let new_offset_in_tb = unsafe {
					ffmpeg::ffi::av_rescale_q(
						*global_offset_micros,
						ffmpeg::ffi::AV_TIME_BASE_Q,
						mapping.input_time_base.into(),
					)
				};
				linear_pts = raw_pts + new_offset_in_tb;

				// Final safety check
				if linear_pts <= last {
					linear_pts = last + 1;
				}
			}
		}

		mapping.last_pts = Some(linear_pts);
		linear_pts
	}

	/// Calculate frame duration in stream ticks based on FPS.
	fn calculate_duration_from_fps(fps: Rational, time_base: Rational) -> i64 {
		if fps.numerator() > 0 {
			(time_base.denominator() as f64 / time_base.numerator() as f64 / (fps.numerator() as f64 / fps.denominator() as f64))
				.round() as i64
		} else {
			0
		}
	}

	/// Calculate frame duration in stream ticks based on sample count.
	fn calculate_duration_from_samples(samples: i64, rate: i64, time_base: Rational) -> i64 {
		if rate > 0 {
			let num = time_base.numerator() as i64;
			let den = time_base.denominator() as i64;
			(samples * den) / (num * rate)
		} else {
			1024 // Fallback
		}
	}

	/// Calculate the final PTS for the output stream, applying silence skipping logic.
	fn calculate_final_output_pts(linear_pts: i64, mapping: &StreamMapping, segment_offset: &SegmentOffset) -> i64 {
		let time_offset_in_ticks = (segment_offset.time_offset / f64::from(mapping.output_time_base)).round() as i64;

		let linear_output_pts =
			unsafe { ffmpeg::ffi::av_rescale_q(linear_pts, mapping.input_time_base.into(), mapping.output_time_base.into()) };

		linear_output_pts - time_offset_in_ticks
	}

	for (stream, packet) in ictx.packets() {
		let ist_index = stream.index();

		// Find mapping for this stream
		let mapping = match stream_mappings.iter_mut().find(|m| m.input_index == ist_index) {
			Some(m) => m,
			None => continue, // Stream not mapped
		};

		// Calculate timestamp in seconds
		let packet_pts = packet.pts().unwrap_or(0);
		let packet_timestamp = packet_pts as f64 * f64::from(mapping.input_time_base);

		// Find which segment this matches
		// We use a simple linear search since N is small
		let current_segment = segment_offsets
			.iter()
			.find(|s| packet_timestamp >= s.start && packet_timestamp < s.end);

		match mapping.media_type {
			media::Type::Video => {
				// Decode ALL video packets to maintain decoder state (reference frames)
				video_dec.send_packet(&packet)?;

				let mut decoded = frame::Video::empty();
				while video_dec.receive_frame(&mut decoded).is_ok() {
					let input_pts = decoded.pts().unwrap_or(0);

					// Calculate correct duration based on frame rate if available
					let fps = video_dec.frame_rate();
					let duration_ticks = if let Some(fps) = fps {
						calculate_duration_from_fps(fps, mapping.input_time_base)
					} else {
						0
					};

					// Linearize (Presentation Time) -> Updates Global Offset if needed
					let linear_pts = linearize_timestamp(mapping, &mut global_offset_micros, input_pts, duration_ticks);

					// We use the original input_pts (Raw Time) for segment slicing checks
					let frame_timestamp = input_pts as f64 * f64::from(mapping.input_time_base);

					if let Some(seg_offset) = segment_offsets
						.iter()
						.find(|s| frame_timestamp >= s.start && frame_timestamp < s.end)
					{
						let final_pts = calculate_final_output_pts(linear_pts, mapping, seg_offset);

						let output_duration_ts = if let Some(fps) = fps {
							calculate_duration_from_fps(fps, mapping.output_time_base)
						} else {
							0
						};

						write_raw_video_frame(&mut octx, &decoded, mapping, final_pts, output_duration_ts)?;
					} else {
						// Silent frame, drop
					}
				}
			},
			media::Type::Audio => {
				if let Some(_offset) = current_segment {
					if let Some(decoder) = audio_decoders.get_mut(&ist_index) {
						decoder.send_packet(&packet)?;

						let mut decoded = frame::Audio::empty();
						while decoder.receive_frame(&mut decoded).is_ok() {
							let input_pts = decoded.pts().unwrap_or(0);

							// Calculate Duration for Linearization
							let samples = decoded.samples() as i64;
							let rate = decoder.rate() as i64;
							let duration_ticks = calculate_duration_from_samples(samples, rate, mapping.input_time_base);

							// Linearize (Presentation Time) -> Updates Global Offset if needed
							let linear_pts = linearize_timestamp(mapping, &mut global_offset_micros, input_pts, duration_ticks);

							let tb_val = f64::from(mapping.input_time_base);
							let raw_start_time = input_pts as f64 * tb_val;
							let duration_seconds = if rate > 0 { samples as f64 / rate as f64 } else { 0.0 };
							let raw_end_time = raw_start_time + duration_seconds;

							for segment in segments.all_segments() {
								if let Some(overlap) = crate::segment::get_overlap(raw_start_time, raw_end_time, segment.start, segment.end) {
									let overlap_start_rel = (overlap.start - raw_start_time).max(0.0);
									let overlap_end_rel = (overlap.end.unwrap() - raw_start_time).min(raw_end_time - raw_start_time);

									let samples_keep_start = (overlap_start_rel * rate as f64).round() as usize;
									let samples_keep_end = (overlap_end_rel * rate as f64).round() as usize;
									let sample_count = decoded.samples();

									if samples_keep_start < sample_count && samples_keep_end > samples_keep_start {
										let output_tb = mapping.output_time_base;
										let trim_ticks = unsafe {
											ffmpeg::ffi::av_rescale_q(samples_keep_start as i64, Rational(1, rate as i32).into(), output_tb.into())
										};

										if let Some(offset_struct) = segment_offsets.iter().find(|s| (s.start - segment.start).abs() < 0.001) {
											let base_pts = calculate_final_output_pts(linear_pts, mapping, offset_struct);
											let final_pts = base_pts + trim_ticks;

											// Trim, Resample, Encode, Write
											let keep_samples_len = samples_keep_end - samples_keep_start;
											crate::audio_trim::trim_audio_frame(&mut decoded, samples_keep_start, keep_samples_len);

											let mut resampled_frame = frame::Audio::empty();
											if let Some(resampler) = audio_resamplers.get_mut(&ist_index) {
												resampler.run(&decoded, &mut resampled_frame)?;
											}

											resampled_frame.set_pts(Some(final_pts));

											if let Some(encoder) = audio_encoders.get_mut(&ist_index) {
												encoder.send_frame(&resampled_frame)?;

												let mut out_packet = Packet::empty();
												flush_audio_encoder(encoder, &mut out_packet, &mut octx, mapping)?;
											}
										}
									}
								}
							}
						}
					}
				}
			},
			media::Type::Subtitle => {
				if let Some(offset) = current_segment {
					// Copy subtitle packets with shifted timing
					let mut out_packet = packet.clone();
					let output_timestamp = packet_timestamp - offset.time_offset;
					out_packet.set_stream(mapping.output_index);

					// Manually set PTS based on shift
					let tb_num = mapping.output_time_base.numerator() as f64;
					let tb_den = mapping.output_time_base.denominator() as f64;
					let new_pts = (output_timestamp * (tb_den / tb_num)).round() as i64;

					out_packet.set_pts(Some(new_pts));
					out_packet.set_dts(Some(new_pts)); // Subtitles usually PTS=DTS

					out_packet.rescale_ts(mapping.input_time_base, mapping.output_time_base);

					out_packet.write_interleaved(&mut octx)?;
				}
			},
			_ => {
				// Handle other streams
			},
		}
	}

	// Flush decoders
	// Re-find mapping for video stream using the captured index
	if let Some(mapping) = stream_mappings.iter_mut().find(|m| m.input_index == video_ist_index) {
		video_dec.send_eof()?;
		let mut decoded = frame::Video::empty();
		while video_dec.receive_frame(&mut decoded).is_ok() {
			let frame_pts = decoded.pts().unwrap_or(0);

			let fps = video_dec.frame_rate();
			let duration_ticks = if let Some(fps) = fps {
				calculate_duration_from_fps(fps, mapping.input_time_base)
			} else {
				0
			};

			let linear_pts = linearize_timestamp(mapping, &mut global_offset_micros, frame_pts, duration_ticks);

			let frame_timestamp = frame_pts as f64 * f64::from(mapping.input_time_base);

			// Check if frame is in an audible segment
			if let Some(seg_offset) = segment_offsets
				.iter()
				.find(|s| frame_timestamp >= s.start && frame_timestamp < s.end)
			{
				let final_pts = calculate_final_output_pts(linear_pts, mapping, seg_offset);

				let output_duration_ts = if let Some(fps) = fps {
					calculate_duration_from_fps(fps, mapping.output_time_base)
				} else {
					0
				};

				write_raw_video_frame(&mut octx, &decoded, mapping, final_pts, output_duration_ts)?;
			}
		}
	}

	for (index, decoder) in audio_decoders.iter_mut() {
		decoder.send_eof()?;
		let mut decoded = frame::Audio::empty();
		let encoder = audio_encoders.get_mut(index);
		let mapping = stream_mappings.iter().find(|m| m.input_index == *index);

		while decoder.receive_frame(&mut decoded).is_ok() {
			// Flush
		}

		if let Some(enc) = encoder {
			enc.send_eof()?;
			let mut out_packet = Packet::empty();
			if let Some(map) = mapping {
				flush_audio_encoder(enc, &mut out_packet, &mut octx, map)?;
			}
		}
	}

	// Write trailer
	octx.write_trailer()?;

	info!("Pipeline complete");

	Ok(())
}

/// Create an output context writing to stdout in NUT format
fn create_stdout_output() -> Result<format::context::Output> {
	Ok(format::output_as(&"pipe:1", "nut")?)
}

/// Helper to write a raw video frame to the output context.
/// Encapsulates unsafe FFmpeg packet allocation and buffer copying.
fn write_raw_video_frame(
	octx: &mut format::context::Output,
	decoded: &frame::Video,
	mapping: &StreamMapping,
	pts: i64,
	duration: i64,
) -> Result<()> {
	// Adjust PTS for continuous output
	let mut out_video_frame = decoded.clone();
	out_video_frame.set_pts(Some(pts));

	let mut out_packet = Packet::empty();

	// Use safe getters for frame properties
	let format = out_video_frame.format();
	let width = out_video_frame.width();
	let height = out_video_frame.height();
	let align = 1;

	// Calculate required buffer size for the full image (all planes)
	// SAFETY: av_image_get_buffer_size computes safe size from valid format/dims.
	let size = unsafe { ffmpeg::ffi::av_image_get_buffer_size(format.into(), width as i32, height as i32, align) };

	if size > 0 {
		let pkt_ptr = out_packet.as_mut_ptr();

		// Allocate packet with correct size
		// SAFETY: pkt_ptr is valid. av_new_packet handles allocation.
		let alloc_ret = unsafe { ffmpeg::ffi::av_new_packet(pkt_ptr, size) };

		if alloc_ret >= 0 {
			unsafe {
				// SAFETY: Valid buffer and frame pointers ensured by size check and alloc check.
				let frame_ptr = out_video_frame.as_ptr();
				ffmpeg::ffi::av_image_copy_to_buffer(
					(*pkt_ptr).data,
					size,
					(*frame_ptr).data.as_ptr() as *const *const u8,
					(*frame_ptr).linesize.as_ptr(),
					format.into(),
					width as i32,
					height as i32,
					align,
				);
			}
		}
	} else {
		warn!(
			"Failed to calculate buffer size for video frame: width={} height={} format={:?}",
			width, height, format
		);
	}

	// Set packet properties included updated PTS/DTS
	out_packet.set_pts(Some(pts));
	out_packet.set_dts(Some(pts));
	out_packet.set_duration(duration);
	out_packet.set_stream(mapping.output_index);

	out_packet.write_interleaved(octx)?;

	Ok(())
}

/// Helper to flush an audio encoder and write any remaining packets to the output context.
fn flush_audio_encoder(
	encoder: &mut encoder::Audio,
	out_packet: &mut Packet,
	octx: &mut format::context::Output,
	mapping: &StreamMapping,
) -> Result<()> {
	let encoder_tb = encoder.time_base();
	while encoder.receive_packet(out_packet).is_ok() {
		out_packet.set_stream(mapping.output_index);
		out_packet.rescale_ts(encoder_tb, mapping.output_time_base);
		out_packet.write_interleaved(octx)?;
	}
	Ok(())
}
