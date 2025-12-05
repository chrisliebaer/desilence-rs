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
}

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
	/// Audio stream index used for silence detection
	pub detection_stream_index: usize,
	/// Whether to output all streams or only video + detection audio
	pub output_all_streams: bool,
}

/// Run the streaming pipeline, outputting audible segments to stdout.
///
/// This decodes the input file and selectively outputs frames that fall
/// within audible segments, writing to stdout in NUT format.
pub fn run_pipeline<P: AsRef<std::path::Path>>(
	input_path: P,
	segments: &SegmentList,
	config: &PipelineConfig,
) -> Result<PipelineStats> {
	let input_path = input_path.as_ref();

	info!(
			path = %input_path.display(),
			audible_segments = segments.audible_count(),
			"Starting streaming pipeline"
	);

	ffmpeg::init()?;

	let mut ictx = format::input(input_path)?;

	// Create output context for stdout with NUT format
	// We need to use pipe: protocol
	let mut octx = create_stdout_output()?;

	// Build stream mappings
	let mut stream_mappings: Vec<StreamMapping> = Vec::new();
	let mut video_decoder: Option<(usize, codec::decoder::Video)> = None;
	let mut audio_decoders: HashMap<usize, codec::decoder::Audio> = HashMap::new();
	let mut audio_encoders: HashMap<usize, codec::encoder::Audio> = HashMap::new();
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
				encoder_ctx.set_time_base(ist.time_base());
				encoder_ctx.set_frame_rate(decoder.frame_rate());

				let opened = encoder_ctx.open()?;
				ost.set_parameters(&opened);

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

	// Process packets
	let mut stats = PipelineStats::default();
	let mut current_output_pts: HashMap<usize, i64> = HashMap::new();

	// Track which segment we're currently in for efficient lookup
	let audible_segments: Vec<_> = segments.audible_segments().cloned().collect();

	info!(
		streams = stream_mappings.len(),
		audible_segments = audible_segments.len(),
		"Starting frame processing"
	);

	let (_video_ist_index, mut video_dec) = video_decoder.unwrap();

	for (stream, packet) in ictx.packets() {
		let ist_index = stream.index();

		// Find mapping for this stream
		let mapping = match stream_mappings.iter().find(|m| m.input_index == ist_index) {
			Some(m) => m,
			None => continue, // Stream not mapped
		};

		// Calculate timestamp in seconds
		let packet_pts = packet.pts().unwrap_or(0);
		let packet_timestamp = packet_pts as f64 * f64::from(mapping.input_time_base);

		// Check if this packet is in an audible segment (for non-decoded streams)
		let is_packet_audible = audible_segments.iter().any(|seg| seg.contains(packet_timestamp));

		match mapping.media_type {
			media::Type::Video => {
				// Decode ALL video packets to maintain decoder state (reference frames)
				video_dec.send_packet(&packet)?;

				let mut decoded = frame::Video::empty();
				while video_dec.receive_frame(&mut decoded).is_ok() {
					let frame_pts = decoded.pts().unwrap_or(0);
					let frame_timestamp = frame_pts as f64 * f64::from(mapping.input_time_base);

					if !audible_segments.iter().any(|seg| seg.contains(frame_timestamp)) {
						continue; // Skip silent frames
					}

					// Adjust PTS for continuous output
					let output_pts = current_output_pts.entry(mapping.output_index).or_insert(0);

					decoded.set_pts(Some(*output_pts));

					// For rawvideo, we can write the frame data directly as a packet
					let mut out_packet = Packet::empty();

					// Calculate frame duration
					let frame_duration = if let Some(fps) = video_dec.frame_rate() {
						if fps.numerator() > 0 {
							let tb = mapping.output_time_base;
							// Duration = (TB.den * FPS.den) / (TB.num * FPS.num)
							let num = tb.denominator() as i64 * fps.denominator() as i64;
							let den = tb.numerator() as i64 * fps.numerator() as i64;
							(num + den / 2) / den
						} else {
							packet.duration()
						}
					} else {
						packet.duration()
					};

					// Use safe getters for frame properties
					let format = decoded.format();
					let width = decoded.width();
					let height = decoded.height();
					let align = 1;

					// Calculate required buffer size for the full image (all planes)
					// SAFETY: av_image_get_buffer_size computes size from valid format/dims.
					// We rely on Into<AVPixelFormat> for safe conversion.
					let size = unsafe { ffmpeg::ffi::av_image_get_buffer_size(format.into(), width as i32, height as i32, align) };

					if size > 0 {
						let pkt_ptr = out_packet.as_mut_ptr();

						// Allocate packet with correct size
						// CRITICAL: av_new_packet resets PTS/DTS, so we must set them AFTER this call
						// SAFETY: pkt_ptr is obtained from initialized Packet via FFI wrapper.
						let alloc_ret = unsafe { ffmpeg::ffi::av_new_packet(pkt_ptr, size) };

						if alloc_ret >= 0 {
							unsafe {
								// SAFETY:
								// 1. pkt_ptr->data is valid buffer of 'size' bytes allocated above.
								// 2. decoded is valid Frame, so as_ptr() is valid.
								// 3. (*frame_ptr).data and linesize are valid for the given format.
								// 4. av_image_copy_to_buffer flattens planar data safely if size is correct.
								let frame_ptr = decoded.as_ptr();
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

					// Set packet properties AFTER allocation
					out_packet.set_pts(Some(*output_pts));
					out_packet.set_dts(Some(*output_pts));
					out_packet.set_duration(frame_duration);
					out_packet.set_stream(mapping.output_index);

					out_packet.write_interleaved(&mut octx)?;

					*output_pts += frame_duration;
					stats.output_frames += 1;
				}
			},
			media::Type::Audio => {
				if let Some(decoder) = audio_decoders.get_mut(&ist_index) {
					decoder.send_packet(&packet)?;

					let mut decoded = frame::Audio::empty();
					while decoder.receive_frame(&mut decoded).is_ok() {
						let frame_pts = decoded.pts().unwrap_or(0);
						let frame_timestamp = frame_pts as f64 * f64::from(mapping.input_time_base);

						if !audible_segments.iter().any(|seg| seg.contains(frame_timestamp)) {
							continue; // Skip silent frames
						}

						let output_pts = current_output_pts.entry(mapping.output_index).or_insert(0);

						// Resample/Convert frame to packed format
						let mut resampled_frame = frame::Audio::empty();

						// Let the resampler handle details, just run it
						if let Some(resampler) = audio_resamplers.get_mut(&ist_index) {
							resampler.run(&decoded, &mut resampled_frame)?;
						}

						// Update PTS for the new continuous stream
						resampled_frame.set_pts(Some(*output_pts));

						// Encode frame to packet
						if let Some(encoder) = audio_encoders.get_mut(&ist_index) {
							encoder.send_frame(&resampled_frame)?;

							let mut out_packet = Packet::empty();
							while encoder.receive_packet(&mut out_packet).is_ok() {
								out_packet.set_stream(mapping.output_index);
								out_packet.write_interleaved(&mut octx)?;
								stats.output_packets += 1;
							}
						}

						let samples = decoded.samples();
						*output_pts += samples as i64;
					}
				}
			},
			media::Type::Subtitle => {
				if is_packet_audible {
					// Copy subtitle packets with adjusted timing
					let mut out_packet = packet.clone();
					let _output_pts = current_output_pts.entry(mapping.output_index).or_insert(0);

					out_packet.set_stream(mapping.output_index);
					out_packet.rescale_ts(mapping.input_time_base, mapping.output_time_base);
					out_packet.write_interleaved(&mut octx)?;

					stats.output_packets += 1;
				}
			},
			_ => {
				if is_packet_audible {
					// Similar logic for other streams if needed, or just skipping
				}
			},
		}
	}

	// Flush decoders
	video_dec.send_eof()?;
	let mut decoded = frame::Video::empty();
	while video_dec.receive_frame(&mut decoded).is_ok() {
		// Process remaining frames
		stats.output_frames += 1;
	}

	for (_, decoder) in audio_decoders.iter_mut() {
		decoder.send_eof()?;
		let mut decoded = frame::Audio::empty();
		while decoder.receive_frame(&mut decoded).is_ok() {
			stats.output_frames += 1;
		}
	}

	// Write trailer
	octx.write_trailer()?;

	info!(
		output_frames = stats.output_frames,
		output_packets = stats.output_packets,
		dropped_packets = stats.dropped_packets,
		"Pipeline complete"
	);

	Ok(stats)
}

/// Create an output context writing to stdout in NUT format
fn create_stdout_output() -> Result<format::context::Output> {
	// SAFETY: We use "nut" format explicitly because simple "pipe:1" output detection might fail
	// or default to mpegts which we don't want.
	// ffmpeg-next exposes output_as which calls avformat_alloc_output_context2 safely.
	Ok(format::output_as(&"pipe:1", "nut")?)
}

/// Statistics from pipeline execution
#[derive(Debug, Default)]
pub struct PipelineStats {
	/// Number of frames written to output
	pub output_frames: u64,
	/// Number of packets written to output
	pub output_packets: u64,
	/// Number of packets dropped (in silent segments)
	pub dropped_packets: u64,
}

impl PipelineStats {
	/// Calculate percentage of content kept
	pub fn kept_percentage(&self) -> f64 {
		let total = self.output_packets + self.dropped_packets;
		if total == 0 {
			100.0
		} else {
			(self.output_packets as f64 / total as f64) * 100.0
		}
	}
}
