//! Silence detection using FFmpeg's silencedetect filter

use std::path::Path;

use ffmpeg_next::{
	self as ffmpeg,
	filter,
	format,
	frame,
	media,
};
use tracing::{
	debug,
	info,
	trace,
};

use crate::error::{
	DesilenceError,
	Result,
};

/// Information about an audio stream
#[derive(Debug, Clone)]
pub struct AudioStreamInfo {
	/// Stream index in the container
	pub index: usize,
	/// Codec name
	pub codec_name: String,
	/// Sample rate in Hz
	pub sample_rate: u32,
	/// Number of channels
	pub channels: u16,
	/// Stream duration in seconds (if known)
	pub duration: Option<f64>,
	/// Stream title/name (from metadata)
	pub title: Option<String>,
}

/// Information about all streams in a file
#[derive(Debug)]
pub struct StreamInfo {
	/// All audio streams
	pub audio_streams: Vec<AudioStreamInfo>,
	/// Video stream index (if any)
	pub video_stream_index: Option<usize>,
	/// Total duration in seconds (if known)
	pub duration: Option<f64>,
}

/// Get information about streams in a file
/// Get information about streams in a file
pub fn get_stream_info(ictx: &format::context::Input) -> Result<StreamInfo> {
	let mut audio_streams = Vec::new();
	let mut video_stream_index = None;

	for stream in ictx.streams() {
		let params = stream.parameters();
		let medium = params.medium();

		match medium {
			media::Type::Audio => {
				let codec = ffmpeg::codec::context::Context::from_parameters(params)?;
				let decoder = codec.decoder().audio()?;

				let duration = if stream.duration() > 0 {
					Some(stream.duration() as f64 * f64::from(stream.time_base()))
				} else {
					None
				};

				let metadata = stream.metadata();
				let title = metadata
					.get("title")
					.or_else(|| metadata.get("name"))
					.or_else(|| metadata.get("language"))
					.map(|s| s.to_string());

				audio_streams.push(AudioStreamInfo {
					index: stream.index(),
					codec_name: decoder
						.codec()
						.map(|c| c.name().to_string())
						.unwrap_or_else(|| "unknown".to_string()),
					sample_rate: decoder.rate(),
					channels: decoder.channels(),
					duration,
					title,
				});
			},
			media::Type::Video => {
				if video_stream_index.is_none() {
					video_stream_index = Some(stream.index());
				}
			},
			_ => {},
		}
	}

	let duration = if ictx.duration() > 0 {
		Some(ictx.duration() as f64 / f64::from(ffmpeg::ffi::AV_TIME_BASE))
	} else {
		None
	};

	Ok(StreamInfo {
		audio_streams,
		video_stream_index,
		duration,
	})
}

/// Print stream information to stderr
pub fn print_stream_info(info: &StreamInfo) {
	eprintln!("Stream Information:");
	eprintln!("==================");

	if let Some(duration) = info.duration {
		eprintln!("Total duration: {:.2}s", duration);
	}

	if let Some(idx) = info.video_stream_index {
		eprintln!("Video stream: index {}", idx);
	} else {
		eprintln!("Video stream: none");
	}

	eprintln!("\nAudio streams ({}):", info.audio_streams.len());
	for (i, audio) in info.audio_streams.iter().enumerate() {
		let title_part = audio
			.title
			.as_ref()
			.map(|t| format!(" Title: \"{}\"", t))
			.unwrap_or_default();

		eprintln!(
			"  [{}] index={}, codec={}, {}Hz, {} channels{},{}",
			i,
			audio.index,
			audio.codec_name,
			audio.sample_rate,
			audio.channels,
			audio.duration.map(|d| format!(", {:.2}s", d)).unwrap_or_default(),
			title_part
		);
	}
}

/// Result of silence detection
#[derive(Debug)]
pub struct SilenceDetectionResult {
	/// Detected silence segments as (start, end) pairs
	/// End is None if silence continues to end of file
	pub silence_segments: Vec<(f64, Option<f64>)>,
	/// Total duration of detected silence in seconds
	pub total_silence_duration: f64,
	/// Duration of input file in seconds (if known)
	pub input_duration: Option<f64>,
}

/// Detect silence in audio stream using ffmpeg's silencedetect filter.
///
/// # Arguments
/// * `path` - Path to input video file
/// * `noise_threshold` - Silence threshold (e.g., "-50dB")
/// * `duration` - Minimum silence duration in seconds
/// * `audio_stream_index` - Which audio stream to analyze (None = first)
/// * `merge_audio` - If true, merge all audio streams before detection
pub fn detect_silence<P: AsRef<Path>>(
	path: P,
	noise_threshold: &str,
	duration: f64,
	audio_stream_index: Option<usize>,
	merge_audio: Option<Vec<usize>>,
) -> Result<SilenceDetectionResult> {
	let path = path.as_ref();
	info!(path = %path.display(), "Starting silence detection");

	if !path.exists() {
		return Err(DesilenceError::InputNotFound {
			path: path.to_path_buf(),
		});
	}

	let mut ictx = format::input(path)?;

	// Find the audio streams to use
	let stream_info = get_stream_info(&ictx)?;

	if stream_info.audio_streams.is_empty() {
		return Err(DesilenceError::NoAudioStream);
	}

	// Determine target streams
	let target_streams: Vec<&AudioStreamInfo> = if let Some(indices) = merge_audio {
		if indices.is_empty() {
			// Merge all audio streams
			stream_info.audio_streams.iter().collect()
		} else {
			// Merge specific streams
			let mut selected = Vec::new();
			for &idx in &indices {
				let stream = stream_info
					.audio_streams
					.iter()
					.find(|s| s.index == idx)
					.ok_or(DesilenceError::InvalidAudioStreamIndex {
						index: idx,
						count: stream_info.audio_streams.len(),
					})?;
				selected.push(stream);
			}
			selected
		}
	} else if let Some(idx) = audio_stream_index {
		// Single stream mode
		let stream = stream_info
			.audio_streams
			.iter()
			.find(|s| s.index == idx)
			.ok_or(DesilenceError::InvalidAudioStreamIndex {
				index: idx,
				count: stream_info.audio_streams.len(),
			})?;
		vec![stream]
	} else {
		// Default: first stream
		vec![&stream_info.audio_streams[0]]
	};



	info!(
		count = target_streams.len(),
		indices = ?target_streams.iter().map(|s| s.index).collect::<Vec<_>>(),
		"Analyzing audio streams"
	);

	// Parse threshold once at the start
	let threshold_linear = parse_threshold(noise_threshold)?;

	// Setup decoders map
	use std::collections::HashMap;
	let mut decoders: HashMap<usize, ffmpeg::decoder::Audio> = HashMap::new();
	
	for target in &target_streams {
		let stream = ictx.stream(target.index).unwrap();
		let context = ffmpeg::codec::context::Context::from_parameters(stream.parameters())?;
		let decoder = context.decoder().audio()?;
		decoders.insert(target.index, decoder);
	}

	// use a filter graph to unify the code path.
	let mut graph = filter::Graph::new();
	
	graph.add(
		&filter::find("abuffersink").ok_or_else(|| DesilenceError::FilterGraph {
			message: "abuffersink filter not found".to_string(),
		})?,
		"out",
		"",
	)?;

	for target in &target_streams {
		let decoder = decoders.get(&target.index).unwrap();
		let args = format!(
			"time_base={}:sample_rate={}:sample_fmt={}:channel_layout=0x{:x}",
			ictx.stream(target.index).unwrap().time_base(),
			decoder.rate(),
			decoder.format().name(),
			decoder.channel_layout().bits()
		);
		
		let name = format!("in_{}", target.index);
		graph.add(
			&filter::find("abuffer").ok_or_else(|| DesilenceError::FilterGraph {
				message: "abuffer filter not found".to_string(),
			})?,
			&name,
			&args,
		)?;
	}

	// For single streams, link directly to sink, for multiple streams, use amerge.
	if target_streams.len() > 1 {
		let amerge_name = "merge";
		graph.add(
			&filter::find("amerge").ok_or_else(|| DesilenceError::FilterGraph {
				message: "amerge filter not found".to_string(),
			})?,
			amerge_name,
			&format!("inputs={}", target_streams.len()),
		)?;

		let mut sink = graph.get("out").unwrap();
		let mut amerge = graph.get(amerge_name).unwrap();

		amerge.link(0, &mut sink, 0);

		for (i, target) in target_streams.iter().enumerate() {
			let name = format!("in_{}", target.index);
			let mut source = graph.get(&name).unwrap();
			source.link(0, &mut amerge, i as u32);
		}
	} else {
		let target = target_streams[0];
		let name = format!("in_{}", target.index);
		
		let mut sink = graph.get("out").unwrap();
		let mut source = graph.get(&name).unwrap();
		
		source.link(0, &mut sink, 0);
	}

	graph.validate().map_err(|e| DesilenceError::FilterGraph {
		message: format!("Filter graph validation failed: {}", e),
	})?;

	let mut sink = graph.get("out").ok_or_else(|| DesilenceError::FilterGraph {
		message: "Sink 'out' not found after graph validation".to_string(),
	})?;
	
	// Pair every decoder with its corresponding graph source context.
	let mut processors: HashMap<usize, (ffmpeg::decoder::Audio, ffmpeg::filter::Context)> = HashMap::with_capacity(decoders.len());
	for (index, decoder) in decoders {
		let name = format!("in_{}", index);
		let source = graph.get(&name).ok_or_else(|| DesilenceError::FilterGraph {
			message: format!("Source {} not found after graph validation", name),
		})?;
		processors.insert(index, (decoder, source));
	}

	// Process loop
	let mut silence_segments: Vec<(f64, Option<f64>)> = Vec::new();
	let mut current_silence_start: Option<f64> = None;
	let mut total_silence_duration = 0.0;
	let mut frame_count = 0u64;

	// The silencedetect filter in ffmpeg is convoluted to use as a library.
	//
	// But we can simply implement our own silence detection by analyzing audio levels.
	// This gives the added benefit of being able to use more advanced audio processing
	// techniques if needed.

	info!("Processing audio frames for silence detection...");

	for (stream_iter, packet) in ictx.packets() {
		// Fast lookup: do we have a processor (decoder + source) for this stream index?
		if let Some((decoder, source)) = processors.get_mut(&stream_iter.index()) {
			decoder.send_packet(&packet)?;
			
			let mut decoded = frame::Audio::empty();
			while decoder.receive_frame(&mut decoded).is_ok() {
				source.source().add(&decoded).map_err(|e| DesilenceError::FilterGraph {
					message: format!("Failed to add frame to source: {}", e),
				})?;

				let mut filtered = frame::Audio::empty();
				let sink_time_base = sink.sink().time_base();

				while sink.sink().frame(&mut filtered).is_ok() {
					process_frame(
						&filtered,
						&mut frame_count,
						threshold_linear,
						duration,
						sink_time_base,
						&mut current_silence_start,
						&mut silence_segments,
						&mut total_silence_duration
					);
				}
			}
		}
	}

	// Flush decoders
	for (_, (decoder, source)) in processors.iter_mut() {
		decoder.send_eof()?;
		let mut decoded = frame::Audio::empty();
		while decoder.receive_frame(&mut decoded).is_ok() {
			source.source().add(&decoded).ok(); // Best effort flush

			let mut filtered = frame::Audio::empty();
			while sink.sink().frame(&mut filtered).is_ok() {
				process_frame(
					&filtered,
					&mut frame_count,
					threshold_linear,
					duration,
					sink.sink().time_base(),
					&mut current_silence_start,
					&mut silence_segments,
					&mut total_silence_duration
				);
			}
		}
		
		// Flush filter sources
		source.source().flush().ok();
	}
	
	// Final flush of the graph sink
	let mut filtered = frame::Audio::empty();
	while sink.sink().frame(&mut filtered).is_ok() {
		process_frame(
			&filtered,
			&mut frame_count,
			threshold_linear,
			duration,
			sink.sink().time_base(),
			&mut current_silence_start,
			&mut silence_segments,
			&mut total_silence_duration
		);
	}

	// Handle case where file ends in silence
	if let Some(start) = current_silence_start {
		let silence_duration = stream_info.duration.map(|d| d - start);
		if silence_duration.is_none_or(|d| d >= duration) {
			silence_segments.push((start, stream_info.duration));
			if let Some(d) = silence_duration {
				total_silence_duration += d;
			}
			debug!(start, end = ?stream_info.duration, "Detected silence at end of file");
		}
	}

	info!(
		segments = silence_segments.len(),
		total_duration = total_silence_duration,
		frames_processed = frame_count,
		"Silence detection complete"
	);

	Ok(SilenceDetectionResult {
		silence_segments,
		total_silence_duration,
		input_duration: stream_info.duration,
	})
}

// Helper to avoid duplicate code
#[allow(clippy::too_many_arguments)]
fn process_frame(
	frame: &frame::Audio,
	count: &mut u64,
	threshold: f64,
	min_duration: f64,
	time_base: ffmpeg::Rational,
	current_start: &mut Option<f64>,
	segments: &mut Vec<(f64, Option<f64>)>,
	total_duration: &mut f64
) {
	*count += 1;
	let pts = frame.pts().unwrap_or(0);
	let timestamp = pts as f64 * f64::from(time_base);
	let is_silent = is_frame_silent(frame, threshold);

	match (*current_start, is_silent) {
		(None, true) => {
			*current_start = Some(timestamp);
			trace!(timestamp, "Silence started");
		},
		(Some(start), false) => {
			let silence_duration = timestamp - start;
			if silence_duration >= min_duration {
				segments.push((start, Some(timestamp)));
				*total_duration += silence_duration;
				debug!(start, end = timestamp, duration = silence_duration, "Detected silence segment");
			}
			*current_start = None;
		},
		_ => {},
	}
}

/// Check if an audio frame is silent based on threshold.
/// This is a simplified implementation that checks max amplitude.
///
/// # Arguments
/// * `frame` - The audio frame to analyze
/// * `threshold_linear` - The linear amplitude threshold (pre-converted from dB)
fn is_frame_silent(frame: &frame::Audio, threshold_linear: f64) -> bool {
	// Get audio samples and calculate max amplitude
	// Handle different sample formats
	let plane_count = frame.planes();
	if plane_count == 0 {
		return true;
	}

	// Simple approach: check if any sample exceeds threshold
	// We'll work with the first plane (works for both planar and interleaved)
	let samples = frame.samples();
	if samples == 0 {
		return true;
	}

	// Try to read as f32 planar (common format after filtering)
	let data = frame.data(0);
	if data.is_empty() {
		return true;
	}

	// Calculate max amplitude for this frame based on sample format
	let format = frame.format();
	let max_amplitude = match format {
		format::Sample::I16(_) => data
			.chunks_exact(2)
			.map(|c| i16::from_ne_bytes(c.try_into().unwrap()))
			.map(|s| (s as f64 / i16::MAX as f64).abs())
			.fold(0.0_f64, |a, b| a.max(b)),
		format::Sample::I32(_) => data
			.chunks_exact(4)
			.map(|c| i32::from_ne_bytes(c.try_into().unwrap()))
			.map(|s| (s as f64 / i32::MAX as f64).abs())
			.fold(0.0_f64, |a, b| a.max(b)),
		format::Sample::F32(_) => data
			.chunks_exact(4)
			.map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
			.map(|s| (s as f64).abs())
			.fold(0.0_f64, |a, b| a.max(b)),
		format::Sample::F64(_) => data
			.chunks_exact(8)
			.map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
			.map(|s| s.abs())
			.fold(0.0_f64, |a, b| a.max(b)),
		format::Sample::U8(_) => {
			// U8 is centered at 128
			data
				.iter()
				.map(|&s| ((s as f64 - 128.0) / 128.0).abs())
				.fold(0.0_f64, |a, b| a.max(b))
		},
		_ => {
			// Unknown format, assume not silent to avoid masking content
			return false;
		},
	};

	max_amplitude < threshold_linear
}

/// Parse a dB threshold string (e.g., "-50dB" or "-50") into linear amplitude.
///
/// Returns the linear amplitude value (0.0 to 1.0 scale).
pub fn parse_threshold(threshold: &str) -> Result<f64> {
	let threshold_str = threshold
		.trim()
		.trim_end_matches("dB")
		.trim_end_matches("db")
		.trim_end_matches("DB");

	let threshold_db: f64 = threshold_str.parse().map_err(|_| DesilenceError::InvalidThreshold {
		value: threshold.to_string(),
	})?;

	// Convert dB to linear amplitude: 10^(dB/20)
	let threshold_linear = 10.0_f64.powf(threshold_db / 20.0);

	debug!(
		threshold_db = threshold_db,
		threshold_linear = threshold_linear,
		"Parsed noise threshold"
	);

	Ok(threshold_linear)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_threshold_parsing_valid() {
		// Test various valid threshold formats
		let linear = parse_threshold("-50dB").unwrap();
		assert!((linear - 0.00316).abs() < 0.001);

		let linear = parse_threshold("-50db").unwrap();
		assert!((linear - 0.00316).abs() < 0.001);

		let linear = parse_threshold("-50").unwrap();
		assert!((linear - 0.00316).abs() < 0.001);

		let linear = parse_threshold("  -30dB  ").unwrap();
		assert!((linear - 0.0316).abs() < 0.001);
	}

	#[test]
	fn test_threshold_parsing_invalid() {
		// Test invalid threshold formats return errors
		assert!(parse_threshold("invalid").is_err());
		assert!(parse_threshold("dB").is_err());
		assert!(parse_threshold("abc123").is_err());
	}
}
