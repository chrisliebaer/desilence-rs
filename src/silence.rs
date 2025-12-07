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
	warn,
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
	eprintln!("Available Audio Streams:");

	for (i, audio) in info.audio_streams.iter().enumerate() {
		let title_part = audio.title.as_ref().map(|t| format!(" \"{}\"", t)).unwrap_or_default();

		eprintln!(
			"  [{}] {}{}, {}Hz, {} channels{}",
			i,
			audio.codec_name,
			title_part,
			audio.sample_rate,
			audio.channels,
			audio.duration.map(|d| format!(", {:.2}s", d)).unwrap_or_default(),
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
/// * `target_streams` - Specific audio stream indices (absolute) to analyze. If None, the first audio stream is
///   selected automatically. If multiple are provided, they are merged.
pub fn detect_silence<P: AsRef<Path>>(
	path: P,
	noise_threshold: &str,
	duration: f64,
	target_streams: Option<Vec<usize>>,
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
	let selected_streams: Vec<&AudioStreamInfo> = if let Some(indices) = target_streams {
		if indices.is_empty() {
			return Err(DesilenceError::InvalidAudioStreamIndex {
				index: 0,
				count: 0,
			});
		}
		let mut selected = Vec::new();
		for idx in indices {
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
	} else {
		// Default: Use the first available audio stream
		vec![&stream_info.audio_streams[0]]
	};

	// Format stream names for logging
	let stream_names: Vec<String> = selected_streams
		.iter()
		.map(|s| {
			let title = s.title.as_deref().unwrap_or("untitled");
			format!("Audio Stream #{} \"{}\"", s.index, title)
		})
		.collect();

	info!("Analyzing streams: {}", stream_names.join(", "));

	// Parse threshold into numeric dB value
	let threshold_db = parse_threshold(noise_threshold)?;

	// Setup decoders map
	use std::collections::HashMap;
	let mut decoders: HashMap<usize, ffmpeg::decoder::Audio> = HashMap::new();

	for target in &selected_streams {
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

	for target in &selected_streams {
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

	// Add silencedetect filter
	let silencedetect_args = format!("noise={}dB:d={}", threshold_db, duration);
	let silencedetect_name = "silencedetect";
	graph.add(
		&filter::find("silencedetect").ok_or_else(|| DesilenceError::FilterGraph {
			message: "silencedetect filter not found".to_string(),
		})?,
		silencedetect_name,
		&silencedetect_args,
	)?;

	// Connect silencedetect -> sink
	let mut sink = graph.get("out").unwrap();
	let mut silencedetect = graph.get(silencedetect_name).unwrap();
	silencedetect.link(0, &mut sink, 0);

	// Connect inputs -> [Amerge] -> silencedetect
	if selected_streams.len() > 1 {
		let amerge_name = "merge";
		graph.add(
			&filter::find("amerge").ok_or_else(|| DesilenceError::FilterGraph {
				message: "amerge filter not found".to_string(),
			})?,
			amerge_name,
			&format!("inputs={}", selected_streams.len()),
		)?;

		let mut amerge = graph.get(amerge_name).unwrap();
		amerge.link(0, &mut silencedetect, 0);

		for (i, target) in selected_streams.iter().enumerate() {
			let name = format!("in_{}", target.index);
			let mut source = graph.get(&name).unwrap();
			source.link(0, &mut amerge, i as u32);
		}
	} else {
		let target = selected_streams[0];
		let name = format!("in_{}", target.index);
		let mut source = graph.get(&name).unwrap();
		source.link(0, &mut silencedetect, 0);
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

	info!("Processing audio frames for silence detection via silencedetect filter...");

	// We'll use a closure to process filter output frames
	let mut process_filter_output = |sink: &mut ffmpeg::filter::Context| -> Result<()> {
		let mut filtered = frame::Audio::empty();
		while sink.sink().frame(&mut filtered).is_ok() {
			frame_count += 1;

			let metadata = filtered.metadata();

			// Check for silence start
			if let Some(start_str) = metadata.get("lavfi.silence_start") {
				if let Ok(start) = start_str.parse::<f64>() {
					if current_silence_start.is_none() {
						current_silence_start = Some(start);
						trace!(start, "Silence start flagged");
					}
				}
			}

			// Check for silence duration (end of silence)
			if let Some(duration_str) = metadata.get("lavfi.silence_duration") {
				if let Ok(duration) = duration_str.parse::<f64>() {
					if let Some(start) = current_silence_start {
						silence_segments.push((start, Some(start + duration)));
						total_silence_duration += duration;
						debug!(start, duration, "Detected silence segment");
						current_silence_start = None;
					} else {
						// Found duration without start.
						warn!("Found silence duration without start track at frame {}", frame_count);
					}
				}
			}
		}
		Ok(())
	};

	for (stream_iter, packet) in ictx.packets() {
		// Fast lookup: do we have a processor (decoder + source) for this stream index?
		if let Some((decoder, source)) = processors.get_mut(&stream_iter.index()) {
			decoder.send_packet(&packet)?;

			let mut decoded = frame::Audio::empty();
			while decoder.receive_frame(&mut decoded).is_ok() {
				source.source().add(&decoded).map_err(|e| DesilenceError::FilterGraph {
					message: format!("Failed to add frame to source: {}", e),
				})?;

				process_filter_output(&mut sink)?;
			}
		}
	}

	// Flush decoders
	for (_, (decoder, source)) in processors.iter_mut() {
		decoder.send_eof()?;
		let mut decoded = frame::Audio::empty();
		while decoder.receive_frame(&mut decoded).is_ok() {
			source.source().add(&decoded).ok(); // Best effort flush

			process_filter_output(&mut sink)?;
		}

		// Flush filter sources
		source.source().flush().ok();
	}

	// Final flush of the graph sink
	process_filter_output(&mut sink)?;

	// Handle case where file ends in silence (trailing silence)
	if let Some(start) = current_silence_start {
		let calc_duration = if let Some(d) = stream_info.duration {
			(d - start).max(0.0)
		} else {
			0.0
		};

		if calc_duration >= duration {
			silence_segments.push((start, stream_info.duration));
			total_silence_duration += calc_duration;
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

/// Parse a dB threshold string (e.g., "-50dB" or "-50") into numeric dB value.
///
/// Returns the dB value (e.g. -50.0).
pub fn parse_threshold(threshold: &str) -> Result<f64> {
	let threshold_str = threshold
		.trim()
		.trim_end_matches("dB")
		.trim_end_matches("db")
		.trim_end_matches("DB");

	let threshold_db: f64 = threshold_str.parse().map_err(|_| DesilenceError::InvalidThreshold {
		value: threshold.to_string(),
	})?;

	debug!(threshold_db = threshold_db, "Parsed noise threshold");

	Ok(threshold_db)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_threshold_parsing_valid() {
		// Test various valid threshold formats
		let linear = parse_threshold("-50dB").unwrap();
		assert!((linear - -50.0).abs() < 0.001);

		let linear = parse_threshold("-50db").unwrap();
		assert!((linear - -50.0).abs() < 0.001);

		let linear = parse_threshold("-50").unwrap();
		assert!((linear - -50.0).abs() < 0.001);

		let linear = parse_threshold("  -30dB  ").unwrap();
		assert!((linear - -30.0).abs() < 0.001);
	}

	#[test]
	fn test_threshold_parsing_invalid() {
		// Test invalid threshold formats return errors
		assert!(parse_threshold("invalid").is_err());
		assert!(parse_threshold("dB").is_err());
		assert!(parse_threshold("abc123").is_err());
	}
}
