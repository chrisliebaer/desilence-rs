//! Silence detection using FFmpeg's silencedetect filter

use std::path::Path;

use ffmpeg_next::{
	self as ffmpeg,
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
pub fn get_stream_info<P: AsRef<Path>>(path: P) -> Result<StreamInfo> {
	let path = path.as_ref();

	if !path.exists() {
		return Err(DesilenceError::InputNotFound {
			path: path.to_path_buf(),
		});
	}

	let ictx = format::input(path)?;

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

				audio_streams.push(AudioStreamInfo {
					index: stream.index(),
					codec_name: decoder
						.codec()
						.map(|c| c.name().to_string())
						.unwrap_or_else(|| "unknown".to_string()),
					sample_rate: decoder.rate(),
					channels: decoder.channels(),
					duration,
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
		eprintln!(
			"  [{}] index={}, codec={}, {}Hz, {} channels{}",
			i,
			audio.index,
			audio.codec_name,
			audio.sample_rate,
			audio.channels,
			audio.duration.map(|d| format!(", {:.2}s", d)).unwrap_or_default()
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
	merge_audio: bool,
) -> Result<SilenceDetectionResult> {
	let path = path.as_ref();
	info!(path = %path.display(), "Starting silence detection");

	if !path.exists() {
		return Err(DesilenceError::InputNotFound {
			path: path.to_path_buf(),
		});
	}

	let mut ictx = format::input(path)?;

	// Find the audio stream to use
	let stream_info = get_stream_info(path)?;

	if stream_info.audio_streams.is_empty() {
		return Err(DesilenceError::NoAudioStream);
	}

	let target_stream = if merge_audio {
		warn!("Merge audio mode not yet implemented, using first stream");
		// TODO: Implement audio stream merging using amerge filter
		&stream_info.audio_streams[0]
	} else if let Some(idx) = audio_stream_index {
		stream_info
			.audio_streams
			.iter()
			.find(|s| s.index == idx)
			.ok_or(DesilenceError::InvalidAudioStreamIndex {
				index: idx,
				count: stream_info.audio_streams.len(),
			})?
	} else {
		&stream_info.audio_streams[0]
	};

	info!(
			stream_index = target_stream.index,
			codec = %target_stream.codec_name,
			sample_rate = target_stream.sample_rate,
			channels = target_stream.channels,
			"Using audio stream"
	);

	// Set up decoder for the audio stream
	let stream = ictx.stream(target_stream.index).unwrap();
	let context = ffmpeg::codec::context::Context::from_parameters(stream.parameters())?;
	let mut decoder = context.decoder().audio()?;


	// Parse threshold once at the start for proper error handling
	let threshold_linear = parse_threshold(noise_threshold)?;

	// Process audio frames and collect silence segments
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

	let in_time_base = stream.time_base();

	for (stream_iter, packet) in ictx.packets() {
		if stream_iter.index() != target_stream.index {
			continue;
		}

		decoder.send_packet(&packet)?;

		let mut decoded = frame::Audio::empty();
		while decoder.receive_frame(&mut decoded).is_ok() {
			frame_count += 1;

			// Get timestamp
			let pts = decoded.pts().unwrap_or(0);
			let timestamp = pts as f64 * f64::from(in_time_base);

			// Analyze audio level using pre-parsed threshold
			let is_silent = is_frame_silent(&decoded, threshold_linear);

			trace!(
				frame = frame_count,
				pts = pts,
				timestamp = timestamp,
				silent = is_silent,
				"Processed audio frame"
			);

			// Track silence segments
			match (current_silence_start, is_silent) {
				(None, true) => {
					// Starting silence
					current_silence_start = Some(timestamp);
					trace!(timestamp, "Silence started");
				},
				(Some(start), false) => {
					// Ending silence
					let silence_duration = timestamp - start;
					if silence_duration >= duration {
						silence_segments.push((start, Some(timestamp)));
						total_silence_duration += silence_duration;
						debug!(
							start,
							end = timestamp,
							duration = silence_duration,
							"Detected silence segment"
						);
					}
					current_silence_start = None;
				},
				_ => {},
			}
		}
	}

	// Flush decoder
	decoder.send_eof()?;
	let mut decoded = frame::Audio::empty();
	while decoder.receive_frame(&mut decoded).is_ok() {
		frame_count += 1;
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
