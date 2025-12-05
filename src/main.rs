//! desilence-rs: Remove silent segments from video files
//!
//! Streams output to stdout in NUT format for consumption by ffmpeg.

use clap::Parser;
use desilence_rs::{
	cli::Args,
	error::DesilenceError,
	pipeline::{
		self,
		PipelineConfig,
	},
	segment,
	silence::{
		self,
		print_stream_info,
	},
};
use miette::Result;
use tracing::{
	info,
	warn,
};
use tracing_subscriber::{
	fmt,
	prelude::*,
	EnvFilter,
};

fn main() -> Result<()> {
	// Parse CLI arguments
	let args = Args::parse();

	// Initialize tracing
	let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(args.log_level().to_string()));

	tracing_subscriber::registry()
		.with(fmt::layer().with_writer(std::io::stderr))
		.with(filter)
		.init();

	// Run the application
	run(args).map_err(miette::Report::new)
}

fn run(args: Args) -> std::result::Result<(), DesilenceError> {
	// Validate input file exists
	if !args.input.exists() {
		return Err(DesilenceError::InputNotFound {
			path: args.input.clone(),
		});
	}

	info!(input = %args.input.display(), "Starting desilence-rs");

	// Get stream information
	let stream_info = silence::get_stream_info(&args.input)?;

	// Handle --list-streams
	if args.list_streams {
		print_stream_info(&stream_info);
		return Ok(());
	}

	// Validate audio stream exists
	if stream_info.audio_streams.is_empty() {
		return Err(DesilenceError::NoAudioStream);
	}

	// Determine which audio stream to use for detection
	let detection_stream = if let Some(idx) = args.audio_stream {
		// User specified stream index - find it
		stream_info
			.audio_streams
			.iter()
			.position(|s| s.index == idx)
			.map(|_| idx)
			.ok_or(DesilenceError::InvalidAudioStreamIndex {
				index: idx,
				count: stream_info.audio_streams.len(),
			})?
	} else {
		// Use first audio stream
		stream_info.audio_streams[0].index
	};

	// Validate threshold format early for better error messages
	// (detect_silence will also validate, but we want to fail fast)
	silence::parse_threshold(&args.noise_threshold)?;

	info!(
			stream = detection_stream,
			threshold = %args.noise_threshold,
			min_duration = args.duration,
			"Detecting silence"
	);

	// Detect silence segments
	let detection_result = silence::detect_silence(
		&args.input,
		&args.noise_threshold,
		args.duration,
		Some(detection_stream),
		args.merge_audio,
	)?;

	// Check if any silence was detected
	if detection_result.silence_segments.is_empty() {
		warn!("No silence segments detected in input");
		return Err(DesilenceError::NoSilenceDetected);
	}

	info!(
		silence_segments = detection_result.silence_segments.len(),
		total_silence = format!("{:.2}s", detection_result.total_silence_duration),
		"Silence detection complete"
	);

	// Invert silence segments to get audible segments
	// Use a minimum segment duration of 1/30th second (one frame at 30fps) to avoid
	// very short segments that might cause issues
	let min_segment_duration = 1.0 / 30.0;
	let segments = segment::invert_silence_segments(
		&detection_result.silence_segments,
		detection_result.input_duration,
		min_segment_duration,
	);

	// Validate we have audible content
	if !segments.has_audible() {
		return Err(DesilenceError::NoAudibleSegments);
	}

	info!(
		audible_segments = segments.audible_count(),
		audible_duration = format!("{:.2}s", segments.total_audible_duration),
		silent_duration = format!("{:.2}s", segments.total_silent_duration),
		"Segment analysis complete"
	);

	// Log individual segments at debug level
	for seg in segments.all_segments() {
		tracing::debug!("{}", seg);
	}

	// Run the streaming pipeline
	let config = PipelineConfig {
		detection_stream_index: detection_stream,
		output_all_streams: true, // Output all audio streams (NUT supports it)
	};

	let stats = pipeline::run_pipeline(&args.input, &segments, &config)?;

	info!(
		output_frames = stats.output_frames,
		kept_percentage = format!("{:.1}%", stats.kept_percentage()),
		"Processing complete"
	);

	// Print summary to stderr
	if !args.quiet {
		eprintln!();
		eprintln!("Desilence complete:");
		eprintln!("  Silence segments:  {}", detection_result.silence_segments.len());
		eprintln!("  Silence removed:   {:.2}s", detection_result.total_silence_duration);
		eprintln!("  Audible segments:  {}", segments.audible_count());
		eprintln!("  Audible duration:  {:.2}s", segments.total_audible_duration);
		eprintln!("  Output frames:     {}", stats.output_frames);
		eprintln!("  Content kept:      {:.1}%", stats.kept_percentage());
	}

	Ok(())
}
