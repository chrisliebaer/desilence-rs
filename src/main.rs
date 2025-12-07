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
use tracing::level_filters::LevelFilter;
use tracing_subscriber::{
	fmt,
	prelude::*,
	EnvFilter,
};
use ffmpeg_next as ffmpeg;

fn main() -> Result<()> {
	// Handle --dump-help flag before parsing since regular invocation would require all flags to be valid
	if std::env::args().any(|arg| arg == "--dump-help") {
		use clap::CommandFactory;
		let mut cmd = Args::command();
		print!("{}", cmd.render_help());
		std::process::exit(0);
	}

	// Parse CLI arguments
	let args = Args::parse();

	// Mute FFmpeg raw logs (e.g. silencedetect info) so they don't clutter stderr,
	// unless verbose logging is requested.
	if args.log_level() < LevelFilter::DEBUG {
		ffmpeg::log::set_level(ffmpeg::log::Level::Warning);
	}

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

	// Open input file
	let ictx = ffmpeg::format::input(&args.input)?;

	// Get stream information
	let stream_info = silence::get_stream_info(&ictx)?;

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
	// Determine target streams for detection (Absolute Indices)
	let target_streams = if let Some(indices) = &args.merge_audio {
		// Merge mode
		if indices.is_empty() {
			// Merge ALL streams
			let all_indices: Vec<usize> = stream_info.audio_streams.iter().map(|s| s.index).collect();
			Some(all_indices)
		} else {
			// Merge specific streams (relative -> absolute)
			let mut absolute_indices = Vec::new();
			for &idx in indices {
				let stream = stream_info.audio_streams.get(idx).ok_or(DesilenceError::InvalidAudioStreamIndex {
					index: idx,
					count: stream_info.audio_streams.len(),
				})?;
				absolute_indices.push(stream.index);
			}
			Some(absolute_indices)
		}
	} else if let Some(idx) = args.audio_stream {
		// Single stream mode (relative -> absolute)
		let stream = stream_info.audio_streams.get(idx).ok_or(DesilenceError::InvalidAudioStreamIndex {
			index: idx,
			count: stream_info.audio_streams.len(),
		})?;
		Some(vec![stream.index])
	} else {
		// Default (Automatic/First)
		None
	};

	// Determine primary detection stream for pipeline config (first involved stream)
	let detection_stream_index = if let Some(ref streams) = target_streams {
		if streams.is_empty() {
			// Should be caught by detect_silence or empty list check, but fallback to first
			stream_info.audio_streams[0].index
		} else {
			streams[0]
		}
	} else {
		stream_info.audio_streams[0].index
	};

	// Detect silence segments
	let detection_result = silence::detect_silence(
		&args.input,
		&args.noise_threshold,
		args.duration,
		target_streams,
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

	// Prevent writing binary data to terminal unless forced
	if args.output.is_none() {
		use std::io::IsTerminal;
		if std::io::stdout().is_terminal() && !args.force {
			// Print stream info to be helpful
			use silence::print_stream_info;
			print_stream_info(&stream_info);
			return Err(DesilenceError::TerminalOutput);
		}
	}

	// Run the streaming pipeline
	let config = PipelineConfig {
		detection_stream_index: detection_stream_index,
		output_all_streams: true, // Output all audio streams (NUT supports it)
	};

	let stats = pipeline::run_pipeline(&args.input, args.output.as_deref(), &segments, &config)?;

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
