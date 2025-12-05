//! CLI argument parsing for desilence-rs

use std::path::PathBuf;

use clap::Parser;

/// Remove silent segments from video files, streaming output to stdout.
///
/// Output is in NUT format with raw video and PCM audio. Pipe to ffmpeg for encoding:
///
///   desilence-rs -i input.mp4 | ffmpeg -f nut -i pipe: -c:v libx264 -c:a aac output.mp4
#[derive(Parser, Debug, Clone)]
#[command(name = "desilence-rs", version, about, long_about = None)]
pub struct Args {
	/// Input video file path
	#[arg(short, long)]
	pub input: PathBuf,

	/// Silence detection threshold in dB (negative value)
	///
	/// Audio below this level is considered silence.
	/// Lower values (more negative) detect quieter sounds as non-silent.
	#[arg(short = 'n', long, default_value = "-50dB", allow_hyphen_values = true)]
	pub noise_threshold: String,

	/// Minimum silence duration in seconds
	///
	/// Silence segments shorter than this are ignored.
	#[arg(short = 'd', long, default_value = "0.5")]
	pub duration: f64,

	/// Audio stream index to use for silence detection (0-based)
	///
	/// By default, uses the first audio stream. Use -l to list available streams.
	#[arg(short = 'a', long)]
	pub audio_stream: Option<usize>,

	/// Merge all audio streams for silence detection
	///
	/// Expert option: combines all audio streams before detection.
	/// Useful when silence might only be present in some channels.
	#[arg(long, conflicts_with = "audio_stream")]
	pub merge_audio: bool,

	/// List available streams and exit
	#[arg(short = 'l', long)]
	pub list_streams: bool,

	/// Verbose output (repeat for more verbosity: -v, -vv, -vvv)
	#[arg(short, long, action = clap::ArgAction::Count)]
	pub verbose: u8,

	/// Quiet mode - suppress all non-error output
	#[arg(short, long, conflicts_with = "verbose")]
	pub quiet: bool,
}

impl Args {
	/// Get the tracing log level filter based on verbosity settings
	pub fn log_level(&self) -> tracing::level_filters::LevelFilter {
		use tracing::level_filters::LevelFilter;

		if self.quiet {
			LevelFilter::ERROR
		} else {
			match self.verbose {
				0 => LevelFilter::WARN,
				1 => LevelFilter::INFO,
				2 => LevelFilter::DEBUG,
				_ => LevelFilter::TRACE,
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_default_values() {
		let args = Args::parse_from(["desilence-rs", "-i", "test.mp4"]);
		assert_eq!(args.noise_threshold, "-50dB");
		assert_eq!(args.duration, 0.5);
		assert!(args.audio_stream.is_none());
		assert!(!args.merge_audio);
		assert!(!args.list_streams);
		assert_eq!(args.verbose, 0);
		assert!(!args.quiet);
	}

	#[test]
	fn test_custom_threshold() {
		let args = Args::parse_from(["desilence-rs", "-i", "test.mp4", "-n", "-30dB"]);
		assert_eq!(args.noise_threshold, "-30dB");
	}

	#[test]
	fn test_audio_stream_selection() {
		let args = Args::parse_from(["desilence-rs", "-i", "test.mp4", "-a", "1"]);
		assert_eq!(args.audio_stream, Some(1));
	}

	#[test]
	fn test_verbosity_levels() {
		let args = Args::parse_from(["desilence-rs", "-i", "test.mp4"]);
		assert_eq!(args.log_level(), tracing::level_filters::LevelFilter::WARN);

		let args = Args::parse_from(["desilence-rs", "-i", "test.mp4", "-v"]);
		assert_eq!(args.log_level(), tracing::level_filters::LevelFilter::INFO);

		let args = Args::parse_from(["desilence-rs", "-i", "test.mp4", "-vv"]);
		assert_eq!(args.log_level(), tracing::level_filters::LevelFilter::DEBUG);

		let args = Args::parse_from(["desilence-rs", "-i", "test.mp4", "-q"]);
		assert_eq!(args.log_level(), tracing::level_filters::LevelFilter::ERROR);
	}
}
