//! Error types for desilence-rs

use std::path::PathBuf;

use miette::Diagnostic;

/// Result type alias using DesilenceError
pub type Result<T> = std::result::Result<T, DesilenceError>;

/// Main error type for desilence-rs operations
#[derive(Debug, Diagnostic, thiserror::Error)]
pub enum DesilenceError {
	/// FFmpeg library error
	#[error("FFmpeg error: {message}")]
	#[diagnostic(code(desilence::ffmpeg))]
	Ffmpeg {
		message: String,
		#[source]
		source: Option<ffmpeg_next::Error>,
	},

	/// Input file not found or inaccessible
	#[error("Input file not found: {path}")]
	#[diagnostic(
		code(desilence::input_not_found),
		help("Ensure the file exists and you have read permissions")
	)]
	InputNotFound { path: PathBuf },

	/// No audio stream found in input
	#[error("No audio stream found in input file")]
	#[diagnostic(
		code(desilence::no_audio),
		help("The input file must contain at least one audio stream for silence detection")
	)]
	NoAudioStream,

	/// Invalid audio stream index
	#[error("Audio stream index {index} not found (file has {count} audio streams)")]
	#[diagnostic(code(desilence::invalid_stream), help("Use -l/--list-streams to see available streams"))]
	InvalidAudioStreamIndex { index: usize, count: usize },

	/// No video stream found in input
	#[error("No video stream found in input file")]
	#[diagnostic(code(desilence::no_video))]
	NoVideoStream,

	/// No silence detected
	#[error("No silence segments detected in input")]
	#[diagnostic(
		code(desilence::no_silence),
		help("Try adjusting --noise-threshold or --duration parameters")
	)]
	NoSilenceDetected,

	/// No audible segments after inversion
	#[error("No audible segments found - entire file appears to be silent")]
	#[diagnostic(code(desilence::all_silent), help("Try lowering the --noise-threshold value"))]
	NoAudibleSegments,

	/// Filter graph error
	#[error("Failed to create filter graph: {message}")]
	#[diagnostic(code(desilence::filter))]
	FilterGraph { message: String },

	/// Output error
	#[error("Failed to write output: {message}")]
	#[diagnostic(code(desilence::output))]
	Output { message: String },

	/// Segment processing error
	#[error("Segment processing error: {message}")]
	#[diagnostic(code(desilence::segment))]
	Segment { message: String },

	/// Invalid noise threshold format
	#[error("Invalid noise threshold format: {value}")]
	#[diagnostic(
		code(desilence::invalid_threshold),
		help("Threshold should be a negative number in dB, e.g., '-50dB' or '-50'")
	)]
	InvalidThreshold { value: String },

	/// I/O error
	#[error("I/O error: {0}")]
	#[diagnostic(code(desilence::io))]
	Io(#[from] std::io::Error),
}

impl From<ffmpeg_next::Error> for DesilenceError {
	fn from(err: ffmpeg_next::Error) -> Self {
		DesilenceError::Ffmpeg {
			message: err.to_string(),
			source: Some(err),
		}
	}
}
