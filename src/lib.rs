//! desilence-rs: Remove silent segments from video files
//!
//! This crate provides functionality to detect and remove silent segments from video files,
//! streaming the output to stdout in NUT format for consumption by an external ffmpeg encoder.

pub mod cli;
pub mod error;
pub mod pipeline;
pub mod segment;
pub mod silence;

pub use error::{
	DesilenceError,
	Result,
};
