//! Segment representation and manipulation

use std::fmt;

/// Type of audio segment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentType {
	/// Audible content (sound above threshold)
	Audible,
	/// Silent content (sound below threshold)
	Silent,
}

impl fmt::Display for SegmentType {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			SegmentType::Audible => write!(f, "audible"),
			SegmentType::Silent => write!(f, "silent"),
		}
	}
}

/// A time segment within the video
#[derive(Debug, Clone)]
pub struct Segment {
	/// Type of this segment (audible or silent)
	pub segment_type: SegmentType,
	/// Start time in seconds
	pub start: f64,
	/// End time in seconds (None means until end of file)
	pub end: Option<f64>,
}

impl Segment {
	/// Create a new segment
	pub fn new(segment_type: SegmentType, start: f64, end: Option<f64>) -> Self {
		Self {
			segment_type,
			start,
			end,
		}
	}

	/// Get the duration of this segment, if end is known
	pub fn duration(&self) -> Option<f64> {
		self.end.map(|e| e - self.start)
	}

	/// Check if this segment is audible
	pub fn is_audible(&self) -> bool {
		self.segment_type == SegmentType::Audible
	}

	/// Check if a timestamp falls within this segment
	pub fn contains(&self, timestamp: f64) -> bool {
		timestamp >= self.start && self.end.is_none_or(|e| timestamp < e)
	}
}

impl fmt::Display for Segment {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self.end {
			Some(end) => write!(
				f,
				"{}: {:.3}s - {:.3}s (duration: {:.3}s)",
				self.segment_type,
				self.start,
				end,
				end - self.start
			),
			None => write!(f, "{}: {:.3}s - end", self.segment_type, self.start),
		}
	}
}

/// Collection of segments with utility methods
#[derive(Debug, Clone)]
pub struct SegmentList {
	segments: Vec<Segment>,
	/// Total duration of audible segments
	pub total_audible_duration: f64,
	/// Total duration of silent segments
	pub total_silent_duration: f64,
}

impl SegmentList {
	/// Create a new segment list from a vector of segments
	pub fn new(segments: Vec<Segment>) -> Self {
		let mut total_audible = 0.0;
		let mut total_silent = 0.0;

		for seg in &segments {
			if let Some(duration) = seg.duration() {
				match seg.segment_type {
					SegmentType::Audible => total_audible += duration,
					SegmentType::Silent => total_silent += duration,
				}
			}
		}

		Self {
			segments,
			total_audible_duration: total_audible,
			total_silent_duration: total_silent,
		}
	}

	/// Get all audible segments
	pub fn audible_segments(&self) -> impl Iterator<Item = &Segment> {
		self.segments.iter().filter(|s| s.is_audible())
	}

	/// Get all silent segments
	pub fn silent_segments(&self) -> impl Iterator<Item = &Segment> {
		self.segments.iter().filter(|s| !s.is_audible())
	}

	/// Get all segments
	pub fn all_segments(&self) -> &[Segment] {
		&self.segments
	}

	/// Find segment containing a timestamp, returns (segment, is_audible)
	pub fn find_segment(&self, timestamp: f64) -> Option<&Segment> {
		self.segments.iter().find(|s| s.contains(timestamp))
	}

	/// Check if a timestamp is in an audible segment
	pub fn is_audible_at(&self, timestamp: f64) -> bool {
		self.find_segment(timestamp).is_some_and(|s| s.is_audible())
	}

	/// Get count of audible segments
	pub fn audible_count(&self) -> usize {
		self.segments.iter().filter(|s| s.is_audible()).count()
	}

	/// Get count of silent segments
	pub fn silent_count(&self) -> usize {
		self.segments.iter().filter(|s| !s.is_audible()).count()
	}

	/// Check if there are any audible segments
	pub fn has_audible(&self) -> bool {
		self.segments.iter().any(|s| s.is_audible())
	}
}

/// Invert silence segments to get audible segments.
///
/// Given a list of silence segments (start, end), produces a complete
/// segment list alternating between audible and silent segments.
///
/// Handles edge cases:
/// - Video starting with silence
/// - Video ending in audible segment
/// - Segments shorter than min_segment_duration are absorbed into adjacent segments
pub fn invert_silence_segments(
	silence_segments: &[(f64, Option<f64>)],
	video_duration: Option<f64>,
	min_segment_duration: f64,
) -> SegmentList {
	let mut segments = Vec::new();

	if silence_segments.is_empty() {
		// No silence detected - entire video is audible
		segments.push(Segment::new(SegmentType::Audible, 0.0, video_duration));
		return SegmentList::new(segments);
	}

	let mut last_end = 0.0;

	for (silence_start, silence_end) in silence_segments {
		// Add audible segment before this silence (if any)
		if *silence_start > last_end {
			let audible_duration = silence_start - last_end;
			if audible_duration >= min_segment_duration {
				segments.push(Segment::new(SegmentType::Audible, last_end, Some(*silence_start)));
			} else {
				tracing::debug!(
					duration = audible_duration,
					min = min_segment_duration,
					"Skipping short audible segment"
				);
			}
		}

		// Add silent segment
		if let Some(end) = silence_end {
			let silence_duration = end - silence_start;
			segments.push(Segment::new(SegmentType::Silent, *silence_start, Some(*end)));
			last_end = *end;
			tracing::trace!(
				start = silence_start,
				end = end,
				duration = silence_duration,
				"Silent segment"
			);
		} else {
			// Silence continues to end of file
			segments.push(Segment::new(SegmentType::Silent, *silence_start, None));
			return SegmentList::new(segments);
		}
	}

	// Add final audible segment if video continues after last silence
	if let Some(duration) = video_duration {
		if last_end < duration {
			let final_duration = duration - last_end;
			if final_duration >= min_segment_duration {
				segments.push(Segment::new(SegmentType::Audible, last_end, Some(duration)));
			} else {
				tracing::debug!(
					duration = final_duration,
					min = min_segment_duration,
					"Skipping short final audible segment"
				);
			}
		}
	} else {
		// Unknown video duration - assume audible until end
		segments.push(Segment::new(SegmentType::Audible, last_end, None));
	}

	SegmentList::new(segments)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_invert_no_silence() {
		let silence = vec![];
		let segments = invert_silence_segments(&silence, Some(10.0), 0.0);
		assert_eq!(segments.audible_count(), 1);
		assert_eq!(segments.silent_count(), 0);

		let seg = &segments.all_segments()[0];
		assert!(seg.is_audible());
		assert_eq!(seg.start, 0.0);
		assert_eq!(seg.end, Some(10.0));
	}

	#[test]
	fn test_invert_single_silence_middle() {
		let silence = vec![(3.0, Some(5.0))];
		let segments = invert_silence_segments(&silence, Some(10.0), 0.0);

		assert_eq!(segments.audible_count(), 2);
		assert_eq!(segments.silent_count(), 1);

		let all = segments.all_segments();
		assert!(all[0].is_audible());
		assert_eq!(all[0].start, 0.0);
		assert_eq!(all[0].end, Some(3.0));

		assert!(!all[1].is_audible());
		assert_eq!(all[1].start, 3.0);
		assert_eq!(all[1].end, Some(5.0));

		assert!(all[2].is_audible());
		assert_eq!(all[2].start, 5.0);
		assert_eq!(all[2].end, Some(10.0));
	}

	#[test]
	fn test_invert_silence_at_start() {
		let silence = vec![(0.0, Some(2.0))];
		let segments = invert_silence_segments(&silence, Some(10.0), 0.0);

		assert_eq!(segments.audible_count(), 1);
		assert_eq!(segments.silent_count(), 1);

		let all = segments.all_segments();
		assert!(!all[0].is_audible());
		assert_eq!(all[0].start, 0.0);

		assert!(all[1].is_audible());
		assert_eq!(all[1].start, 2.0);
	}

	#[test]
	fn test_invert_silence_at_end() {
		let silence = vec![(8.0, Some(10.0))];
		let segments = invert_silence_segments(&silence, Some(10.0), 0.0);

		assert_eq!(segments.audible_count(), 1);
		assert_eq!(segments.silent_count(), 1);

		let all = segments.all_segments();
		assert!(all[0].is_audible());
		assert_eq!(all[0].end, Some(8.0));

		assert!(!all[1].is_audible());
		assert_eq!(all[1].start, 8.0);
	}

	#[test]
	fn test_min_segment_duration() {
		// Audible segment from 2.0 to 2.05 should be skipped with min 0.1
		let silence = vec![(0.0, Some(2.0)), (2.05, Some(5.0))];
		let segments = invert_silence_segments(&silence, Some(10.0), 0.1);

		// Should only have one audible segment (5.0 to 10.0)
		assert_eq!(segments.audible_count(), 1);
	}

	#[test]
	fn test_segment_contains() {
		let seg = Segment::new(SegmentType::Audible, 5.0, Some(10.0));
		assert!(!seg.contains(4.9));
		assert!(seg.contains(5.0));
		assert!(seg.contains(7.5));
		assert!(!seg.contains(10.0)); // exclusive end
	}

	#[test]
	fn test_segment_contains_open_end() {
		let seg = Segment::new(SegmentType::Audible, 5.0, None);
		assert!(!seg.contains(4.9));
		assert!(seg.contains(5.0));
		assert!(seg.contains(100.0)); // no end means goes forever
	}
	#[test]
	fn test_segment_list_stats() {
		let segments = vec![
			Segment::new(SegmentType::Audible, 0.0, Some(10.0)),
			Segment::new(SegmentType::Silent, 10.0, Some(15.0)),
			Segment::new(SegmentType::Audible, 15.0, Some(20.0)),
		];
		let list = SegmentList::new(segments);

		assert_eq!(list.audible_count(), 2);
		assert_eq!(list.silent_count(), 1);
		assert!((list.total_audible_duration - 15.0).abs() < 0.001);
		assert!((list.total_silent_duration - 5.0).abs() < 0.001);
		assert!(list.has_audible());
	}
}
