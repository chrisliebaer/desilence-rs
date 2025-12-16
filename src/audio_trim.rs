use ffmpeg::{
	ffi,
	frame,
};
use ffmpeg_next as ffmpeg;

/// Trim an audio frame in-place by adjusting its data pointers and sample count.
///
/// # Arguments
/// * `frame` - The audio frame to trim.
/// * `start_sample` - Number of samples to skip from the beginning.
/// * `keep_samples` - Number of samples to keep after the start offset.
///
/// # Safety
/// This function relies on FFmpeg's internal frame structure. It modifies raw pointers
/// to "slide" the view of the data. The underlying buffer is not reallocated, but
/// the frame metadata (nb_samples) is updated to reflect the new size.
pub fn trim_audio_frame(frame: &mut frame::Audio, start_sample: usize, keep_samples: usize) {
	if start_sample == 0 && keep_samples == frame.samples() {
		return;
	}

	unsafe {
		let ptr = frame.as_mut_ptr();
		(*ptr).nb_samples = keep_samples as i32;

		if start_sample > 0 {
			// Calculate byte offset per sample
			let sample_fmt = std::mem::transmute::<i32, ffi::AVSampleFormat>((*ptr).format);
			let bytes_per_sample = ffi::av_get_bytes_per_sample(sample_fmt);
			let channels = (*ptr).ch_layout.nb_channels;
			let is_planar = ffi::av_sample_fmt_is_planar(sample_fmt) == 1;

			if bytes_per_sample > 0 {
				let offset_bytes = if is_planar {
					start_sample as i32 * bytes_per_sample
				} else {
					start_sample as i32 * bytes_per_sample * channels
				};

				// Update pointers for all planes
				let planes = if is_planar { channels } else { 1 };
				for i in 0..planes {
					let data_ptr = (*ptr).data[i as usize];
					if !data_ptr.is_null() {
						(*ptr).data[i as usize] = data_ptr.add(offset_bytes as usize);
					}
				}
			}
		}
	}
}
