# desilence-rs

A high-performance video silence remover that streams processed video directly to `ffmpeg`.

## Overview

`desilence-rs` intelligently removes silent segments from video files without generating temporary files or requiring multiple passes. It functions as a smart filter: it decodes input video, discards silent frames, and streams the remaining content (raw video/audio) directly to `stdout`.

Because the output is raw, you simply pipe it to `ffmpeg` to encode the final result in any format, quality, or codec you prefer.

## Usage

The core workflow creates a pipeline:
**Input File** $\to$ **desilence-rs** $\to$ **Pipe** $\to$ **FFmpeg** $\to$ **Output File**

```bash
./desilence-rs -i input.mp4 | ffmpeg -f nut -i pipe: -c:v libx264 -c:a aac output.mp4
```

The `-f nut` specifies the input format as `nut`, which is a low-overhead container used by `desilence-rs` to stream the raw video/audio data from `stdout`.

> [!IMPORTANT]
> **Preserving Multiple Streams**: FFmpeg defaults to selecting only the "best" video and audio stream (usually the first one). `desilence-rs` outputs **ALL** streams. To ensure the final output file contains all streams (e.g. multi-track audio), you **MUST** add `-map 0` to your ffmpeg command:
> ```bash
> ./desilence-rs ... | ffmpeg ... -map 0 ... output.mkv
> ```

### Why Pipe?
This design provides **maximum flexibility** compared to all-in-one tools:
- **Zero Quality Loss**: Intermediate stream is raw/lossless.
- **Universal Encoding**: Use any encoder ffmpeg supports (x264, HEVC, AV1, NVENC, etc.).
- **No Disk Bottlenecks**: Processing happens entirely in memory and streams data.
- **Single Pass**: The video is decoded only once.

### Examples

#### Standard H.264 Encoding
Good for general compatibility.
```bash
# -f nut -i pipe: tells ffmpeg to read the stream from stdin
./desilence-rs -i input.mp4 | ffmpeg -f nut -i pipe: -c:v libx264 -preset fast -c:a aac output.mp4
```

#### High Quality / Archival
Using H.265 (HEVC) and Opus audio for better compression.
```bash
./desilence-rs -i lecture.mkv | ffmpeg -f nut -i pipe: -c:v libx265 -crf 20 -c:a libopus -b:a 128k output.mkv
```

#### Hardware Acceleration (NVIDIA GPU)
Extremely fast processing using NVENC.
```bash
./desilence-rs -i game.mp4 | ffmpeg -f nut -i pipe: -c:v h264_nvenc -preset p7 -c:a aac output.mp4
```

#### Custom Silence Detection
Adjust the noise threshold (`-50dB` by default) and minimum silence period.
```bash
# Remove silence quieter than -40dB lasting longer than 0.3s
./desilence-rs -i input.mp4 -n -40dB -d 0.3 | ffmpeg -f nut -i pipe: ...
```

## Installation

### Prerequisites
- **Rust 1.70+**
- **FFmpeg** (static or shared libraries, depending on build method)

### Building on Windows

The project uses a PowerShell script `build.ps1` that handles dependencies automatically.

#### Option A: Automatic Build (Recommended for CI/Fresh Installs)
This mode downloads a pre-built FFmpeg release from `gyan.dev` and bundles the required DLLs.
```powershell
.\build.ps1
```

#### Option B: Local VCPKG Build
If you have `vcpkg` installed and want to use your system libraries:
1.  Ensure `VCPKG_ROOT` is set in your environment (or `.env` file).
2.  Install FFmpeg with the `x64-windows` triplet:
    ```powershell
    vcpkg install ffmpeg[gpl,x264,mp3lame]:x64-windows
    ```
3.  Run the build script with the `-Local` flag:
    ```powershell
    .\build.ps1 -Local
    ```

### Building on Linux (Ubuntu/Debian)

#### Option A: Dynamic Linking (Recommended for Package Managers)
1.  Install build dependencies (including FFmpeg development headers):
    ```bash
    sudo apt-get install pkg-config libclang-dev clang libavcodec-dev libavformat-dev libavfilter-dev libavdevice-dev libavutil-dev
    ```
2.  Build:
    ```bash
    cargo build --release
    ```
    *Result*: A small binary that requires system FFmpeg libraries.

#### Option B: Static Linking (Portable)
1.  Install build tools:
    ```bash
    sudo apt-get install yasm nasm pkg-config libclang-dev clang
    ```
2.  Build with static feature:
    ```bash
    cargo build --release --features static
    ```
    *Result*: A larger, self-contained binary.

### Docker Build
Build a container that includes both the binary and a standalone ffmpeg:
```bash
docker build -t desilence-rs .
```

### Docker Usage

You can use the Docker container in two main ways. Remember to use `--rm` for cleanup and bind mounts (`-v`) to access your files.

#### 1. Streaming to Host FFmpeg
Run `desilence-rs` inside the container but pipe the output to a local `ffmpeg` instance.
This is ideal if your local ffmpeg has codecs and settings you want to use.

```bash
# Powershell example
# Mount current directory ($PWD) to /data inside container
docker run --rm -v ${PWD}:/data desilence-rs -i /data/input.mp4 | ffmpeg -f nut -i pipe: -map 0 -c copy output.mp4
```

#### 2. Full Processing Inside Container
Run the entire pipeline inside the container. This uses the bundled ffmpeg.

```bash
# Run a shell command inside the container to execute the pipeline
docker run --rm -v ${PWD}:/data --entrypoint /bin/sh desilence-rs -c "desilence-rs -i /data/input.mp4 | ffmpeg -f nut -i pipe: -map 0 -c:v libx264 -c:a aac /data/output.mp4"
```

## Options

<!-- BEGIN CLI OPTIONS -->
```text
Remove silence from video files, streaming output to stdout

Usage: desilence-rs [OPTIONS] --input <INPUT>

Options:
  -i, --input <INPUT>
          Input video file path
  -o, --output <OUTPUT>
          Output file path (defaults to stdout)
  -n, --noise-threshold <NOISE_THRESHOLD>
          Silence detection threshold in dB (negative value) [default: -50dB]
  -d, --duration <DURATION>
          Minimum silence duration in seconds [default: 0.5]
  -a, --audio-stream <AUDIO_STREAM>
          Audio stream index to use for silence detection (0-based)
      --merge-audio [<MERGE_AUDIO>...]
          Merge audio streams for silence detection
  -f, --force
          Force output to terminal even if it looks like a TTY
  -l, --list-streams
          List available streams and exit
  -v, --verbose...
          Verbose output (repeat for more verbosity: -v, -vv, -vvv)
  -q, --quiet
          Quiet mode - suppress all non-error output
  -h, --help
          Print help (see more with '--help')
  -V, --version
          Print version
```
<!-- END CLI OPTIONS -->
