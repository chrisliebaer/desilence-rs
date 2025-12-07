# Build stage
FROM rust:latest AS builder

# Install build dependencies
# ffmpeg-next requires libav*-dev for dynamic linking
RUN apt-get update && apt-get install -y \
	pkg-config libclang-dev clang \
	libavcodec-dev libavformat-dev libavfilter-dev libavdevice-dev libavutil-dev \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Run tests
RUN cargo test --release

# Build release binary
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Install standalone FFmpeg
RUN apt-get update && apt-get install -y \
	ffmpeg \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/target/release/desilence-rs /usr/local/bin/desilence-rs

# Set entrypoint (can be overridden to run ffmpeg directly if needed)
ENTRYPOINT ["desilence-rs"]
CMD ["--help"]
