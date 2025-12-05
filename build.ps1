# Build script for desilence-rs
# Run with: 
#   .\build.ps1          (CI/Clean mode: Downloads gyan.dev ffmpeg)
#   .\build.ps1 -Local   (Local mode: Uses existing VCPKG/system ffmpeg)

param (
	[switch]$Local
)

$ErrorActionPreference = "Stop"

# Load .env file if it exists
if (Test-Path ".env") {
	Get-Content ".env" | ForEach-Object {
		if ($_ -match "^\s*([^#=]+)\s*=\s*(.*)$") {
			$name = $matches[1].Trim()
			$val = $matches[2].Trim()
			[System.Environment]::SetEnvironmentVariable($name, $val, "Process")
		}
	}
}

if (-not $Local) {
	Write-Host "CI Mode: Setting up pre-built FFmpeg from gyan.dev..." -ForegroundColor Cyan
	
	$ffmpegUrl = "https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-7.1.1-full_build-shared.7z"
	$libsDir = Join-Path $PSScriptRoot "libs"
	$ffmpegArchive = Join-Path $libsDir "ffmpeg.7z"
	$extractDir = Join-Path $libsDir "ffmpeg_extract"

	# Create libs directory
	if (-not (Test-Path $libsDir)) { New-Item -ItemType Directory -Path $libsDir | Out-Null }
	
	# Download if not present
	if (-not (Test-Path $ffmpegArchive)) {
		Write-Host "Downloading FFmpeg from $ffmpegUrl..."
		Invoke-WebRequest -Uri $ffmpegUrl -OutFile $ffmpegArchive
	}

	# Extract using 7z (Required for .7z files, available in CI environment and common locally)
	# Clean up previous extraction attempt to ensure freshness
	if (Test-Path $extractDir) {
		Write-Host "Cleaning up previous extraction directory..."
		Remove-Item -Path $extractDir -Recurse -Force
	}

	# Extract using 7z
	Write-Host "Extracting FFmpeg (using 7z)..."
	
	if (-not (Get-Command "7z" -ErrorAction SilentlyContinue)) {
		Write-Error "Command '7z' not found in PATH."
		exit 1
	}

	$process = Start-Process -FilePath "7z" -ArgumentList "x", "`"$ffmpegArchive`"", "-o`"$extractDir`"", "-y" -Wait -NoNewWindow -PassThru
	if ($process.ExitCode -ne 0) {
		Write-Error "7z extraction failed with exit code $($process.ExitCode)"
		exit 1
	}

	# Find the inner directory
	$innerDir = Get-ChildItem -Path $extractDir -Directory | Select-Object -First 1
	if ($null -eq $innerDir) {
		Write-Error "Could not find extracted FFmpeg directory in $extractDir"
		exit 1
	}
	
	$ffmpegRoot = $innerDir.FullName
	Write-Host "Found FFmpeg at $ffmpegRoot"

	# Set FFMPEG_DIR for build.rs
	$env:FFMPEG_DIR = $ffmpegRoot
	
	# Add to PATH so DLLs are found during build scripts if needed
	$env:PATH = "$($ffmpegRoot)\bin;$env:PATH"
} else {
	Write-Host "Local Mode: Using system/VCPKG FFmpeg configuration..." -ForegroundColor Cyan
	if ($env:VCPKG_ROOT) {
		Write-Host "VCPKG_ROOT is set: $env:VCPKG_ROOT"
	}
}

Write-Host "Starting cargo build..." -ForegroundColor Cyan
cargo build --release

if ($LASTEXITCODE -eq 0) {
	Write-Host "Build successful!" -ForegroundColor Green
	
	# Copy DLLs to release folder
	$targetDir = Join-Path $PSScriptRoot "target\release"
	if (-not (Test-Path $targetDir)) { New-Item -ItemType Directory -Path $targetDir | Out-Null }

	if (-not $Local) {
		Write-Host "Copying DLLs from FFmpeg release..."
		$binDir = Join-Path $env:FFMPEG_DIR "bin"
		Copy-Item "$binDir\*.dll" $targetDir -Force
		Write-Host "DLLs copied."
	} elseif ($env:VCPKG_ROOT) {
		# Attempt to find vcpkg DLLs for x64-windows
		Write-Host "Local Mode: Attempting to copy DLLs from VCPKG..."
		$vcpkgBin = Join-Path $env:VCPKG_ROOT "installed\x64-windows\bin"
		if (Test-Path $vcpkgBin) {
			Copy-Item "$vcpkgBin\*.dll" $targetDir -Force
			Write-Host "VCPKG DLLs copied from $vcpkgBin"
		} else {
			Write-Warning "Could not find VCPKG bin directory at $vcpkgBin. Ensure you installed with triplet x64-windows."
		}
	}
} else {
	Write-Error "Build failed with exit code $LASTEXITCODE"
	exit $LASTEXITCODE
}
