#!/usr/bin/env python3
"""
Update or validate README.md CLI options section.

This script updates the Options section in README.md with the actual help text
from the desilence-rs binary, or validates that it matches in CI mode.

Usage:
    # Update README.md (local development):
    python3 scripts/update-readme-help.py --update

    # Validate README.md matches binary (CI):
    python3 scripts/update-readme-help.py --check

Requirements:
    - Python 3.8+
    - Built desilence-rs binary in target/release/ or provided via --binary

The README.md must have markers:
    <!-- BEGIN CLI OPTIONS -->
    ...content to replace...
    <!-- END CLI OPTIONS -->
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def get_help_text(binary_path: Path) -> str:
	"""Extract help text from desilence-rs using --dump-help flag."""
	try:
		result = subprocess.run(
			[str(binary_path), "--dump-help"],
			capture_output=True,
			text=True,
			check=True,
		)
		return result.stdout
	except subprocess.CalledProcessError as e:
		print(f"Error: Failed to run {binary_path} --dump-help", file=sys.stderr)
		print(f"stderr: {e.stderr}", file=sys.stderr)
		sys.exit(1)
	except FileNotFoundError:
		print(f"Error: Binary not found at {binary_path}", file=sys.stderr)
		print("Build the project first with: cargo build --release", file=sys.stderr)
		sys.exit(1)


def format_help_for_readme(help_text: str) -> str:
	"""Format help text for inclusion in README markdown."""
	# Wrap in code fence
	return f"```text\n{help_text.rstrip()}\n```"


def update_readme(readme_path: Path, new_content: str, check_only: bool) -> bool:
	"""
	Update or validate the CLI options section in README.md.

	Returns True if content matches (or was updated), False if mismatch in check mode.
	"""
	if not readme_path.exists():
		print(f"Error: README not found at {readme_path}", file=sys.stderr)
		sys.exit(1)

	readme_text = readme_path.read_text()

	# Find the markers
	begin_marker = "<!-- BEGIN CLI OPTIONS -->"
	end_marker = "<!-- END CLI OPTIONS -->"

	if begin_marker not in readme_text or end_marker not in readme_text:
		print(f"Error: README markers not found", file=sys.stderr)
		print(f"Expected markers: {begin_marker} and {end_marker}", file=sys.stderr)
		sys.exit(1)

	# Extract content between markers
	pattern = re.compile(
		rf"{re.escape(begin_marker)}\s*\n(.*?)\n\s*{re.escape(end_marker)}",
		re.DOTALL,
	)

	match = pattern.search(readme_text)
	if not match:
		print(f"Error: Could not parse content between markers", file=sys.stderr)
		sys.exit(1)

	current_content = match.group(1).strip()
	new_content_stripped = new_content.strip()

	if current_content == new_content_stripped:
		return True

	if check_only:
		print("Error: README CLI options section is outdated!", file=sys.stderr)
		print("\nExpected:", file=sys.stderr)
		print(new_content_stripped[:500], file=sys.stderr)
		print("\nCurrent:", file=sys.stderr)
		print(current_content[:500], file=sys.stderr)
		print("\nRun locally with --update to fix:", file=sys.stderr)
		print("  uv run scripts/update-readme-help.py --update", file=sys.stderr)
		return False

	# Update the README
	updated_text = pattern.sub(
		f"{begin_marker}\n{new_content_stripped}\n{end_marker}",
		readme_text,
	)

	readme_path.write_text(updated_text)
	print(f"✓ Updated {readme_path}")
	return True


def main():
	parser = argparse.ArgumentParser(
		description="Update or validate README.md CLI options section",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog=__doc__,
	)
	parser.add_argument(
		"--check",
		action="store_true",
		help="Check if README is up-to-date (CI mode)",
	)
	parser.add_argument(
		"--update",
		action="store_true",
		help="Update README with current help text (local mode)",
	)
	parser.add_argument(
		"--binary",
		type=Path,
		help="Path to desilence-rs binary (default: target/release/desilence-rs or .exe)",
	)
	parser.add_argument(
		"--readme",
		type=Path,
		default=Path("README.md"),
		help="Path to README.md (default: README.md)",
	)

	args = parser.parse_args()

	if not args.check and not args.update:
		parser.error("Must specify either --check or --update")

	if args.check and args.update:
		parser.error("Cannot specify both --check and --update")

	# Determine binary path
	if args.binary:
		binary_path = args.binary
	else:
		# Try common locations
		candidates = [
			Path("target/release/desilence-rs"),
			Path("target/release/desilence-rs.exe"),
			Path("desilence-rs"),
			Path("desilence-rs.exe"),
		]
		binary_path = None
		for candidate in candidates:
			if candidate.exists():
				binary_path = candidate
				break

		if not binary_path:
			print("Error: Could not find desilence-rs binary", file=sys.stderr)
			print("Tried:", file=sys.stderr)
			for candidate in candidates:
				print(f"  - {candidate}", file=sys.stderr)
			print("\nBuild the project first: cargo build --release", file=sys.stderr)
			sys.exit(1)

	print(f"Using binary: {binary_path}")

	# Extract help text
	help_text = get_help_text(binary_path)
	formatted_help = format_help_for_readme(help_text)

	# Update or validate README
	check_only = args.check
	success = update_readme(args.readme, formatted_help, check_only)

	if check_only:
		if success:
			print("✓ README CLI options section is up-to-date")
			sys.exit(0)
		else:
			sys.exit(1)
	else:
		sys.exit(0)


if __name__ == "__main__":
	main()
