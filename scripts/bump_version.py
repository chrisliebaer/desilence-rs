import sys
import re

def bump_version(version_str, level):
    major, minor, patch = map(int, version_str.split('.'))
    
    if level == 'major':
        major += 1
        minor = 0
        patch = 0
    elif level == 'minor':
        minor += 1
        patch = 0
    elif level == 'patch':
        patch += 1
    else:
        raise ValueError(f"Unknown level: {level}")
        
    return f"{major}.{minor}.{patch}"

def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python bump_version.py <patch|minor|major> <cargo_toml_path> [output_version_file]", file=sys.stderr)
        sys.exit(1)

    level = sys.argv[1]
    file_path = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) == 4 else None

    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: Cargo file not found at {file_path}", file=sys.stderr)
        sys.exit(1)

    # Regex to find version = "x.y.z"
    version_pattern = re.compile(r'^version\s*=\s*"(\d+\.\d+\.\d+)"', re.MULTILINE)
    
    match = version_pattern.search(content)
    if not match:
        print("Error: Could not find version string in Cargo.toml", file=sys.stderr)
        sys.exit(1)

    current_version = match.group(1)
    try:
        new_version = bump_version(current_version, level)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Bumping version: {current_version} -> {new_version}", file=sys.stderr)
    
    new_content = version_pattern.sub(f'version = "{new_version}"', content, count=1)
    
    with open(file_path, 'w') as f:
        f.write(new_content)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(new_version)
    else:
        # Fallback to stdout if no file specified
        print(new_version)

if __name__ == "__main__":
    main()
