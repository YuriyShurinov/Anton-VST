#!/usr/bin/env bash
#
# package_all.sh - Detect OS and run the appropriate installer packager
#
# Usage: ./scripts/package_all.sh [build_dir]
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${1:-build}"

echo "=== DeFeedback Pro Installer Packager ==="
echo ""

OS="$(uname -s)"

case "$OS" in
    Darwin)
        echo "Detected macOS. Running macOS packager..."
        echo ""
        exec "$SCRIPT_DIR/package_macos.sh" "$BUILD_DIR"
        ;;
    Linux)
        echo "ERROR: Linux packaging is not yet supported."
        echo "       DeFeedback Pro currently targets macOS and Windows."
        exit 1
        ;;
    MINGW*|MSYS*|CYGWIN*|Windows_NT)
        echo "Detected Windows. Running Windows packager..."
        echo ""

        # Convert to Windows-style path for the batch file
        BAT_SCRIPT="$(cygpath -w "$SCRIPT_DIR/package_windows.bat" 2>/dev/null || echo "$SCRIPT_DIR/package_windows.bat")"

        # If running from Git Bash / MSYS, invoke cmd.exe
        if command -v cmd.exe &>/dev/null; then
            cmd.exe //c "$BAT_SCRIPT" "$BUILD_DIR"
        else
            echo "ERROR: Cannot find cmd.exe to run the Windows batch script."
            echo "       Please run scripts/package_windows.bat directly from a Command Prompt."
            exit 1
        fi
        ;;
    *)
        echo "ERROR: Unknown operating system: $OS"
        echo "       Supported platforms: macOS (Darwin), Windows (MINGW/MSYS/CYGWIN)"
        exit 1
        ;;
esac
