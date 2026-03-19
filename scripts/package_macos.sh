#!/usr/bin/env bash
#
# package_macos.sh - Create a .pkg installer for DeFeedback Pro (macOS)
#
# Usage: ./scripts/package_macos.sh [build_dir]
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${1:-build}"

# Resolve build dir relative to project root if not absolute
if [[ "$BUILD_DIR" != /* ]]; then
    BUILD_DIR="$PROJECT_DIR/$BUILD_DIR"
fi

ARTEFACTS_DIR="$BUILD_DIR/DeFeedbackPro_artefacts/Release"
VST3_BUNDLE="$ARTEFACTS_DIR/VST3/DeFeedback Pro.vst3"
AU_BUNDLE="$ARTEFACTS_DIR/AU/DeFeedback Pro.component"
MODEL_FILE="$BUILD_DIR/models/nsnet2.onnx"

PKG_DIR="$BUILD_DIR/installer_staging"
OUTPUT_PKG="$PROJECT_DIR/DeFeedback_Pro_Installer.pkg"

APP_VERSION="1.0.0"
APP_IDENTIFIER="com.defeedbackpro"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

echo "=== DeFeedback Pro macOS Installer Builder ==="
echo ""

if [[ ! -d "$VST3_BUNDLE" ]]; then
    echo "ERROR: VST3 bundle not found at: $VST3_BUNDLE"
    echo "       Make sure you have built the project in Release mode first."
    exit 1
fi

HAS_AU=true
if [[ ! -d "$AU_BUNDLE" ]]; then
    echo "WARNING: AU component not found at: $AU_BUNDLE"
    echo "         The installer will include VST3 only."
    HAS_AU=false
fi

if ! command -v pkgbuild &>/dev/null; then
    echo "ERROR: pkgbuild not found. This script must be run on macOS."
    exit 1
fi

if ! command -v productbuild &>/dev/null; then
    echo "ERROR: productbuild not found. This script must be run on macOS."
    exit 1
fi

# ---------------------------------------------------------------------------
# Prepare staging area
# ---------------------------------------------------------------------------

echo "Preparing staging area..."
rm -rf "$PKG_DIR"
mkdir -p "$PKG_DIR/vst3_root/Library/Audio/Plug-Ins/VST3"
mkdir -p "$PKG_DIR/scripts_postinstall"
mkdir -p "$PKG_DIR/components"
mkdir -p "$PKG_DIR/resources"

# Copy VST3 bundle
echo "Copying VST3 bundle..."
cp -R "$VST3_BUNDLE" "$PKG_DIR/vst3_root/Library/Audio/Plug-Ins/VST3/"

# Copy model into VST3 bundle Resources if it exists
if [[ -f "$MODEL_FILE" ]]; then
    RESOURCES_DIR="$PKG_DIR/vst3_root/Library/Audio/Plug-Ins/VST3/DeFeedback Pro.vst3/Contents/Resources/models"
    mkdir -p "$RESOURCES_DIR"
    cp "$MODEL_FILE" "$RESOURCES_DIR/"
    echo "Copied model file into VST3 bundle Resources."
fi

# Copy AU bundle if present
if $HAS_AU; then
    mkdir -p "$PKG_DIR/au_root/Library/Audio/Plug-Ins/Components"
    cp -R "$AU_BUNDLE" "$PKG_DIR/au_root/Library/Audio/Plug-Ins/Components/"

    # Copy model into AU bundle Resources if it exists
    if [[ -f "$MODEL_FILE" ]]; then
        AU_RESOURCES_DIR="$PKG_DIR/au_root/Library/Audio/Plug-Ins/Components/DeFeedback Pro.component/Contents/Resources/models"
        mkdir -p "$AU_RESOURCES_DIR"
        cp "$MODEL_FILE" "$AU_RESOURCES_DIR/"
        echo "Copied model file into AU bundle Resources."
    fi
fi

# ---------------------------------------------------------------------------
# Create postinstall script to remove quarantine attributes
# ---------------------------------------------------------------------------

cat > "$PKG_DIR/scripts_postinstall/postinstall" << 'POSTINSTALL'
#!/bin/bash
# Remove quarantine attributes so the plugin loads without Gatekeeper warnings
xattr -dr com.apple.quarantine "/Library/Audio/Plug-Ins/VST3/DeFeedback Pro.vst3" 2>/dev/null || true
xattr -dr com.apple.quarantine "/Library/Audio/Plug-Ins/Components/DeFeedback Pro.component" 2>/dev/null || true
exit 0
POSTINSTALL
chmod +x "$PKG_DIR/scripts_postinstall/postinstall"

# ---------------------------------------------------------------------------
# Build component packages
# ---------------------------------------------------------------------------

echo "Building VST3 component package..."
pkgbuild \
    --root "$PKG_DIR/vst3_root" \
    --identifier "${APP_IDENTIFIER}.vst3" \
    --version "$APP_VERSION" \
    --scripts "$PKG_DIR/scripts_postinstall" \
    --install-location "/" \
    "$PKG_DIR/components/DeFeedbackPro_VST3.pkg"

if $HAS_AU; then
    echo "Building AU component package..."
    pkgbuild \
        --root "$PKG_DIR/au_root" \
        --identifier "${APP_IDENTIFIER}.au" \
        --version "$APP_VERSION" \
        --scripts "$PKG_DIR/scripts_postinstall" \
        --install-location "/" \
        "$PKG_DIR/components/DeFeedbackPro_AU.pkg"
fi

# ---------------------------------------------------------------------------
# Create distribution.xml
# ---------------------------------------------------------------------------

echo "Creating distribution.xml..."

if $HAS_AU; then
    DISTRIBUTION_XML=$(cat << 'DISTXML'
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
    <title>DeFeedback Pro</title>
    <welcome language="en" mime-type="text/html"><![CDATA[
        <html><body>
        <h1>DeFeedback Pro Installer</h1>
        <p>This installer will install the following components:</p>
        <ul>
            <li><b>DeFeedback Pro VST3</b> - VST3 plugin for compatible DAWs</li>
            <li><b>DeFeedback Pro AU</b> - Audio Unit plugin for Logic Pro, GarageBand, and other AU hosts</li>
        </ul>
        <p>Both plugin formats include the ML denoising model required for real-time feedback suppression.</p>
        <p>After installation, restart your DAW to see the plugin.</p>
        </body></html>
    ]]></welcome>
    <options customize="allow" require-scripts="false" rootVolumeOnly="true"/>
    <choices-outline>
        <line choice="vst3"/>
        <line choice="au"/>
    </choices-outline>
    <choice id="vst3" title="VST3 Plugin" description="Installs DeFeedback Pro VST3 to /Library/Audio/Plug-Ins/VST3/" selected="true">
        <pkg-ref id="com.defeedbackpro.vst3"/>
    </choice>
    <choice id="au" title="Audio Unit Plugin" description="Installs DeFeedback Pro AU to /Library/Audio/Plug-Ins/Components/" selected="true">
        <pkg-ref id="com.defeedbackpro.au"/>
    </choice>
    <pkg-ref id="com.defeedbackpro.vst3" version="1.0.0" onConclusion="none">DeFeedbackPro_VST3.pkg</pkg-ref>
    <pkg-ref id="com.defeedbackpro.au" version="1.0.0" onConclusion="none">DeFeedbackPro_AU.pkg</pkg-ref>
</installer-gui-script>
DISTXML
)
else
    DISTRIBUTION_XML=$(cat << 'DISTXML'
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
    <title>DeFeedback Pro</title>
    <welcome language="en" mime-type="text/html"><![CDATA[
        <html><body>
        <h1>DeFeedback Pro Installer</h1>
        <p>This installer will install the following component:</p>
        <ul>
            <li><b>DeFeedback Pro VST3</b> - VST3 plugin for compatible DAWs</li>
        </ul>
        <p>The plugin includes the ML denoising model required for real-time feedback suppression.</p>
        <p>After installation, restart your DAW to see the plugin.</p>
        </body></html>
    ]]></welcome>
    <options customize="allow" require-scripts="false" rootVolumeOnly="true"/>
    <choices-outline>
        <line choice="vst3"/>
    </choices-outline>
    <choice id="vst3" title="VST3 Plugin" description="Installs DeFeedback Pro VST3 to /Library/Audio/Plug-Ins/VST3/" selected="true">
        <pkg-ref id="com.defeedbackpro.vst3"/>
    </choice>
    <pkg-ref id="com.defeedbackpro.vst3" version="1.0.0" onConclusion="none">DeFeedbackPro_VST3.pkg</pkg-ref>
</installer-gui-script>
DISTXML
)
fi

echo "$DISTRIBUTION_XML" > "$PKG_DIR/distribution.xml"

# ---------------------------------------------------------------------------
# Build final product installer
# ---------------------------------------------------------------------------

echo "Building final installer package..."
productbuild \
    --distribution "$PKG_DIR/distribution.xml" \
    --package-path "$PKG_DIR/components" \
    --version "$APP_VERSION" \
    "$OUTPUT_PKG"

echo ""
echo "=== Build complete ==="
echo "Installer: $OUTPUT_PKG"
echo ""

# Cleanup staging
rm -rf "$PKG_DIR"
echo "Staging area cleaned up."
