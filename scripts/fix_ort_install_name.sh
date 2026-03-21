#!/bin/bash
# Fix ONNX Runtime install name in a plugin binary.
# Usage: fix_ort_install_name.sh <binary> <frameworks_dir>
# Tries all known @rpath patterns; silently skips those not found.

BINARY="$1"
FW_DIR="$2"
FW_DYLIB="$FW_DIR/libonnxruntime.dylib"
TARGET="@loader_path/../Frameworks/libonnxruntime.dylib"

if [ ! -f "$BINARY" ]; then echo "Binary not found: $BINARY"; exit 0; fi
if [ ! -f "$FW_DYLIB" ]; then echo "Dylib not found: $FW_DYLIB"; exit 0; fi

# Try changing all known install name patterns
for NAME in \
    "@rpath/libonnxruntime.dylib" \
    "@rpath/libonnxruntime.1.17.1.dylib" \
    "@rpath/libonnxruntime.1.dylib"; do
    install_name_tool -change "$NAME" "$TARGET" "$BINARY" 2>/dev/null
done

# Fix the dylib's own id
install_name_tool -id "$TARGET" "$FW_DYLIB" 2>/dev/null

echo "Fixed ORT install name in $(basename "$BINARY")"
exit 0
