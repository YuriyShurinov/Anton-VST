@echo off
REM =========================================================================
REM package_windows.bat - Create an installer for DeFeedback Pro (Windows)
REM
REM Usage: scripts\package_windows.bat [build_dir]
REM =========================================================================

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."
set "BUILD_DIR=%~1"

if "%BUILD_DIR%"=="" set "BUILD_DIR=build"

REM Resolve relative path
if not exist "%BUILD_DIR%\" (
    set "BUILD_DIR=%PROJECT_DIR%\%BUILD_DIR%"
)

set "ARTEFACTS_DIR=%BUILD_DIR%\DeFeedbackPro_artefacts\Release"
set "VST3_DIR=%ARTEFACTS_DIR%\VST3\DeFeedback Pro.vst3"
set "MODEL_FILE=%BUILD_DIR%\models\nsnet2.onnx"
set "ONNXRT_DLL=%PROJECT_DIR%\onnxruntime-win-x64-1.17.1\lib\onnxruntime.dll"
set "STAGING_DIR=%BUILD_DIR%\installer_staging"
set "ISS_FILE=%SCRIPT_DIR%installer_win.iss"

echo ===================================================
echo  DeFeedback Pro Windows Installer Builder
echo ===================================================
echo.

REM ---------------------------------------------------------------------------
REM Validation
REM ---------------------------------------------------------------------------

if not exist "%VST3_DIR%\" (
    echo ERROR: VST3 bundle not found at: %VST3_DIR%
    echo        Make sure you have built the project in Release mode first.
    exit /b 1
)

if not exist "%ONNXRT_DLL%" (
    echo WARNING: onnxruntime.dll not found at: %ONNXRT_DLL%
    echo          The installer will not include the ONNX Runtime DLL.
    set "HAS_ONNXRT=0"
) else (
    set "HAS_ONNXRT=1"
)

if not exist "%MODEL_FILE%" (
    echo WARNING: Model file not found at: %MODEL_FILE%
    echo          The installer will not include the ML model.
    set "HAS_MODEL=0"
) else (
    set "HAS_MODEL=1"
)

REM ---------------------------------------------------------------------------
REM Prepare staging area
REM ---------------------------------------------------------------------------

echo Preparing staging area...
if exist "%STAGING_DIR%" rmdir /s /q "%STAGING_DIR%"
mkdir "%STAGING_DIR%\vst3"
mkdir "%STAGING_DIR%\models"

echo Copying VST3 bundle...
xcopy /e /i /q /y "%VST3_DIR%" "%STAGING_DIR%\vst3\DeFeedback Pro.vst3\"

if "%HAS_ONNXRT%"=="1" (
    echo Copying onnxruntime.dll...
    copy /y "%ONNXRT_DLL%" "%STAGING_DIR%\vst3\"
)

if "%HAS_MODEL%"=="1" (
    echo Copying ML model...
    copy /y "%MODEL_FILE%" "%STAGING_DIR%\models\"
)

REM ---------------------------------------------------------------------------
REM Try Inno Setup, otherwise fall back to ZIP
REM ---------------------------------------------------------------------------

set "ISCC="

REM Check common Inno Setup locations
if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" (
    set "ISCC=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
) else if exist "C:\Program Files\Inno Setup 6\ISCC.exe" (
    set "ISCC=C:\Program Files\Inno Setup 6\ISCC.exe"
) else if exist "C:\Program Files (x86)\Inno Setup 5\ISCC.exe" (
    set "ISCC=C:\Program Files (x86)\Inno Setup 5\ISCC.exe"
)

REM Also check PATH
if "%ISCC%"=="" (
    where iscc >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "tokens=*" %%i in ('where iscc') do set "ISCC=%%i"
    )
)

if not "%ISCC%"=="" (
    echo.
    echo Found Inno Setup: %ISCC%
    echo Compiling installer...
    echo.

    "%ISCC%" /DProjectDir="%PROJECT_DIR%" /DBuildDir="%BUILD_DIR%" /DStagingDir="%STAGING_DIR%" "%ISS_FILE%"

    if !errorlevel! neq 0 (
        echo ERROR: Inno Setup compilation failed.
        exit /b 1
    )

    echo.
    echo === Installer built successfully ===
    echo Output: %PROJECT_DIR%\DeFeedback_Pro_Installer.exe
) else (
    echo.
    echo Inno Setup not found. Creating ZIP package instead...
    echo.

    REM Create a README for the ZIP
    (
        echo DeFeedback Pro - Manual Installation
        echo =====================================
        echo.
        echo To install DeFeedback Pro manually:
        echo.
        echo 1. Copy the "DeFeedback Pro.vst3" folder to:
        echo    C:\Program Files\Common Files\VST3\
        echo.
        echo 2. Copy "onnxruntime.dll" to the same directory as the VST3:
        echo    C:\Program Files\Common Files\VST3\DeFeedback Pro.vst3\Contents\x86_64-win\
        echo.
        echo 3. Copy the "models" folder to:
        echo    C:\Program Files\Common Files\VST3\DeFeedback Pro.vst3\Contents\Resources\models\
        echo.
        echo 4. Restart your DAW.
    ) > "%STAGING_DIR%\INSTALL_README.txt"

    REM Use PowerShell to create ZIP
    set "ZIP_OUTPUT=%PROJECT_DIR%\DeFeedback_Pro_Windows.zip"
    powershell -NoProfile -Command "Compress-Archive -Path '%STAGING_DIR%\*' -DestinationPath '%ZIP_OUTPUT%' -Force"

    if !errorlevel! neq 0 (
        echo ERROR: Failed to create ZIP archive.
        exit /b 1
    )

    echo.
    echo === ZIP package created ===
    echo Output: %ZIP_OUTPUT%
    echo.
    echo NOTE: For a proper installer with uninstall support, install Inno Setup 6
    echo       from https://jrsoftware.org/isinfo.php and re-run this script.
)

REM ---------------------------------------------------------------------------
REM Cleanup
REM ---------------------------------------------------------------------------

echo.
echo Cleaning up staging area...
rmdir /s /q "%STAGING_DIR%"

echo Done.
endlocal
