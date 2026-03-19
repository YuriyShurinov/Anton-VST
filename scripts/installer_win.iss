; =========================================================================
; DeFeedback Pro - Inno Setup Installer Script
;
; Compile with: ISCC.exe /DProjectDir="..." /DBuildDir="..." /DStagingDir="..." installer_win.iss
; Or run package_windows.bat which invokes this automatically.
; =========================================================================

#ifndef StagingDir
  #define StagingDir "..\build\installer_staging"
#endif
#ifndef ProjectDir
  #define ProjectDir ".."
#endif
#ifndef BuildDir
  #define BuildDir "..\build"
#endif

#define AppName "DeFeedback Pro"
#define AppVersion "1.0.0"
#define AppPublisher "DeFeedback Pro"
#define AppURL "https://github.com/defeedbackpro"

[Setup]
AppId={{B8E3F2A1-4C7D-4E9B-A5F6-1D2E3F4A5B6C}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}
DefaultDirName={commoncf}\VST3\{#AppName}.vst3
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
DisableDirPage=yes
LicenseFile={#ProjectDir}\scripts\LICENSE_INSTALLER.txt
OutputDir={#ProjectDir}
OutputBaseFilename=DeFeedback_Pro_Installer
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible
WizardStyle=modern
SetupIconFile=compiler:SetupClassicIcon.ico
UninstallDisplayName={#AppName}
MinVersion=10.0

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
; VST3 plugin bundle
Source: "{#StagingDir}\vst3\DeFeedback Pro.vst3\*"; DestDir: "{commoncf}\VST3\DeFeedback Pro.vst3"; Flags: ignoreversion recursesubdirs createallsubdirs

; ONNX Runtime DLL - placed alongside the plugin binary
Source: "{#StagingDir}\vst3\onnxruntime.dll"; DestDir: "{commoncf}\VST3\DeFeedback Pro.vst3\Contents\x86_64-win"; Flags: ignoreversion; Check: FileExists(ExpandConstant('{#StagingDir}\vst3\onnxruntime.dll'))

; ML model file
Source: "{#StagingDir}\models\nsnet2.onnx"; DestDir: "{commoncf}\VST3\DeFeedback Pro.vst3\Contents\Resources\models"; Flags: ignoreversion; Check: FileExists(ExpandConstant('{#StagingDir}\models\nsnet2.onnx'))

[Icons]
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"

[UninstallDelete]
Type: filesandordirs; Name: "{commoncf}\VST3\DeFeedback Pro.vst3"

[Messages]
WelcomeLabel1=Welcome to the {#AppName} Setup Wizard
WelcomeLabel2=This will install {#AppName} {#AppVersion} on your computer.%n%n{#AppName} is a real-time feedback suppression VST3 plugin powered by machine learning.%n%nThe installer will place the VST3 plugin, ONNX Runtime library, and ML model in the standard VST3 directory.%n%nPlease close any DAW applications before continuing.

[Code]
function FileExists(const FileName: string): Boolean;
begin
  Result := FileOrDirExists(FileName);
end;
