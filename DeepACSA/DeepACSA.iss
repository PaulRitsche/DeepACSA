[Setup]
AppName=DeepACSA0.3.2
AppVersion=0.3.2
DefaultDirName={pf}\DeepACSA
DefaultGroupName=DeepASA
OutputDir=installer
OutputBaseFilename=DeepACSA0.3.2_Installer
Compression=lzma
SolidCompression=yes
SetupIconFile=gui_helpers\icon.ico

[Files]
Source: "dist\DeepACSA\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\DeepACSA0.3.2"; Filename: "{app}\deep_acsa_gui.exe"
Name: "{commondesktop}\DeepACSA0.3.2"; Filename: "{app}\deep_acsa_gui.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"