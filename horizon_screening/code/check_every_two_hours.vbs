Option Explicit
Dim fs, shell, scriptPath, command
Set fs = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
scriptPath = fs.BuildPath(fs.GetParentFolderName(WScript.ScriptFullName), "check_every_two_hours.ps1")
command = "powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File """ & scriptPath & """"
shell.Run command, 0, True
