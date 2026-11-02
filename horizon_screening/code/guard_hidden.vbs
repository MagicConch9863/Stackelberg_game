' Windowless entry for the existing minute monitor; no solver is launched here.
Option Explicit
Dim fs, shell, root, scriptPath, command
Set fs = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
root = fs.GetParentFolderName(fs.GetParentFolderName(WScript.ScriptFullName))
scriptPath = fs.BuildPath(fs.GetParentFolderName(root), "monte_carlo_3seeds\code\guard.ps1")
command = "powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File """ & scriptPath & """"
shell.Run command, 0, True
