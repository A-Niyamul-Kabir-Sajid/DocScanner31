$ErrorActionPreference = 'Stop'
Select-String -Path 'd:\Codes\KUET PROJECTS\3 1 Embedded\DocsMaker\DocumentScanner\app.py' `
  -Pattern 'args\.(scan_mode|no_quality_gate|host|web|auto_capture)' |
  ForEach-Object { "L$($_.LineNumber): $($_.Line)" }
