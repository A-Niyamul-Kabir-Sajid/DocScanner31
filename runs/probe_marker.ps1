$f='d:\Codes\KUET PROJECTS\3 1 Embedded\DocsMaker\DocumentScanner\app.py'
$b = [System.IO.File]::ReadAllText($f, [System.Text.Encoding]::UTF8)
$pos = $b.IndexOf("Grid-builder helpers")
Write-Output ("marker pos = {0}" -f $pos)
$start = $pos - 220
$len = 440
Write-Output ("---- slice ----")
Write-Output ($b.Substring($start, $len))
Write-Output ("---- end slice ----")
