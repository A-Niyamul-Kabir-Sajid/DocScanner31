$f='d:\Codes\KUET PROJECTS\3 1 Embedded\DocsMaker\DocumentScanner\app.py'
$b = [System.IO.File]::ReadAllText($f, [System.Text.Encoding]::UTF8)
$anchor = "# Grid-builder helpers (used by _render_live)"
$pos = $b.IndexOf($anchor)
Write-Output ("anchor pos = {0}" -f $pos)
Write-Output ("---- 240 before ----")
Write-Output ($b.Substring($pos-240, 240))
Write-Output ("---- 240 after anchor ----")
Write-Output ($b.Substring($pos, 240))