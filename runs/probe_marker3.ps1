$f='d:\Codes\KUET PROJECTS\3 1 Embedded\DocsMaker\DocumentScanner\app.py'
$b = [System.IO.File]::ReadAllText($f, [System.Text.Encoding]::UTF8)
$anchor = "# Grid-builder helpers (used by _render_live)"
$pos = $b.IndexOf($anchor)
# Show 30 chars before anchor (so we can find unique anchor prefix)
$pre = $b.Substring([Math]::Max(0, $pos-60), 60)
Write-Output ("PRE 60 >>>{0}<<<" -f $pre)
$post = $b.Substring($pos, [Math]::Min(80, $b.Length-$pos))
Write-Output ("POST 80 >>>{0}<<<" -f $post)
