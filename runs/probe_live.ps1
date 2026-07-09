$f = 'd:\Codes\KUET PROJECTS\3 1 Embedded\DocsMaker\DocumentScanner\app.py'
$lines = Get-Content $f -Encoding utf8
$start = -1; $end = -1
for ($i = 0; $i -lt $lines.Count; $i++) {
  if ($lines[$i] -match '    def _render_live\(self') { $start = $i }
  if ($start -ne -1 -and $lines[$i] -match '^    def [A-Za-z_]') {
    if ($i -gt $start) { $end = $i - 1; break }
  }
}
if ($end -eq -1) { $end = $lines.Count - 1 }
Write-Host "=== _render_live: lines $($start+1)..$($end+1) ==="
for ($i = $start; $i -le $end; $i++) {
  Write-Host ("L{0:d4}: {1}" -f ($i+1), $lines[$i])
}
