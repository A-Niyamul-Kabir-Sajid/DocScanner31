$f='d:\Codes\KUET PROJECTS\3 1 Embedded\DocsMaker\DocumentScanner\app.py'
$b = [System.IO.File]::ReadAllText($f, [System.Text.Encoding]::UTF8)
$start_marker = "def _render_camera_offline("
$end_marker = "def _render_live_fallback(self, frame: np.ndarray) -> np.ndarray:"
$si = $b.IndexOf($start_marker)
$ei = $b.IndexOf($end_marker)
Write-Output ("start at {0}, end at {1}, length={2}" -f $si, $ei, ($ei-$si))
Write-Output ("---- BYTES ----")
Write-Output $b.Substring($si, $ei-$si)
