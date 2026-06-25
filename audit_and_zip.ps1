# audit_and_zip.ps1 - verify the GTZAN data/repair placement, then build a slim zip.
# Strips: .wav (GTZAN audio) and the phonon dataset artifacts (re-pullable - see
# projects\phonon\data\raw\PROVENANCE.md). Structure preserved; .git excluded.
# Run from the INFO_698_experiments repo root:  .\audit_and_zip.ps1
$ErrorActionPreference = "Stop"
$repo = (Get-Location).Path
$d    = Join-Path $repo "projects\genre\data"
$raw  = Join-Path $d "raw"

function Check($label, $actual, $expected) {
    $pass = "$actual" -eq "$expected"
    $tag  = if ($pass) { "[ OK ]" } else { "[FAIL]" }
    $col  = if ($pass) { "Green" }  else { "Red" }
    Write-Host ("  {0}  {1,-26} {2}  (expect {3})" -f $tag, $label, $actual, $expected) -ForegroundColor $col
}

Write-Host "`n==== AUDIT  $repo ====" -ForegroundColor Cyan
Check "raw wav total"   (Get-ChildItem "$raw\genres_original"   -Recurse -Filter *.wav -EA SilentlyContinue).Count 1000
Check "raw grey total"  (Get-ChildItem "$raw\images_grey_scale" -Recurse -Filter *.png -EA SilentlyContinue).Count 1000
Check "jazz wavs"       (Get-ChildItem "$raw\genres_original\jazz"   -Filter *.wav -EA SilentlyContinue).Count 100
Check "jazz greys"      (Get-ChildItem "$raw\images_grey_scale\jazz" -Filter *.png -EA SilentlyContinue).Count 100
Check "jazz00054 grey"  (Test-Path "$raw\images_grey_scale\jazz\jazz00054.png") $true
Check "features_30_sec" (Test-Path "$raw\features_30_sec.csv") $true
Check "features_3_sec"  (Test-Path "$raw\features_3_sec.csv")  $true
Check "data README.md"  (Test-Path "$d\README.md")             $true
Check "after note"      (Test-Path "$d\after\README.txt")      $true
Check "before snapshot" (Test-Path "$d\before\eda_stats.json") $true
Check "data_doctor.py"  (Test-Path "projects\genre\src\data_doctor.py") $true
Check "raw NOT in git"  ((git status --porcelain "$raw" 2>$null | Measure-Object).Count) 0

# phonon dataset audit: present locally (so training works) but excluded from the zip
$pdata = Join-Path $repo "projects\phonon\data\raw"
Check "phonon dataset pkl" (Test-Path "$pdata\phdos_e3nn_len51max1000_fwin101ord3.pkl") $true
Check "phonon PROVENANCE"  (Test-Path "$pdata\PROVENANCE.md") $true

Write-Host "`n==== BUILD slim zip (structure preserved) ====" -ForegroundColor Cyan
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$out = Join-Path (Split-Path $repo -Parent) "INFO_698_no_wav.zip"
if (Test-Path $out) { Remove-Item $out }
$base = $repo.TrimEnd('\') + '\'

# Path regexes to keep OUT of the zip (re-pullable or huge). Add one line per rule.
# Backslashes are doubled for .NET regex; '\\' matches a literal path separator.
$excludePatterns = @(
    '\\\.git\\',                                                 # git internals
    '\\projects\\phonon\\data\\raw\\.*\.(pkl|torch|csv|zip)$'    # phonon dataset (re-pull via PROVENANCE.md)
)

$files = Get-ChildItem -LiteralPath $repo -Recurse -File | Where-Object {
    $f = $_
    ($_.Extension -ne ".wav") -and
    ($_.FullName -ne $out) -and
    (-not ($excludePatterns | Where-Object { $f.FullName -match $_ }))
}

$zip = [System.IO.Compression.ZipFile]::Open($out, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($f in $files) {
        $rel = $f.FullName.Substring($base.Length)   # keep folder structure
        [void][System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $zip, $f.FullName, $rel,
            [System.IO.Compression.CompressionLevel]::Optimal)
    }
} finally { $zip.Dispose() }
$mb = [math]::Round((Get-Item $out).Length / 1MB, 1)
Write-Host ("  [ OK ]  wrote {0}" -f $out) -ForegroundColor Green
Write-Host ("          {0} files, {1} MB, no .wav, no phonon dataset, structure preserved" -f $files.Count, $mb) -ForegroundColor Green
Write-Host ""
