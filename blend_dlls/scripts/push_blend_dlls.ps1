param(
  [string]$RepoUrl = "https://github.com/sin-boo/blender_optmazation-for-large-models-frame-gen",
  [string]$Branch = "not-wroking",
  [string]$SourceDir = "C:\Users\FT.ZQ\Desktop\dllsstff\dllsstff"
)

$ErrorActionPreference = 'Stop'

function Resolve-GitPath {
  $candidates = @(
    "$env:LOCALAPPDATA\Programs\Git\cmd\git.exe",
    "$env:ProgramFiles\Git\cmd\git.exe",
    "$env:ProgramFiles(x86)\Git\cmd\git.exe"
  )
  foreach ($c in $candidates) {
    if (Test-Path $c) { return $c }
  }
  throw "git.exe not found in expected locations."
}

$git = Resolve-GitPath
Write-Host "Using git at: $git"

$desktop = [Environment]::GetFolderPath('Desktop')
$repoRoot = Join-Path $desktop 'blender_optmazation-for-large-models-frame-gen'

if (-not (Test-Path $repoRoot)) {
  & $git clone $RepoUrl $repoRoot
} else {
  Write-Host "Repo already exists at $repoRoot"
}

& $git -C $repoRoot remote -v
& $git -C $repoRoot fetch --all --prune
& $git -C $repoRoot checkout -B $Branch

$dest = Join-Path $repoRoot 'blend_dlls'
if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }

Write-Host "Copying from $SourceDir to $dest"
Copy-Item -Recurse -Force $SourceDir $dest

& $git -C $repoRoot add blend_dlls
$status = & $git -C $repoRoot status --porcelain
if ($status) {
  & $git -C $repoRoot commit -m "Add blend_dlls (renamed from dllsstff) - self-sufficient screen capture app"
} else {
  Write-Host "No changes to commit."
}

try {
  & $git -C $repoRoot push -u origin $Branch
  Write-Host "Pushed to origin/$Branch successfully."
} catch {
  Write-Warning "Push failed. You may need to authenticate (GitHub)."
  Write-Host "Options:"
  Write-Host "  - If you have Git Credential Manager, run git commands again and sign in when prompted."
  Write-Host "  - Or set a Personal Access Token as GITHUB_TOKEN and run:"
  Write-Host "      $env:GITHUB_TOKEN='...'"
  Write-Host "      git -C `"$repoRoot`" remote set-url origin https://$env:GITHUB_TOKEN@github.com/sin-boo/blender_optmazation-for-large-models-frame-gen.git"
  Write-Host "      git -C `"$repoRoot`" push -u origin $Branch"
  exit 1
}

