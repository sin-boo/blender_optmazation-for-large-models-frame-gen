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

# Remove any embedded git repositories to avoid checkout/push issues
Write-Host "Scanning for embedded .git directories under blend_dlls..."
$gitDirs = Get-ChildItem -Path $dest -Recurse -Force -Directory -Filter ".git" -ErrorAction SilentlyContinue
foreach ($g in $gitDirs) {
  try {
    Write-Host "Removing embedded git dir:" $g.FullName
    Remove-Item -Recurse -Force -LiteralPath $g.FullName
  } catch {
    Write-Warning "Failed to remove embedded git dir: $($g.FullName)"
  }
}

& $git -C $repoRoot add blend_dlls
$status = & $git -C $repoRoot status --porcelain
if ($status) {
  & $git -C $repoRoot commit -m "Add blend_dlls (renamed from dllsstff) - self-sufficient screen capture app"
} else {
  Write-Host "No changes to commit."
}

function Invoke-GitChecked {
  param([string[]]$Args)
  & $git @Args
  if ($LASTEXITCODE -ne 0) { throw "git command failed: $Args" }
}

$pushOk = $true
try {
  Invoke-GitChecked -Args @('-C', $repoRoot, 'push', '-u', 'origin', $Branch)
} catch {
  $pushOk = $false
}

if (-not $pushOk) {
  Write-Warning "Initial push failed, attempting rebase onto origin/$Branch..."
  try {
    Invoke-GitChecked -Args @('-C', $repoRoot, 'fetch', 'origin')
    # Ensure we have the remote branch locally
    & $git -C $repoRoot branch --list $Branch | Out-Null
    Invoke-GitChecked -Args @('-C', $repoRoot, 'pull', '--rebase', 'origin', $Branch)
    Invoke-GitChecked -Args @('-C', $repoRoot, 'push', '-u', 'origin', $Branch)
    $pushOk = $true
  } catch {
    Write-Warning "Rebase push failed, attempting force-with-lease..."
    try {
      Invoke-GitChecked -Args @('-C', $repoRoot, 'push', '--force-with-lease', '-u', 'origin', $Branch)
      $pushOk = $true
    } catch {
      $pushOk = $false
    }
  }
}

if ($pushOk) {
  Write-Host "Pushed to origin/$Branch successfully."
} else {
  Write-Warning "Push failed. You may need to authenticate (GitHub) or resolve conflicts."
  Write-Host "Options:"
  Write-Host "  - If you have Git Credential Manager, run git commands again and sign in when prompted."
  Write-Host "  - Or set a Personal Access Token as GITHUB_TOKEN and run:"
  Write-Host "      $env:GITHUB_TOKEN='...'"
  Write-Host "      git -C `"$repoRoot`" remote set-url origin https://$env:GITHUB_TOKEN@github.com/sin-boo/blender_optmazation-for-large-models-frame-gen.git"
  Write-Host "      git -C `"$repoRoot`" push -u origin $Branch"
  exit 1
}

