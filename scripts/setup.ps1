# AEGIS-Ω Setup Script for Windows PowerShell
# Universal AI Safety Protocol - ETH Zürich PhD Research Project
# Author: H M Shujaat Zaheer
# Supervisor: Prof. Dr. David Basin

param(
    [switch]$Development,
    [switch]$WithCuda,
    [switch]$SkipTests,
    [string]$PythonPath = "python"
)

$ErrorActionPreference = "Stop"

# Color output functions
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) { Write-Output $args }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Write-Header($message) {
    Write-ColorOutput Cyan "`n=========================================="
    Write-ColorOutput Cyan "  $message"
    Write-ColorOutput Cyan "==========================================`n"
}

function Write-Success($message) {
    Write-ColorOutput Green "[✓] $message"
}

function Write-Info($message) {
    Write-ColorOutput Yellow "[i] $message"
}

function Write-Error($message) {
    Write-ColorOutput Red "[✗] $message"
}

# Banner
Write-Host @"
    _    _____ ____ ___ ____        ___  __  __ _____ ____    _    
   / \  | ____/ ___|_ _/ ___|      / _ \|  \/  | ____/ ___|  / \   
  / _ \ |  _|| |  _ | |\___ \ ____| | | | |\/| |  _|| |  _  / _ \  
 / ___ \| |__| |_| || | ___) |____| |_| | |  | | |__| |_| |/ ___ \ 
/_/   \_\_____\____|___|____/      \___/|_|  |_|_____\____/_/   \_\
                                                                    
        Universal AI Safety Protocol - Setup Script
        TCP/IP of AI Safety Infrastructure
"@ -ForegroundColor Cyan

Write-Header "AEGIS-Ω Environment Setup"

# Check Python version
Write-Info "Checking Python installation..."
try {
    $pythonVersion = & $PythonPath --version 2>&1
    if ($pythonVersion -match "Python (\d+)\.(\d+)") {
        $major = [int]$Matches[1]
        $minor = [int]$Matches[2]
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
            Write-Error "Python 3.10+ required. Found: $pythonVersion"
            exit 1
        }
        Write-Success "Python version: $pythonVersion"
    }
} catch {
    Write-Error "Python not found. Please install Python 3.10+"
    exit 1
}

# Create virtual environment
Write-Header "Creating Virtual Environment"
$venvPath = ".venv"

if (Test-Path $venvPath) {
    Write-Info "Virtual environment already exists at $venvPath"
    $response = Read-Host "Recreate? (y/N)"
    if ($response -eq "y" -or $response -eq "Y") {
        Remove-Item -Recurse -Force $venvPath
        & $PythonPath -m venv $venvPath
        Write-Success "Virtual environment recreated"
    }
} else {
    & $PythonPath -m venv $venvPath
    Write-Success "Virtual environment created at $venvPath"
}

# Activate virtual environment
Write-Info "Activating virtual environment..."
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    . $activateScript
    Write-Success "Virtual environment activated"
} else {
    Write-Error "Activation script not found"
    exit 1
}

# Upgrade pip
Write-Header "Upgrading pip"
python -m pip install --upgrade pip wheel setuptools
Write-Success "pip upgraded"

# Install dependencies
Write-Header "Installing Dependencies"

if ($Development) {
    Write-Info "Installing in development mode with all extras..."
    pip install -e ".[dev,test,docs]"
} else {
    Write-Info "Installing in production mode..."
    pip install -e .
}

if ($WithCuda) {
    Write-Info "Installing CUDA-enabled packages..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
}

Write-Success "Dependencies installed"

# Run tests if not skipped
if (-not $SkipTests) {
    Write-Header "Running Tests"
    try {
        python -m pytest tests/ -v --tb=short
        Write-Success "All tests passed"
    } catch {
        Write-Error "Some tests failed. Check output above."
    }
}

# Verify installation
Write-Header "Verifying Installation"
try {
    python -c "from aegis_omega import AEGISOmega, create_aegis; print('Core module: OK')"
    python -c "from aegis_omega.mfotl import EUAIActSpecifications; print('MFOTL module: OK')"
    python -c "from aegis_omega.zkml import FoldedZKMLProver; print('ZKML module: OK')"
    python -c "from aegis_omega.category_theory import SafetyCategory; print('Category Theory module: OK')"
    Write-Success "All modules verified"
} catch {
    Write-Error "Module verification failed"
    exit 1
}

# Setup complete
Write-Header "Setup Complete!"

Write-Host @"
AEGIS-Ω has been successfully installed!

Quick Start:
-----------
1. Activate environment:  .\.venv\Scripts\Activate.ps1

2. Run EU AI Act compliance demo:
   python examples/01_eu_ai_act_compliance.py

3. Run Zero-Knowledge Proof demo:
   python examples/02_zero_knowledge_proofs.py

4. Import in your code:
   from aegis_omega import create_aegis_for_eu_ai_act
   aegis = create_aegis_for_eu_ai_act()

Documentation: https://github.com/shujaat-zaheer/aegis-omega
Issues: https://github.com/shujaat-zaheer/aegis-omega/issues

"@ -ForegroundColor Green

# Git setup (optional)
$setupGit = Read-Host "Initialize Git repository? (y/N)"
if ($setupGit -eq "y" -or $setupGit -eq "Y") {
    Write-Header "Initializing Git Repository"
    
    if (-not (Test-Path ".git")) {
        git init
        Write-Success "Git repository initialized"
    }
    
    # Create .gitignore if not exists
    if (-not (Test-Path ".gitignore")) {
        @"
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Testing
.pytest_cache/
.coverage
htmlcov/

# Documentation
docs/_build/

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db

# LaTeX auxiliary files
*.aux
*.log
*.out
*.toc
*.synctex.gz
*.fls
*.fdb_latexmk

# Sensitive data
*.key
*.pem
secrets/
"@ | Out-File -Encoding utf8 .gitignore
        Write-Success ".gitignore created"
    }
    
    git add .
    git commit -m "Initial commit: AEGIS-Ω Universal AI Safety Protocol"
    Write-Success "Initial commit created"
    
    Write-Info "To push to GitHub:"
    Write-Host "  git remote add origin https://github.com/YOUR_USERNAME/aegis-omega.git"
    Write-Host "  git branch -M main"
    Write-Host "  git push -u origin main"
}
