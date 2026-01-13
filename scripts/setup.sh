#!/bin/bash
# AEGIS-Ω Setup Script for Linux/Mac
# Universal AI Safety Protocol - ETH Zürich PhD Research Project
# Author: H M Shujaat Zaheer
# Supervisor: Prof. Dr. David Basin

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Options
DEVELOPMENT=false
WITH_CUDA=false
SKIP_TESTS=false
PYTHON_CMD="python3"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev|--development)
            DEVELOPMENT=true
            shift
            ;;
        --cuda)
            WITH_CUDA=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --python)
            PYTHON_CMD="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dev, --development  Install with development dependencies"
            echo "  --cuda                Install CUDA-enabled PyTorch"
            echo "  --skip-tests          Skip running tests after installation"
            echo "  --python PATH         Specify Python executable (default: python3)"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Banner
echo -e "${CYAN}"
cat << 'EOF'
    _    _____ ____ ___ ____        ___  __  __ _____ ____    _    
   / \  | ____/ ___|_ _/ ___|      / _ \|  \/  | ____/ ___|  / \   
  / _ \ |  _|| |  _ | |\___ \ ____| | | | |\/| |  _|| |  _  / _ \  
 / ___ \| |__| |_| || | ___) |____| |_| | |  | | |__| |_| |/ ___ \ 
/_/   \_\_____\____|___|____/      \___/|_|  |_|_____\____/_/   \_\
                                                                    
        Universal AI Safety Protocol - Setup Script
        TCP/IP of AI Safety Infrastructure
EOF
echo -e "${NC}"

header() {
    echo -e "\n${CYAN}===========================================${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}===========================================${NC}\n"
}

success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

info() {
    echo -e "${YELLOW}[i]${NC} $1"
}

error() {
    echo -e "${RED}[✗]${NC} $1"
}

header "AEGIS-Ω Environment Setup"

# Check Python version
info "Checking Python installation..."
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [[ $MAJOR -lt 3 ]] || [[ $MAJOR -eq 3 && $MINOR -lt 10 ]]; then
    error "Python 3.10+ required. Found: Python $PYTHON_VERSION"
    exit 1
fi
success "Python version: $PYTHON_VERSION"

# Create virtual environment
header "Creating Virtual Environment"
VENV_PATH=".venv"

if [[ -d "$VENV_PATH" ]]; then
    info "Virtual environment already exists at $VENV_PATH"
    read -p "Recreate? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_PATH"
        $PYTHON_CMD -m venv "$VENV_PATH"
        success "Virtual environment recreated"
    fi
else
    $PYTHON_CMD -m venv "$VENV_PATH"
    success "Virtual environment created at $VENV_PATH"
fi

# Activate virtual environment
info "Activating virtual environment..."
source "$VENV_PATH/bin/activate"
success "Virtual environment activated"

# Upgrade pip
header "Upgrading pip"
pip install --upgrade pip wheel setuptools
success "pip upgraded"

# Install dependencies
header "Installing Dependencies"

if $DEVELOPMENT; then
    info "Installing in development mode with all extras..."
    pip install -e ".[dev,test,docs]"
else
    info "Installing in production mode..."
    pip install -e .
fi

if $WITH_CUDA; then
    info "Installing CUDA-enabled packages..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

success "Dependencies installed"

# Run tests if not skipped
if ! $SKIP_TESTS; then
    header "Running Tests"
    if python -m pytest tests/ -v --tb=short; then
        success "All tests passed"
    else
        error "Some tests failed. Check output above."
    fi
fi

# Verify installation
header "Verifying Installation"
python -c "from aegis_omega import AEGISOmega, create_aegis; print('Core module: OK')"
python -c "from aegis_omega.mfotl import EUAIActSpecifications; print('MFOTL module: OK')"
python -c "from aegis_omega.zkml import FoldedZKMLProver; print('ZKML module: OK')"
python -c "from aegis_omega.category_theory import SafetyCategory; print('Category Theory module: OK')"
success "All modules verified"

# Setup complete
header "Setup Complete!"

echo -e "${GREEN}"
cat << EOF
AEGIS-Ω has been successfully installed!

Quick Start:
-----------
1. Activate environment:  source .venv/bin/activate

2. Run EU AI Act compliance demo:
   python examples/01_eu_ai_act_compliance.py

3. Run Zero-Knowledge Proof demo:
   python examples/02_zero_knowledge_proofs.py

4. Import in your code:
   from aegis_omega import create_aegis_for_eu_ai_act
   aegis = create_aegis_for_eu_ai_act()

Documentation: https://github.com/shujaat-zaheer/aegis-omega
Issues: https://github.com/shujaat-zaheer/aegis-omega/issues

EOF
echo -e "${NC}"

# Git setup (optional)
read -p "Initialize Git repository? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    header "Initializing Git Repository"
    
    if [[ ! -d ".git" ]]; then
        git init
        success "Git repository initialized"
    fi
    
    # Create .gitignore if not exists
    if [[ ! -f ".gitignore" ]]; then
        cat > .gitignore << 'GITIGNORE'
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
GITIGNORE
        success ".gitignore created"
    fi
    
    git add .
    git commit -m "Initial commit: AEGIS-Ω Universal AI Safety Protocol"
    success "Initial commit created"
    
    info "To push to GitHub:"
    echo "  git remote add origin https://github.com/YOUR_USERNAME/aegis-omega.git"
    echo "  git branch -M main"
    echo "  git push -u origin main"
fi
