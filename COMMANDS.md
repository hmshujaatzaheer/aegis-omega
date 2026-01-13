# AEGIS-Ω: Complete Command Reference

## 🚀 Quick Start

### Windows (PowerShell)

```powershell
# Clone repository
git clone https://github.com/YOUR_USERNAME/aegis-omega.git
cd aegis-omega

# Run setup script (creates venv, installs deps, runs tests)
.\scripts\setup.ps1

# Or manual setup:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

### Linux/Mac (Bash)

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/aegis-omega.git
cd aegis-omega

# Run setup script
chmod +x scripts/setup.sh
./scripts/setup.sh

# Or manual setup:
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## 📦 Installation Options

### Basic Installation

```bash
pip install -e .
```

### Development Installation (includes test/lint tools)

```bash
pip install -e ".[dev,test,docs]"
```

### With CUDA Support (for GPU acceleration)

```bash
pip install -e .
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=aegis_omega --cov-report=html

# Run specific test file
pytest tests/test_aegis_omega.py -v

# Run specific test class
pytest tests/test_aegis_omega.py::TestStreamingMFOTL -v

# Run specific test
pytest tests/test_aegis_omega.py::TestStreamingMFOTL::test_bounded_memory_guarantee -v
```

---

## 🎯 Running Examples

```bash
# Example 1: EU AI Act Compliance
python examples/01_eu_ai_act_compliance.py

# Example 2: Zero-Knowledge Proofs
python examples/02_zero_knowledge_proofs.py

# Example 3: Categorical Composition
python examples/03_categorical_composition.py
```

---

## 🔧 Development Commands

### Code Quality

```bash
# Format code with Black
black src/ tests/

# Sort imports
isort src/ tests/

# Lint with Ruff
ruff check src/ tests/

# Type checking
mypy src/aegis_omega --ignore-missing-imports

# Run all quality checks
black --check src/ tests/ && isort --check-only src/ tests/ && ruff check src/ tests/ && mypy src/aegis_omega
```

### Building

```bash
# Build package
python -m build

# Check package
twine check dist/*
```

---

## 📄 LaTeX Compilation

```bash
# Navigate to docs directory
cd docs

# Compile PDF (requires pdflatex, biber)
pdflatex AEGIS_Omega_Proposal.tex
biber AEGIS_Omega_Proposal
pdflatex AEGIS_Omega_Proposal.tex
pdflatex AEGIS_Omega_Proposal.tex

# Or use latexmk
latexmk -pdf AEGIS_Omega_Proposal.tex
```

---

## 🌐 Git Commands

### Initial Setup

```bash
# Initialize repository
git init

# Add remote
git remote add origin https://github.com/YOUR_USERNAME/aegis-omega.git

# Initial commit
git add .
git commit -m "Initial commit: AEGIS-Ω Universal AI Safety Protocol"

# Push to main
git branch -M main
git push -u origin main
```

### Daily Workflow

```bash
# Check status
git status

# Stage changes
git add .

# Commit with message
git commit -m "feat: add streaming MFOTL optimization"

# Push changes
git push

# Pull latest
git pull
```

### Branching

```bash
# Create feature branch
git checkout -b feature/streaming-mfotl

# Switch back to main
git checkout main

# Merge feature branch
git merge feature/streaming-mfotl

# Delete branch after merge
git branch -d feature/streaming-mfotl
```

---

## 🐳 Docker Commands (Optional)

```dockerfile
# Build image
docker build -t aegis-omega .

# Run container
docker run -it aegis-omega

# Run with volume mount
docker run -it -v $(pwd):/app aegis-omega

# Run tests in container
docker run aegis-omega pytest tests/ -v
```

---

## 📊 Benchmarking

```bash
# Install benchmark dependencies
pip install pytest-benchmark

# Run benchmarks
pytest benchmarks/ --benchmark-json=results.json

# Compare with previous run
pytest benchmarks/ --benchmark-compare
```

---

## 🔐 Security Scanning

```bash
# Install security tools
pip install bandit safety

# Run Bandit (code security)
bandit -r src/ -ll

# Check dependencies
safety check
```

---

## 📚 Documentation

```bash
# Install docs dependencies
pip install sphinx sphinx-rtd-theme

# Build documentation
cd docs
make html

# View documentation
open _build/html/index.html  # Mac
xdg-open _build/html/index.html  # Linux
start _build/html/index.html  # Windows
```

---

## 🛠 Troubleshooting

### Common Issues

**Virtual environment not activating (Windows):**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Import errors:**
```bash
# Reinstall in development mode
pip install -e . --force-reinstall
```

**Tests failing with import errors:**
```bash
# Ensure you're in the project root
cd /path/to/aegis-omega
pip install -e ".[test]"
```

**LaTeX compilation errors:**
```bash
# Install required packages (Ubuntu)
sudo apt-get install texlive-full

# Or minimal installation
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended
```

---

## 📝 Quick Code Examples

### Basic Usage

```python
from aegis_omega import create_aegis_for_eu_ai_act, AIAction
from datetime import datetime

# Create AEGIS instance with EU AI Act specs
aegis = create_aegis_for_eu_ai_act()

# Process an AI action
action = AIAction(
    action_id="action_001",
    action_type="generate_text",
    timestamp=datetime.now(),
    input_data={"prompt": "Hello"},
    output_data={"response": "Hi there!"},
    model_id="gpt-4",
    metadata={"logged": True}
)

# Get safety verdict
verdict = aegis.process_action(action)
print(f"Safe: {verdict.satisfied}")
```

### Generate Zero-Knowledge Proof

```python
from aegis_omega.zkml import FoldedZKMLProver, FoldedZKMLVerifier
from aegis_omega.mfotl import MFOTLBuilder

# Create specification
spec = MFOTLBuilder().always("safe", 0, 10).build()

# Generate proof
prover = FoldedZKMLProver()
prover.compile_formula(spec)
proof = prover.generate_proof(witness_data)

# Verify proof
verifier = FoldedZKMLVerifier()
verifier.register_formula(spec)
is_valid = verifier.verify(proof)
```

### Compose Safe Systems

```python
from aegis_omega.category_theory import SafePipeline, SafetyObject
from aegis_omega.mfotl import MFOTLBuilder

# Create pipeline
pipeline = SafePipeline(name="MyPipeline")

# Add stages
spec1 = MFOTLBuilder().always("validated", 0, 1).build()
spec2 = MFOTLBuilder().always("processed", 0, 1).build()

pipeline.add_stage(SafetyObject("Validator", spec1))
pipeline.add_stage(SafetyObject("Processor", spec2))

# Verify composition
assert pipeline.verify()
```

---

## 🎓 Research Commands

### Run Full Evaluation

```bash
# Run all benchmarks and tests
pytest tests/ -v --cov=aegis_omega
pytest benchmarks/ --benchmark-json=results.json

# Generate coverage report
pytest --cov=aegis_omega --cov-report=html
```

### Reproduce Paper Results

```bash
# Table 1: Streaming throughput
python benchmarks/streaming_throughput.py

# Table 2: Memory usage
python benchmarks/memory_analysis.py

# Table 3: Proof generation time
python benchmarks/zkml_benchmarks.py
```

---

## 📫 Contact & Support

- **Repository**: https://github.com/YOUR_USERNAME/aegis-omega
- **Issues**: https://github.com/YOUR_USERNAME/aegis-omega/issues
- **Documentation**: https://aegis-omega.readthedocs.io

---

**AEGIS-Ω: The TCP/IP of AI Safety Infrastructure**
*Universal AI Safety Protocol - ETH Zürich PhD Research*
