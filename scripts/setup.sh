#!/bin/bash
# ============================================================
#  Caribbean AI Academy - Setup Script
#  "Setting up yuh workshop, one step at a time!"
# ============================================================

set -e

# Colors for that Caribbean vibes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}  CARIBBEAN AI ACADEMY - Setup Script${NC}"
echo -e "${CYAN}  'Wi setting up de ting proper, no shortcuts!'${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# Get the project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo -e "${YELLOW}Project root: ${PROJECT_ROOT}${NC}"
echo ""

# ----------------------------------------------------------
# Step 1: Check Python installation
# ----------------------------------------------------------
echo -e "${BOLD}Step 1: Checking for Python... like checking if de coconut ripe${NC}"

PYTHON_CMD=""

if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
fi

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}  ERROR: Python not found! Yuh need Python 3.8+ to proceed.${NC}"
    echo -e "${RED}  Install it from https://www.python.org/ and come back, yuh hear?${NC}"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo -e "${GREEN}  Found: ${PYTHON_VERSION} -- nice, we good to go!${NC}"

# Check minimum version (3.8+)
PYTHON_MAJOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)")

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo -e "${RED}  ERROR: Python 3.8+ required, but yuh have ${PYTHON_VERSION}.${NC}"
    echo -e "${RED}  Upgrade yuh Python and try again!${NC}"
    exit 1
fi

echo -e "${GREEN}  Python version check passed! We cookin' now!${NC}"
echo ""

# ----------------------------------------------------------
# Step 2: Create virtual environment
# ----------------------------------------------------------
echo -e "${BOLD}Step 2: Creating virtual environment... building we own little island${NC}"

VENV_DIR="$PROJECT_ROOT/venv"

if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}  Virtual environment already exists at ${VENV_DIR}${NC}"
    echo -e "${YELLOW}  Using existing venv -- no need to reinvent de wheel!${NC}"
else
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo -e "${GREEN}  Virtual environment created at ${VENV_DIR}${NC}"
    echo -e "${GREEN}  Fresh like morning dew on ah hibiscus!${NC}"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}  Virtual environment activated!${NC}"
echo ""

# ----------------------------------------------------------
# Step 3: Upgrade pip
# ----------------------------------------------------------
echo -e "${BOLD}Step 3: Upgrading pip... sharpening we tools${NC}"
pip install --upgrade pip --quiet
echo -e "${GREEN}  pip upgraded! Sharp like ah cutlass!${NC}"
echo ""

# ----------------------------------------------------------
# Step 4: Install requirements
# ----------------------------------------------------------
echo -e "${BOLD}Step 4: Installing dependencies... stocking up de pantry${NC}"

REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"

if [ -f "$REQUIREMENTS_FILE" ]; then
    echo -e "${YELLOW}  Installing from requirements.txt... dis might take a lil while.${NC}"
    echo -e "${YELLOW}  Go make some cocoa tea while yuh wait!${NC}"
    echo ""
    pip install -r "$REQUIREMENTS_FILE" --quiet 2>&1 | tail -5
    echo ""
    echo -e "${GREEN}  All dependencies installed! De pantry FULL!${NC}"
else
    echo -e "${RED}  WARNING: requirements.txt not found at ${REQUIREMENTS_FILE}${NC}"
    echo -e "${YELLOW}  Installing basic packages instead...${NC}"
    pip install numpy pandas scikit-learn matplotlib seaborn --quiet
    echo -e "${GREEN}  Basic packages installed!${NC}"
fi
echo ""

# ----------------------------------------------------------
# Step 5: Generate Caribbean datasets
# ----------------------------------------------------------
echo -e "${BOLD}Step 5: Generating Caribbean datasets... planting de data seeds!${NC}"

DATASET_SCRIPT="$PROJECT_ROOT/datasets/caribbean_data/generate_datasets.py"

if [ -f "$DATASET_SCRIPT" ]; then
    $PYTHON_CMD "$DATASET_SCRIPT"
    echo -e "${GREEN}  Datasets generated! Fresh data ready for analysis!${NC}"
else
    echo -e "${RED}  WARNING: generate_datasets.py not found at ${DATASET_SCRIPT}${NC}"
    echo -e "${RED}  Yuh might need to check de repo, something missing!${NC}"
fi
echo ""

# ----------------------------------------------------------
# Step 6: Verify installation
# ----------------------------------------------------------
echo -e "${BOLD}Step 6: Running quick verification... making sure everyting correct${NC}"

$PYTHON_CMD -c "
import sys
packages = ['numpy', 'pandas', 'sklearn', 'matplotlib']
missing = []
for pkg in packages:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f'  WARNING: Missing packages: {missing}')
    print('  Run: pip install -r requirements.txt')
else:
    print('  All core packages verified! Everyting irie!')
"

# Check datasets exist
DATASET_DIR="$PROJECT_ROOT/datasets/caribbean_data"
CSV_COUNT=$(find "$DATASET_DIR" -name "*.csv" 2>/dev/null | wc -l)
echo -e "  Found ${CSV_COUNT} CSV datasets in ${DATASET_DIR}"
echo ""

# ----------------------------------------------------------
# Done!
# ----------------------------------------------------------
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}  SETUP COMPLETE! Yuh ready to start learning AI!${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""
echo -e "${GREEN}  To activate the virtual environment next time:${NC}"
echo -e "${BOLD}    source venv/bin/activate${NC}"
echo ""
echo -e "${GREEN}  To start learning:${NC}"
echo -e "${BOLD}    cd 01_primary_prep/     # For beginners${NC}"
echo -e "${BOLD}    cd 02_forms_1_to_3/     # For intermediate${NC}"
echo -e "${BOLD}    cd 03_forms_4_to_5/     # For advanced${NC}"
echo ""
echo -e "${YELLOW}  'De journey of a thousand lines of code${NC}"
echo -e "${YELLOW}   start with a single print(\"Hello, Caribbean!\")' ${NC}"
echo ""
echo -e "${CYAN}  Now go learn something great, yuh hear! ${NC}"
echo -e "${CYAN}============================================================${NC}"
