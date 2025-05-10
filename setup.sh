#!/bin/bash
set -e

echo "Checking system packages"
required_pkgs=(python3-venv python3-full)
need_update=false
for pkg in "${required_pkgs[@]}"; do
    if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then
        need_update=true
        break
    fi
done

if [ "$need_update" = true ]; then
    echo "Installing missing packages: ${required_pkgs[*]}"
    sudo apt-get update
    sudo apt-get install -y "${required_pkgs[@]}"
else
    echo "All required system packages are already installed."
fi

echo "Setting up Python virtual environment"
VENV_DIR="$HOME/finetune-venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

echo "Activating virtual environment"
source "$VENV_DIR/bin/activate"

echo "Upgrading pip, setuptools, and wheel"
pip install --upgrade pip setuptools wheel

echo "Installing required Python packages"
# PyTorch with CUDA 12.1 support
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121

# Hugging Face stack + essentials
# pip install --upgrade transformers datasets peft accelerate bitsandbytes huggingface_hub sentencepiece protobuf
pip install transformers==4.39.3 \
            peft==0.10.0 \
            datasets==2.18.0 \
            accelerate==0.28.0 \
            bitsandbytes==0.43.1 \
            sentencepiece==0.1.99 \
            protobuf==4.25.3 \
            huggingface_hub==0.22.2

echo "Verifying PyTorch CUDA and GPU availability"
python3 - << 'EOF'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Detected GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: CUDA not available. Check driver or PyTorch version.")
EOF

echo "Setup complete. Virtual environment is ready at $VENV_DIR, execute 'source ~/finetune-venv/bin/activate' to switch to the virtual environment."