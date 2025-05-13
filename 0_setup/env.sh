#!/bin/bash

# Exit on error
set -e

echo "Installing Kaggle"
pip install kaggle

echo "Creating Kaggle Files"
mkdir -p ./.kaggle/
cd ./.kaggle/
touch kaggle.json

echo "Enter your kaggle username: "
read kaggle_username
echo "Enter your kaggle api key: "
read kaggle_key

echo "Writing Kaggle API credentials"
cat <<EOF > ./.kaggle/kaggle.json
{
  "username": "${kaggle_username}",
  "key": "${kaggle_key}"
}
EOF

echo "Setting kaggle cache directory"
export KAGGLEHUB_CACHE=".cache/kagglehub"

echo "Setting hugging face cache directory"
export HF_HOME=".cache/huggingface/hub"
export HF_HUB_CACHE=".cache/huggingface/hub"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "Virtual environment created successfully."
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Get GPU information if available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU information:"
    nvidia-smi
else
    echo "No NVIDIA GPU detected. The model will run on CPU, which will be very slow."
fi