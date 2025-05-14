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
cd ..
cat <<EOF > ./.kaggle/kaggle.json
{
  "username": "${kaggle_username}",
  "key": "${kaggle_key}"
}
EOF

echo "Setting kaggle cache directory"
mkdir -p ./.kaggle/kagglehub
export KAGGLEHUB_CACHE="./.kaggle/kagglehub"

echo "Setting hugging face cache directory"
mkdir -p ./.cache/huggingface/hub
export HF_HOME="./.cache/huggingface/hub"
export HF_HUB_CACHE="./.cache/huggingface/hub"

echo "Navigating to 0_setup directory"
cd 0_setup

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
