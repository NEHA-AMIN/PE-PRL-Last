#!/bin/bash
# =============================================================================
# Setup Script: Download ETTh1 Dataset for Informer2020
# =============================================================================

echo "============================================="
echo "Setting up ETTh1 Dataset"
echo "============================================="

# Create data directory
mkdir -p data/ETT

# Download ETTh1 dataset
echo "Downloading ETTh1.csv..."
wget -O data/ETT/ETTh1.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv

if [ -f "data/ETT/ETTh1.csv" ]; then
    echo "✓ ETTh1.csv downloaded successfully"
    echo "File size: $(du -h data/ETT/ETTh1.csv | cut -f1)"
    echo "Lines: $(wc -l < data/ETT/ETTh1.csv)"
else
    echo "✗ Failed to download ETTh1.csv"
    exit 1
fi

echo "============================================="
echo "Dataset setup complete!"
echo "============================================="

# Made with Bob
