# RDPM Demo Quick Start Guide

This guide helps you quickly get started with the RDPM demo script.

## 🚀 Quick Test (No Data Required)

```bash
# Show sample data structure
python demo.py --show_sample_data

# Show help
python demo.py --help
```

## 📋 Prerequisites Checklist

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Model Checkpoints

⚠️ **Important Privacy Notice**: Model checkpoints are subject to data access restrictions.

Place trained model weights in the `checkpoints/` directory:

- `checkpoints/rdpm_hybrid_best.pth` (for Hybrid Model)
- `checkpoints/rdpm_image_only_best.pth` (for CT-only Model)  
- `checkpoints/rdpm_image_w_attn_best.pth` (for CT+Attention Model)

**How to Obtain Checkpoints**:
1. **Contact authors** for collaborative research opportunities
2. **Non-commercial research use only**
3. **Formal agreement required** (data use agreement/collaboration agreement)
4. **Institutional approval** may be necessary
5. **Research ethics approval** may be required

**Note**: Checkpoint files contain trained weights derived from medical data and cannot be freely distributed.

### 3. Prepare Your Data

⚠️ **Privacy Notice**: If using medical imaging data, ensure you have appropriate ethical approvals and data use permissions.

Organize your data following this structure:
```
your_data/
├── images/
│   ├── patient001.nii.gz
│   ├── patient002.nii.gz
│   └── ...
├── masks/
│   ├── patient001_mask.nii.gz
│   ├── patient002_mask.nii.gz
│   └── ...
└── metadata.json
```

### 4. Create metadata.json
```json
{
  "internal_test": [
    {
      "filename": "patient001",
      "label": 0,
      "DM": 0,
      "maxdiameter": 3.5,
      "HTN": 1,
      "age": 65,
      "eGFR": 75
    }
  ]
}
```

## 🎯 Running the Demo

### Basic Usage
```bash
# Run with RDPM Hybrid Model (recommended)
python demo.py \
  --model hybrid \
  --data_dir /path/to/your_data \
  --json_file /path/to/your_data/metadata.json

# Run with CT-only Model
python demo.py \
  --model image_only \
  --data_dir /path/to/your_data \
  --json_file /path/to/your_data/metadata.json

# Run with CT+Attention Model
python demo.py \
  --model image_w_attn \
  --data_dir /path/to/your_data \
  --json_file /path/to/your_data/metadata.json
```

### Advanced Options
```bash
# Custom checkpoint and output directory
python demo.py \
  --model hybrid \
  --checkpoint /path/to/custom/checkpoint.pth \
  --data_dir /path/to/your_data \
  --json_file /path/to/your_data/metadata.json \
  --output_dir my_results \
  --batch_size 8 \
  --device cuda

# Use different data split
python demo.py \
  --model hybrid \
  --data_dir /path/to/your_data \
  --json_file /path/to/your_data/metadata.json \
  --split external_test
```

## 📊 Expected Output

The demo will create a `demo_results/` directory (or your specified output directory) containing:

- `{model_type}_predictions.csv` - Detailed predictions for each sample
- `{model_type}_summary.json` - Summary metrics and timing information

### Console Output Example:
```
🚀 RDPM Demo Starting...
✓ RDPM Demo initialized for model type: hybrid
✓ Using device: cuda
🔄 Loading hybrid model...
📁 Loading checkpoint from: checkpoints/rdpm_hybrid_best.pth
✓ Checkpoint loaded successfully
✓ Model loaded successfully
   - Total parameters: 2,456,789
   - Trainable parameters: 2,456,789
🔄 Setting up data loader...
✓ Data loader created successfully
   - Number of batches: 25
   - Total samples: 100
🔄 Running inference...
Processing batches: 100%|██████████| 25/25 [00:45<00:00,  1.81s/it]

📊 Inference Results Summary
==================================================
Model Type: HYBRID
Total Samples: 100

⏱️ Timing Information:
   - Total inference time: 45.23 seconds
   - Average per batch: 1.81 seconds
   - Average per sample: 0.452 seconds

🎯 Prediction Distribution:
   - Predicted positive (rapid decline): 23 (23.0%)
   - Predicted negative (stable): 77 (77.0%)

📈 Performance Metrics:
   - Accuracy: 0.8500
   - AUC: 0.8923
   - Sensitivity: 0.7826
   - Specificity: 0.8701
   - Precision: 0.6923
   - F1-Score: 0.7347

✓ Detailed predictions saved to: demo_results/hybrid_predictions.csv
✓ Summary metrics saved to: demo_results/hybrid_summary.json

🎉 Demo completed successfully!
Results saved to: demo_results
```

## 🔧 Troubleshooting

### Common Issues:

1. **Missing dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Missing checkpoint files**:
   - Check if files exist in `checkpoints/` directory
   - **For access to trained weights**: Contact authors for collaboration agreements
   - **Privacy requirement**: Formal data use agreement needed for non-commercial research
   - Or specify custom path with `--checkpoint`

3. **CUDA out of memory**:
   ```bash
   # Use CPU instead
   python demo.py --device cpu ...
   
   # Or reduce batch size
   python demo.py --batch_size 1 ...
   ```

4. **Data loading errors**:
   - Check data directory structure
   - Verify metadata.json format
   - Ensure image/mask files exist

### Validation Commands:
```bash
# Verify installation
python -c "import torch, monai, sklearn; print('All dependencies installed!')"
```

## 💡 Tips

- Start with `--show_sample_data` to understand the expected format
- Use `--batch_size 1` for debugging or memory issues
- Check `demo_results/` for detailed outputs
- Monitor GPU memory usage with `nvidia-smi` if using CUDA
- Use `--device cpu` if you don't have a GPU

## 📞 Support

If you encounter issues:
1. Check this guide
2. Verify your data format matches the sample structure
3. Check that all dependencies are installed correctly

## 🤝 Collaboration and Data Access

For access to trained models and datasets:
- **Research collaboration**: Contact corresponding authors through institutional channels
- **Requirements**: Non-commercial research use only
- **Process**: Formal data use agreement and institutional approvals required
- **Timeline**: Allow sufficient time for legal and ethical review processes
