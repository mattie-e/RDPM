# RDPM Model Checkpoints

This directory contains trained model weights for the RDPM system.

## 🔒 Privacy and Access Notice

**Important**: The model checkpoints are trained on medical imaging data and are subject to privacy regulations and data access restrictions.

### Access Requirements:
- ✅ **Non-commercial research only**
- ✅ **Formal collaboration agreement required**
- ✅ **Institutional approval may be necessary**
- ✅ **Contact authors through official channels**

## Expected Files

The demo script expects the following checkpoint files:

- `rdpm_hybrid_best.pth` - Best checkpoint for the RDPM Hybrid Model
- `rdpm_image_only_best.pth` - Best checkpoint for the CT-only Model  
- `rdpm_image_w_attn_best.pth` - Best checkpoint for the CT with Attention Model

## Checkpoint Format

Each checkpoint file should contain a PyTorch state dict with the following structure:

```python
{
    'model_state_dict': ...,  # Model weights
    'epoch': ...,             # Training epoch (optional)
    'best_auc': ...,          # Best validation AUC (optional)
    'train_loss': ...,        # Final training loss (optional)
    'optimizer_state_dict': ...,  # Optimizer state (optional)
}
```

## Usage

The demo script will automatically load the appropriate checkpoint based on the model type:

```bash
# Will load checkpoints/rdpm_hybrid_best.pth
python demo.py --model hybrid --data_dir /path/to/data --json_file /path/to/metadata.json

# Will load checkpoints/rdpm_image_only_best.pth  
python demo.py --model image_only --data_dir /path/to/data --json_file /path/to/metadata.json

# Will load checkpoints/rdpm_image_w_attn_best.pth
python demo.py --model image_w_attn --data_dir /path/to/data --json_file /path/to/metadata.json
```

You can also specify a custom checkpoint path:

```bash
python demo.py --model hybrid --checkpoint /path/to/custom/checkpoint.pth --data_dir /path/to/data --json_file /path/to/metadata.json
```

## Note

**These checkpoint files are not included in the repository due to privacy and data access restrictions.**

### To Obtain Checkpoints:
1. **Contact the corresponding authors** for collaboration opportunities
2. **Submit formal request** with research project details
3. **Sign data use agreement** for non-commercial research
4. **Obtain institutional approvals** if required
5. **Follow data protection protocols** as specified in agreements

### Alternative:
Train your own models using the provided training utilities with your own dataset (subject to appropriate ethical approvals and data use permissions).
