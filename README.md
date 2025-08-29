# RDPM: Rapid Decline Prediction Model

Multimodal deep learning model for AI-based decision support in patients with Complex RCC.

## Overview

RDPM combines CT imaging data with clinical parameters to predict the likelihood of rapid kidney function decline. The project includes RDPM and two additional ablated model architectures, where critical components are systematically removed to evaluate their impact on performance:

1. **CT-only Model**: Baseline model using only CT images for prediction
2. **CT with Attention Model**: Enhanced CT model using mask-guided attention mechanisms
3. **RDPM (Hybrid Model)**: Multimodal model combining CT imaging features with clinical parameters

![RDPM Model Architecture](/src/RDPM.png)
*Figure 1: RDPM model architecture showing the hybrid approach combining features from CT imaging with clinical data.*

## Installation

```bash
# Clone the repository
git clone https://github.com/username/RDPM.git
cd RDPM

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## System Requirements

### Software Dependencies
- **Python**: 3.8+ (tested on Python 3.8, 3.9, 3.10, 3.11)
- **Operating Systems**: 
  - Linux (Ubuntu 18.04+, CentOS 7+)
  - macOS 10.15+ (Catalina or later)
  - Windows 10+ (with WSL recommended)
- **CUDA**: 11.0+ (for GPU acceleration, optional but recommended)

### Core Dependencies (with tested versions)
- `torch>=1.10.0` (tested with PyTorch 1.12.0, 1.13.0, 2.0.0)
- `monai>=0.9.0` (tested with MONAI 1.1.0, 1.2.0)
- `numpy>=1.20.0` (tested with numpy 1.21.0, 1.23.0)
- `scikit-learn>=1.0.0` (tested with 1.1.0, 1.2.0)
- `nibabel>=3.2.0` (for NIfTI file handling)
- `pandas>=1.3.0` (for clinical data processing)

### Hardware Requirements
- **RAM**: Minimum 16GB, recommended 32GB+ for large datasets
- **Storage**: 50GB+ free space for model training and data storage
- **GPU**: Optional but recommended - NVIDIA GPU with 8GB+ VRAM for training
- **CPU**: Multi-core processor recommended (8+ cores for optimal performance)

See `requirements.txt` for complete dependency list.

## Installation

### Installation Time
Typical installation time: **5-15 minutes** on a standard desktop computer with good internet connection.

### Steps

```bash
# Clone the repository
git clone https://github.com/username/RDPM.git
cd RDPM

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation (optional)
python -c "import torch, monai, sklearn; print('Installation successful!')"
```

## Demo

### Demo Script Available! 

A complete demo script (`demo.py`) is now available that supports all three model architectures.

📖 **See [DEMO_GUIDE.md](DEMO_GUIDE.md) for detailed usage instructions.**

### Prerequisites
1. **Model checkpoints** - Place trained model weights in `checkpoints/` directory:
   - `checkpoints/rdpm_hybrid_best.pth` (for RDPM Hybrid Model)
   - `checkpoints/rdpm_image_only_best.pth` (for CT-only Model)
   - `checkpoints/rdpm_image_w_attn_best.pth` (for CT with Attention Model)
   
2. **Sample dataset** with the structure shown below in Data Preparation

### Running the Demo

```bash
# Run demo with RDPM Hybrid Model (recommended)
python demo.py --model hybrid --data_dir /path/to/your/data --json_file /path/to/your/metadata.json

# Run demo with CT-only Model
python demo.py --model image_only --data_dir /path/to/your/data --json_file /path/to/your/metadata.json

# Run demo with CT + Attention Model  
python demo.py --model image_w_attn --data_dir /path/to/your/data --json_file /path/to/your/metadata.json

# Show sample data structure
python demo.py --show_sample_data
```

### Demo Options
```bash
python demo.py --help
  --model {hybrid,image_only,image_w_attn}  # Model type to use
  --data_dir DATA_DIR                       # Directory with CT images and masks
  --json_file JSON_FILE                     # JSON file with metadata
  --checkpoint CHECKPOINT                   # Custom checkpoint path (optional)
  --batch_size BATCH_SIZE                   # Batch size for inference (default: 4)
  --split {test,internal_test,external_test} # Data split to use
  --device {auto,cpu,cuda}                  # Device for inference
  --output_dir OUTPUT_DIR                   # Directory to save results
```

### Expected Output
The demo will generate:
- **Inference predictions** for each sample
- **Performance metrics** (if labels are provided)
- **Timing information** 
- **Results files** saved to `demo_results/` directory

### Expected Run Time
- **Demo inference**: ~2-5 minutes on a standard desktop computer with GPU
- **Demo inference (CPU)**: ~10-15 minutes on a standard desktop computer
- **Per sample**: ~0.5-2 seconds depending on hardware

### Sample Data Format

Place your CT images and masks in a dedicated directory and create a JSON file with metadata:

```json
{
  "internal_train": [
    {
      "filename": "patient001",
      "label": 0,
      "DM": 0,
      "maxdiameter": 3.5,
      "HTN": 1,
      "age": 65,
      "eGFR": 75
    },
    ...
  ],
  "internal_test": [
    ...
  ]
}
```

### Training & Inference

#### Demo Script (Available Now!)
```bash
# Run inference with any of the three models
python demo.py --model hybrid --data_dir /path/to/data --json_file /path/to/metadata.json
python demo.py --model image_only --data_dir /path/to/data --json_file /path/to/metadata.json  
python demo.py --model image_w_attn --data_dir /path/to/data --json_file /path/to/metadata.json
```

#### Training Scripts (For Development)
The core training functions are available in `src/training/train.py`. Example usage:

```python
from src.models.hybrid_model import HybridModel
from src.data.loaders import CTImageLoader
from src.training.train import train_model

# Initialize components
model = HybridModel(num_classes=2)
data_loader_obj = CTImageLoader(data_dir="data", json_file="metadata.json")
train_loader = data_loader_obj.get_data_loader(split='train')
val_loader = data_loader_obj.get_data_loader(split='val')

# Train model
trained_model = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    device='cuda'
)
```

## Instructions for Use

### Current Status: **Demo Available!**

This codebase now provides a complete demonstration system with the following components:

#### Available Components:
- **Complete demo script** (`demo.py`) supporting all three models
- Model architectures (`src/models/`)
- Data loading utilities (`src/data/`)
- Training functions (`src/training/`)
- Requirements and documentation
- Checkpoint directory structure (`checkpoints/`)

#### Still Missing for Full Production Use:
- **Pre-trained model weights** (checkpoints must be provided separately - see Data Access below)
- **Main training scripts** (e.g., `train_model.py`)
- **Complete data preprocessing pipeline**
- **Sample dataset** (users must provide their own data - see Data Access below)

## 🔒 Data Access and Privacy Notice

**Important**: This project involves medical imaging data and clinical information that are subject to privacy regulations and institutional policies.

### Data Access Requirements

The RDPM models and associated datasets are available only for **non-commercial collaborative research projects** under the following conditions:

1. **Formal Agreement Required**: 
   - A signed data use agreement or collaboration agreement must be established
   - Institutional approval may be required
   - Research ethics approval may be necessary

2. **Non-Commercial Use Only**:
   - Academic research institutions
   - Non-profit research organizations
   - Collaborative research projects with clear academic objectives

3. **Contact Process**:
   - Contact the corresponding authors to discuss collaboration
   - Provide details of your research project and intended use
   - Submit formal request through institutional channels
   - Allow time for legal and ethical review processes

### What Can Be Shared:
- **Source code** (models, training utilities, demo scripts)
- **Documentation** and technical specifications
- **Model architectures** and implementation details

### What Requires Formal Agreement:
- **Trained model weights/checkpoints**
- **Medical imaging datasets**
- **Clinical metadata**
- **Patient data** (even if de-identified)

### Contact Information:
Please contact the corresponding authors through official academic channels to discuss potential collaboration and data access.

📋 **For detailed information, see [DATA_ACCESS_POLICY.md](DATA_ACCESS_POLICY.md)**

### To Use This Software on Your Data:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your data structure:**
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

3. **Obtain model checkpoints** and place them in `checkpoints/` directory:
   - **Important**: Model checkpoints contain trained weights and are subject to data access restrictions
   - Contact authors for collaborative research agreements (see Data Access section above)
   - Formal agreements required for non-commercial research use

4. **Run the demo:**
   ```bash
   python demo.py --model hybrid --data_dir your_data --json_file your_data/metadata.json
   ```

### Quick Start Example:
```bash
# See sample data structure
python demo.py --show_sample_data

# Run inference (assumes checkpoints and data are available)
python demo.py --model hybrid --data_dir /path/to/data --json_file /path/to/metadata.json
```

### For Developers:
To complete this implementation, the following scripts need to be created:
- `train_model.py` - Main training script
- `predict.py` - Inference script
- `demo.py` - Demonstration script
- Integration of existing components into user-friendly interfaces

Coming soon.

## Project Structure
```
RDPM/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py          # Data loading utilities
│   │   └── transforms.py       # Image transformation utilities
│   ├── models/
│   │   ├── __init__.py
│   │   ├── image_only.py       # CT-only Model
│   │   ├── image_w_attn.py     # CT with Attention Model
│   │   └── hybrid_model.py     # RDPM (Hybrid Model)
│   └── training/
│       ├── __init__.py
│       └── train.py            # Training utilities
├── checkpoints/                # Model checkpoint directory
│   ├── README.md               # Checkpoint documentation
│   ├── rdpm_hybrid_best.pth    # Hybrid model weights (to be provided)
│   ├── rdpm_image_only_best.pth # Image-only weights (to be provided)
│   └── rdpm_image_w_attn_best.pth # Image+attention weights (to be provided)
├── demo_results/               # Demo output directory (created when running demo)
├── demo.py                     # Complete demo script ✅
├── DEMO_GUIDE.md               # Detailed demo usage guide ✅
├── DATA_ACCESS_POLICY.md       # Data access and privacy policy ✅
├── requirements.txt            # Python dependencies
└── README.md                   # This file

Files for enhanced functionality (could be added):
├── train_model.py              # Standalone training script
├── predict.py                  # Standalone prediction script
└── evaluation/
    └── test_model.py           # Evaluation utilities
```

## License

MIT License

Copyright (c) 2025 <i>Multimodal deep learning model for AI-based decision support in patients with Complex RCC</i> Authors

**Important Note**: While the source code is licensed under MIT, access to trained model weights and datasets requires separate formal agreements due to medical data privacy regulations. See the "Data Access and Privacy Notice" section above.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

