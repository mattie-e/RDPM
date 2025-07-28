# RDPM: Rapid Decline Prediction Model

Multimodal deep learning model for AI-based decision support in patients with Complex RCC.

## Overview

RDPM combines CT imaging data with clinical parameters to predict the likelihood of rapid kidney function decline. The project includes RDPM and two additional ablated model architectures, where critical components are systematically removed to evaluate their impact on performance:

1. **CT-only Model**: Baseline model using only CT images for prediction
2. **CT with Attention Model**: Enhanced CT model using mask-guided attention mechanisms
3. **RDPM (Hybrid Model)**: Multimodal model combining CT imaging features with clinical parameters

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

## Requirements

See `requirements.txt` for detailed dependencies.


## Usage

### Data Preparation

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

Coming soon.

## Project Structure
```
RDPM/
├── src/
│   ├── data/
│   │   ├── loaders.py          # Data loading utilities
│   │   └── transforms.py       # Image transformation utilities
│   ├── models/
│   │   ├── image_only.py       # CT-only Model
│   │   ├── image_w_attn.py     # CT with Attention Model
│   │   └── hybrid_model.py     # RDPM (Hybrid Model)
│   ├── training/
│   │   ├── train.py            # Training utilities
│   │   └── train_model.py      # Script for training models
│   └── evaluation/
│       └── test_model.py       # Script for testing/inference with models
├── requirements.txt
└── README.md
```

## License

MIT License

Copyright (c) 2025 <i>Multimodal deep learning model for AI-based decision support in patients with Complex RCC</i> Authors

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

