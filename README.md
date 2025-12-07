# Perceptual Influence: Improving the Perceptual Loss Design for Low-Dose CT Enhancement

[![arXiv](https://img.shields.io/badge/arXiv-2509.23025-b31b1b.svg)](https://arxiv.org/abs/2509.23025)

## Key Contributions

This work introduces the **perceptual influence metric**, a novel approach to quantify the relative contribution of perceptual loss components in neural network training. By providing objective guidelines for perceptual loss design, this metric enables researchers and practitioners to optimize loss configurations systematically, leading to significant improvements in noise reduction and structural fidelity for Low-Dose CT image enhancement. The effectiveness of this approach is validated through comprehensive statistical analysis, demonstrating that better-designed perceptual losses outperform commonly used configurations without requiring architectural changes.

## Installation

### Requirements

- Python 3.9.16
- TensorFlow 2.10.0
- See `requirements.txt` for complete dependencies

### Setup

1. Clone the repository:
```bash
git clone https://github.com/vngabriel/perceptual-influence.git
cd perceptual-influence
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
perceptual-influence/
├── src/
│   ├── data/                    # Data loading and preprocessing
│   │   ├── mayo_challenge.py    # Mayo Challenge dataset handler
│   │   └── mayo_challenge_preprocessing.py
│   ├── losses/                  # Loss function implementations
│   │   └── perceptual_loss.py  # Perceptual loss implementation
│   ├── models/                  # Neural network architectures
│   │   └── unet.py             # U-Net implementation
│   ├── method/                  # Analysis methods
│   │   ├── perceptual_influence.py  # Perceptual influence analysis
│   │   └── hypothesis_test.py      # Statistical hypothesis testing
│   ├── utils/                   # Utility functions
│   │   ├── directory_tools.py
│   │   └── system_monitor.py
│   └── main.py                 # Main experiment runner
├── train_experiment.ini        # Training configuration
├── test_experiment.ini         # Testing configuration
└── requirements.txt           # Dependencies
```

## Usage

### Training

1. Configure your experiment in `train_experiment.ini`:
```ini
[Experiment-Management]
mode = train
resume-previous-exp =  no
previous-hash = -
last-epoch-executed = -
data-dir = /path/to/mayo-challenge

[Train-Setup]
database = mayo-challenge
negative_values = no
split-hash = 4840
network = unet
patch-size = 96
patch-skip = 96
loss = perceptual
batch = 8
epochs = 10
lr = 1e-4
seed = 5
overwrite-weights-file = yes

[Network-Hyperparameters]
channels = 32

[Perceptual-Loss-Hyperparameters]
image-space-loss = mse
perceptual-space-loss = mse
perceptual-content-layer = block3_conv2
perceptual-weight = 1e-5
perceptual-model = vgg19-imagenet
weights-path = none
```

2. Run the training:

In`src/main.py`, update the following line:

```python
experiment = Experiment("train_experiment.ini")
```

Then, start the training by running:

```bash
python src/main.py
```

### Testing

1. Configure testing in `test_experiment.ini`:
```ini
[Experiment-Management]
mode = test
previous-hash = 12345
load-specific-epoch = -
data-dir = /path/to/mayo-challenge
output-dir-imgs = /path/to/output

[Test-Setup]
patch-size = 512
train-loss-curve = no
val-loss-curve = no
train-acc-curve = no
val-acc-curve = no
save-curves-as-csv = no
visualize-individual-img = no
compute-metrics-all-set = test
```

2. Run testing:

In`src/main.py`, update the following line:

```python
experiment = Experiment("test_experiment.ini")
```

Then, start the testing by running:

```bash
python src/main.py
```

## Perceptual Loss Configuration

The perceptual loss implementation supports several configurations:

### Perceptual Models
- `vgg19-imagenet`: VGG-19 pretrained on ImageNet
- `vgg19-custom`: VGG-19 with custom weights
- `autoencoder`: Custom autoencoder encoder

### Content Layers
- `block3_conv2`: Mid-level features (recommended)
- `block5_conv4`: High-level features
- Any other layer of VGG-19 can also be used depending on the desired feature representation

### Loss Functions
- **Image Space**: MSE, MAE
- **Perceptual Space**: MSE, MAE

### Perceptual Weight
The perceptual weight $\lambda$ controls the influence of the perceptual component. The paper provides guidelines for optimal weight selection based on perceptual influence analysis.

## Results

The paper demonstrates that:
- Better perceptual loss designs significantly improve noise reduction
- Structural fidelity is enhanced without architectural changes
- Perceptual influence analysis provides objective guidelines for loss design

