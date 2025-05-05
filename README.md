# Kyrgyz TTS Model Training

This project implements a Text-to-Speech (TTS) model for the Kyrgyz language using the SpeechT5 architecture.

## Project Structure

```
.
├── train_kokoro_kyrgyz.py    # Main training script
├── test_kokoro_kyrgyz.py     # Testing script
├── requirements.txt          # Python dependencies
├── environment.yml          # Conda environment file
└── kyrgyz_dataset/          # Dataset directory
    ├── audio_files/         # Audio files (not included in git)
    └── metadata.json        # Dataset metadata
```

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- At least 16GB RAM
- At least 50GB free disk space

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/kyrgyz-tts.git
cd kyrgyz-tts
```

2. Create and activate a conda environment:
```bash
conda env create -f environment.yml
conda activate kyrgyz-tts
```

Or install using pip:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

1. Place your audio files in `kyrgyz_dataset/audio_files/`
2. Ensure your `metadata.json` is properly formatted with the following structure:
```json
[
    {
        "audio_filename": "audio1.wav",
        "transcription": "текст на кыргызском"
    }
]
```

## Training

To train the model:

```bash
python train_kokoro_kyrgyz.py
```

The training script will:
- Load and preprocess the dataset
- Initialize the SpeechT5 model
- Train the model with the specified parameters
- Save checkpoints and the final model

## Testing

To test the trained model:

```bash
python test_kokoro_kyrgyz.py
```

## Model Output

The trained model will be saved in the `kyrgyz-tts-model/` directory with the following structure:
- Model weights
- Tokenizer
- Vocoder
- Speaker embeddings

## Notes

- Large audio files are not included in the git repository
- Training logs are saved in `kokoro_training.log`
- Model checkpoints are saved in `kyrgyz-tts-model/`
- Wandb integration is available for training monitoring
