# ============================================================================
# CELL 1: Installation & Setup
# ============================================================================

print("🚀 Installing CaptionCrafter AI Dependencies...")
print("=" * 70)

# Install core dependencies
!pip install -q openai-whisper
!pip install -q transformers sentencepiece
!pip install -q torch torchvision torchaudio
!pip install -q moviepy
!pip install -q ffmpeg-python
!pip install -q gradio  # For web UI

# Install system dependencies
!apt-get -qq install -y ffmpeg

print("✅ All dependencies installed!")
print("=" * 70)

# Check GPU availability
import torch
if torch.cuda.is_available():
    print(f"🎮 GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️  No GPU available, using CPU (will be slower)")
print("=" * 70)
