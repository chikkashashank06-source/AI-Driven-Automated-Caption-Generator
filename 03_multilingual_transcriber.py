# ============================================================================
# CELL 3: Multilingual Transcriber
# ============================================================================

class MultilingualTranscriber:
    """Transcribe audio/video in multiple languages using Whisper"""

    SUPPORTED_LANGUAGES = {
        'en': 'English', 'hi': 'Hindi', 'es': 'Spanish',
        'fr': 'French', 'de': 'German', 'ja': 'Japanese',
        'ko': 'Korean', 'zh': 'Chinese', 'ar': 'Arabic',
        'ru': 'Russian', 'pt': 'Portuguese', 'it': 'Italian'
    }

    def __init__(self, model_name='base'):
        """
        Initialize Whisper model
        Options: tiny, base, small, medium, large
        Recommended for Colab: base or medium
        """
        print(f"🔄 Loading Whisper '{model_name}' model...")
        self.model = whisper.load_model(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        print(f"✅ Model loaded on {self.device}")

    def transcribe(self, audio_path, language=None, word_timestamps=True):
        """Transcribe audio with optional language specification"""
        print(f"🎧 Transcribing: {os.path.basename(audio_path)}")

        result = self.model.transcribe(
            audio_path,
            language=language,
            word_timestamps=word_timestamps,
            verbose=False
        )

        # Format segments
        segments = []
        for seg in result['segments']:
            segment_data = {
                'start': float(seg['start']),
                'end': float(seg['end']),
                'text': seg['text'].strip(),
                'words': []
            }

            if 'words' in seg and seg['words']:
                for word in seg['words']:
                    segment_data['words'].append({
                        'word': word['word'].strip(),
                        'start': float(word['start']),
                        'end': float(word['end'])
                    })

            segments.append(segment_data)

        print(f"✅ Transcribed {len(segments)} segments in {result['language']}")

        return {
            'language': result['language'],
            'text': result['text'].strip(),
            'segments': segments
        }

print("✅ MultilingualTranscriber class defined")
