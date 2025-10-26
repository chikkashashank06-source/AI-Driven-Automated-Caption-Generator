# ============================================================================
# CELL 4: Multilingual Translator
# ============================================================================

class MultilingualTranslator:
    """Translate captions between languages"""

    TRANSLATION_MODELS = {
        ('en', 'hi'): 'Helsinki-NLP/opus-mt-en-hi',
        ('en', 'es'): 'Helsinki-NLP/opus-mt-en-es',
        ('en', 'fr'): 'Helsinki-NLP/opus-mt-en-fr',
        ('en', 'de'): 'Helsinki-NLP/opus-mt-en-de',
        ('hi', 'en'): 'Helsinki-NLP/opus-mt-hi-en',
        ('es', 'en'): 'Helsinki-NLP/opus-mt-es-en',
        ('fr', 'en'): 'Helsinki-NLP/opus-mt-fr-en',
        ('de', 'en'): 'Helsinki-NLP/opus-mt-de-en',
        ('ja', 'en'): 'Helsinki-NLP/opus-mt-ja-en',
        ('en', 'ja'): 'Helsinki-NLP/opus-mt-en-ja',
        ('ko', 'en'): 'Helsinki-NLP/opus-mt-ko-en',
        ('en', 'ko'): 'Helsinki-NLP/opus-mt-en-ko',
    }

    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self, source_lang, target_lang):
        """Load translation model for language pair"""
        model_key = (source_lang, target_lang)

        if model_key in self.models:
            return

        if model_key not in self.TRANSLATION_MODELS:
            raise ValueError(f"Translation {source_lang}→{target_lang} not supported")

        model_name = self.TRANSLATION_MODELS[model_key]
        print(f"📥 Loading: {source_lang} → {target_lang}")

        self.tokenizers[model_key] = MarianTokenizer.from_pretrained(model_name)
        self.models[model_key] = MarianMTModel.from_pretrained(model_name).to(self.device)

        print(f"✅ Model ready")

    def translate_text(self, text, source_lang, target_lang):
        """Translate single text"""
        if source_lang == target_lang:
            return text

        model_key = (source_lang, target_lang)

        if model_key not in self.models:
            self.load_model(source_lang, target_lang)

        tokenizer = self.tokenizers[model_key]
        model = self.models[model_key]

        inputs = tokenizer(text, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            translated = model.generate(**inputs)

        return tokenizer.decode(translated[0], skip_special_tokens=True)

    def translate_segments(self, segments, source_lang, target_lang):
        """Translate all caption segments"""
        print(f"🌐 Translating {len(segments)} segments: {source_lang} → {target_lang}")

        translated = []
        for seg in segments:
            translated_text = self.translate_text(seg['text'], source_lang, target_lang)

            translated_seg = seg.copy()
            translated_seg['text'] = translated_text
            translated_seg['original_text'] = seg['text']

            translated.append(translated_seg)

        print(f"✅ Translation complete")
        return translated

print("✅ MultilingualTranslator class defined")
