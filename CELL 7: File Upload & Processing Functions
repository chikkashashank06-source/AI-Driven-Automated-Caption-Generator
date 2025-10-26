# ============================================================================
# CELL 7: File Upload & Processing Functions
# ============================================================================

def upload_video():
    """Upload video file to Colab"""
    print("📤 Click 'Choose Files' to upload your video...")
    uploaded = files.upload()

    if uploaded:
        filename = list(uploaded.keys())[0]
        print(f"✅ Uploaded: {filename}")
        return filename
    return None

def process_multilingual_video(
    video_path,
    source_language=None,
    target_languages=['en', 'hi', 'es']
):
    """
    Complete multilingual caption generation pipeline

    Args:
        video_path: Path to video file
        source_language: Source language (None = auto-detect)
        target_languages: List of target language codes

    Returns:
        Dictionary with captions in all languages
    """

    print("\n" + "=" * 70)
    print("🎬 MULTILINGUAL CAPTION GENERATION PIPELINE")
    print("=" * 70)

    # Step 1: Transcribe
    print("\n📝 Step 1: Transcribing video...")
    if source_language:
        result = transcriber.transcribe(video_path, language=source_language)
    else:
        result = transcriber.transcribe(video_path)

    source_lang = result['language']
    source_segments = result['segments']

    print(f"✅ Source language: {source_lang}")
    print(f"✅ Detected {len(source_segments)} segments")
    print(f"✅ Full text: {result['text'][:200]}...")

    # Step 2: Translate to target languages
    all_captions = {source_lang: source_segments}

    for target_lang in target_languages:
        if target_lang != source_lang:
            model_key = (source_lang, target_lang)
            if model_key in translator.TRANSLATION_MODELS:
                print(f"\n🌐 Step 2: Translating to {target_lang}...")
                translated = translator.translate_segments(
                    source_segments,
                    source_lang,
                    target_lang
                )
                all_captions[target_lang] = translated
            else:
                print(f"\n⚠️  Translation from {source_lang} to {target_lang} is not supported. Skipping translation.")


    # Step 3: Generate caption files
    print("\n📄 Step 3: Generating caption files...")
    caption_files = {}

    for lang, segments in all_captions.items():
        srt_file = f"captions_{lang}.srt"
        CaptionGenerator.generate_srt(segments, srt_file)
        caption_files[lang] = srt_file

    print("\n" + "=" * 70)
    print("✅ PROCESSING COMPLETE!")
    print("=" * 70)

    return {
        'source_language': source_lang,
        'captions': all_captions,
        'caption_files': caption_files,
        'text': result['text']
    }

