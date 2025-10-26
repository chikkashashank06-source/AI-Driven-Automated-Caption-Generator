# ============================================================================
# CELL 8: QUICK TEST - Upload and Process
# ============================================================================

# Initialize the transcriber and translator
transcriber = MultilingualTranscriber()
translator = MultilingualTranslator()

# Upload your video
video_file = upload_video()

if video_file:
    # Process video with multilingual captions
    results = process_multilingual_video(
        video_file,
        source_language=None,  # Auto-detect
        target_languages=['en', 'hi', 'es']  # English, Hindi, Spanish
    )

    # Display results
    print("\n📊 RESULTS:")
    print("=" * 70)
    print(f"Source Language: {results['source_language']}")
    print(f"\nFull Text:\n{results['text']}\n")

    # Show captions in each language
    for lang, segments in results['captions'].items():
        print(f"\n{lang.upper()} CAPTIONS:")
        print("-" * 70)
        for seg in segments[:3]:  # Show first 3 segments
            print(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")

    # Download caption files
    print("\n📥 Downloading caption files...")
    for lang, filepath in results['caption_files'].items():
        files.download(filepath)
        print(f"✅ Downloaded: {filepath}")
