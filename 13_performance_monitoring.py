# ============================================================================
# CELL 13: PERFORMANCE MONITORING
# ============================================================================

import time

def benchmark_processing(video_path, target_languages=['en', 'hi', 'es']):
    """Measure processing performance"""

    timings = {}

    # Transcription
    start = time.time()
    result = transcriber.transcribe(video_path)
    timings['transcription'] = time.time() - start

    # Translation
    translation_times = {}
    for target_lang in target_languages:
        if target_lang != result['language']:
            start = time.time()
            translator.translate_segments(
                result['segments'],
                result['language'],
                target_lang
            )
            translation_times[target_lang] = time.time() - start

    timings['translation'] = translation_times

    # Print report
    print("\n⏱️  PERFORMANCE REPORT")
    print("=" * 70)
    print(f"Transcription: {timings['transcription']:.2f}s")
    for lang, t in translation_times.items():
        print(f"Translation to {lang}: {t:.2f}s")
    print(f"Total: {timings['transcription'] + sum(translation_times.values()):.2f}s")
    print("=" * 70)

    return timings

# Run benchmark
if video_file:
    benchmark_processing(video_file, ['en', 'hi', 'es'])
