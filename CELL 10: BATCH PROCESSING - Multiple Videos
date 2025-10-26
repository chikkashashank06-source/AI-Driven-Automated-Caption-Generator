# ============================================================================
# CELL 10: BATCH PROCESSING - Multiple Videos
# ============================================================================

def batch_process_videos(video_paths, target_languages=['en', 'hi', 'es']):
    """Process multiple videos at once"""

    all_results = []

    for i, video_path in enumerate(video_paths, 1):
        print(f"\n{'='*70}")
        print(f"Processing Video {i}/{len(video_paths)}: {video_path}")
        print(f"{'='*70}")

        result = process_multilingual_video(
            video_path,
            target_languages=target_languages
        )

        all_results.append({
            'filename': video_path,
            'results': result
        })

    return all_results

# Example: Upload multiple videos
print("📤 Upload multiple videos for batch processing...")
uploaded_files = files.upload()

if uploaded_files:
    video_list = list(uploaded_files.keys())

    batch_results = batch_process_videos(
        video_list,
        target_languages=['en', 'hi', 'es']
    )

    print("\n✅ Batch processing complete!")
    print(f"Processed {len(batch_results)} videos")
