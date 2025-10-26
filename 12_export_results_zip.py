# ============================================================================
# CELL 12: EXPORT ALL RESULTS TO ZIP
# ============================================================================

import zipfile

def create_export_package(results, video_filename):
    """Create a ZIP file with all outputs"""

    zip_filename = f"captions_{os.path.splitext(video_filename)[0]}.zip"

    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        # Add caption files
        for lang, filepath in results['caption_files'].items():
            if os.path.exists(filepath):
                zipf.write(filepath)

        # Add JSON with full results
        json_file = 'results.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'source_language': results['source_language'],
                'text': results['text'],
                'captions': {
                    lang: [{
                        'start': seg['start'],
                        'end': seg['end'],
                        'text': seg['text']
                    } for seg in segments]
                    for lang, segments in results['captions'].items()
                }
            }, f, indent=2, ensure_ascii=False)

        zipf.write(json_file)

    print(f"✅ Export package created: {zip_filename}")
    return zip_filename

# Create and download export package
if video_file and results:
    export_zip = create_export_package(results, video_file)
    files.download(export_zip)
