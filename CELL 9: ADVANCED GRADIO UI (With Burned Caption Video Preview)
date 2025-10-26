# ============================================================================
# CELL 9: ADVANCED GRADIO UI (With Burned Caption Video Preview)
# ============================================================================

import os, ffmpeg, gradio as gr

def burn_captions(video_path, segments, lang='en', style='default'):
    """Burn captions into a copy of the video using FFmpeg"""
    if not video_path or not os.path.exists(video_path):
        return None

    # Create subtitle file
    srt_file = f"captions_{lang}.srt"
    CaptionGenerator.generate_srt(segments, srt_file)

    output_path = f"captioned_{lang}_{os.path.basename(video_path)}"

    # Run FFmpeg subtitle burn-in process
    try:
        (
            ffmpeg
            .input(video_path)
            .output(output_path, vf=f"subtitles={srt_file}", vcodec='libx264', acodec='aac', preset='ultrafast')
            .overwrite_output()
            .run(quiet=True)
        )
        print(f"✅ Burned subtitles into video: {output_path}")
        return output_path
    except ffmpeg.Error as e:
        print("❌ FFmpeg Error:", e)
        return None


def gradio_process(video_path, target_langs_str):
    """Process uploaded video via Gradio interface"""

    if not video_path or not os.path.exists(video_path):
        return "Please upload a valid video.", "", None, None, None, None

    print(f"📁 Processing file: {video_path}")
    target_langs = [lang.strip() for lang in target_langs_str.split(',') if lang.strip()]

    # Initialize transcriber and translator within the Gradio process function
    # This ensures they are correctly initialized for each Gradio call
    # You can change 'base' to 'medium' or 'large' for potentially higher accuracy (but slower)
    transcriber = MultilingualTranscriber(model_name='base')
    translator = MultilingualTranslator()


    # Transcribe + translate
    results = process_multilingual_video(
        video_path,
        source_language=None,
        target_languages=target_langs
    )

    source_lang = results['source_language']
    captions_preview = f"### Detected Source Language: **{source_lang.upper()}**\n\n"

    # Generate and collect SRT files for all languages
    srt_files_to_download = {}
    for lang, segments in results['captions'].items():
        srt_file = f"captions_{lang}.srt"
        CaptionGenerator.generate_srt(segments, srt_file)
        srt_files_to_download[f"Download {lang.upper()} SRT"] = srt_file # Use dictionary for multiple files

    # Generate full transcript text file
    text_file = "full_transcript.txt"
    CaptionGenerator.generate_text(results['text'], text_file)

    # Burn captions into video (preview) - use the first target language or source if no targets
    video_with_captions = None
    preview_lang = target_langs[0] if target_langs else source_lang
    if preview_lang in results['captions']:
         video_with_captions = burn_captions(video_path, results['captions'][preview_lang], preview_lang)


    # Return outputs
    # The order of returned values must match the order of output components in gr.Interface
    return (
        captions_preview,
        results['text'],
        text_file,
        video_with_captions,
        # Return SRT files as a list of file paths
        list(srt_files_to_download.values())
    )


# BUILD INTERFACE
demo = gr.Interface(
    fn=gradio_process,
    inputs=[
        gr.Video(label="🎥 Upload Video"),
        gr.Textbox(label="🌐 Target Languages (comma-separated)", value="en,hi,es")
    ],
    outputs=[
        gr.Markdown(label="📝 Captions Preview"),
        gr.Textbox(label="📄 Full Transcript"),
        gr.File(label="⬇️ Download Full Transcript"), # New output component for transcript
        gr.Video(label="🎬 Captioned Video Preview"),
        gr.File(label="⬇️ Download All SRT Files", file_count="multiple") # Modified to allow multiple files
    ],
    title="🎬 CaptionCrafter AI – Multilingual Caption Generator",
    description="""
Upload a short video, and this tool automatically transcribes, translates,
and generates captions in multiple languages. It also burns captions
into the video for live preview.

**Supported Source Languages (auto-detected):** English, Hindi, Spanish, French, German, Japanese, Korean, Chinese, Arabic, Russian, Portuguese, Italian.
**Supported Target Languages (for translation):** English (en), Hindi (hi), Spanish (es), French (fr), German (de), Japanese (ja), Korean (ko).

**Note on Accuracy and Speed:** You can adjust the Whisper model size in the code for better accuracy ('medium' or 'large') at the cost of processing time.
""",
    examples=[
        [None, "en,hi,es"],
        [None, "en,fr"]
    ]
)

print("\n🚀 Launching Advanced Gradio Interface with Video Preview...")
print("=" * 70)
demo.launch(share=True)
