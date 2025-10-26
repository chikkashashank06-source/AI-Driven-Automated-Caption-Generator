# AI-Driven-Automated-Caption-Generator
An AI tool that automatically transcribes video speech and generates clean, engaging captions using OpenAI Whisper, Transformers, and MoviePy. Run it in Google Colab with free GPU to create captioned videos instantly for creators and educators.
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
        return "Please upload a valid video.", "", "", "", None

    print(f"📁 Processing file: {video_path}")
    target_langs = [lang.strip() for lang in target_langs_str.split(',') if lang.strip()]

    # Transcribe + translate
    results = process_multilingual_video(
        video_path,
        source_language=None,
        target_languages=target_langs
    )

    source_lang = results['source_language']
    captions_preview = f"### Detected Source Language: **{source_lang.upper()}**\n\n"

    for lang in target_langs:
        if lang in results['captions']:
            captions_preview += f"#### {lang.upper()} Captions\n"
            for seg in results['captions'][lang][:5]:
                captions_preview += f"- [{seg['start']:.1f}-{seg['end']:.1f}s] {seg['text']}\n"
            captions_preview += "\n"

    # Create caption SRT files and burned video
    srt_files = []
    video_with_captions = None
    first_lang = target_langs[0] if target_langs else source_lang

    if first_lang in results['captions']:
        CaptionGenerator.generate_srt(results['captions'][first_lang], f"captions_{first_lang}.srt")
        srt_files.append(f"captions_{first_lang}.srt")

        # Burn captions into video (preview)
        video_with_captions = burn_captions(video_path, results['captions'][first_lang], first_lang)


    # Update outputs to match the new function signature and ensure correct data is returned
    return (
        captions_preview,
        results['text'],
        srt_files[0] if srt_files else None,
        source_lang,
        video_with_captions
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
        gr.File(label="⬇️ Download SRT File"),
        gr.Textbox(label="Detected Source Language"),
        gr.Video(label="🎬 Captioned Video Preview")  # new output component
    ],
    title="🎬 CaptionCrafter AI – Multilingual Caption Generator",
    description="""
Upload a short video, and this tool automatically transcribes, translates,
and generates captions in multiple languages. It also burns captions
into the video for live preview.

**Supports:** English (en), Hindi (hi), Spanish (es), French (fr), German (de).
""",
    examples=[
        [None, "en,hi,es"],
        [None, "en,fr"]
    ]
)

print("\n🚀 Launching Advanced Gradio Interface with Video Preview...")
print("=" * 70)
demo.launch(share=True)