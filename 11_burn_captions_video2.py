# ============================================================================
# CELL 11: BURN CAPTIONS INTO VIDEO (using FFmpeg)
# ============================================================================

def burn_captions_to_video(video_path, segments, language, output_path=None):
    """
    Burn captions directly into video using FFmpeg

    Args:
        video_path: Input video path
        segments: Caption segments
        language: Language code
        output_path: Output video path
    """

    if output_path is None:
        output_path = f"captioned_{language}_{os.path.basename(video_path)}"

    # Generate ASS subtitle file
    ass_file = f"subtitles_{language}.ass"
    CaptionGenerator.generate_ass(segments, 'tiktok', ass_file)

    print(f"🎬 Burning {language} captions into video...")

    # Use FFmpeg to burn subtitles
    input_video = ffmpeg.input(video_path)

    video = input_video.video.filter('ass', ass_file)
    audio = input_video.audio

    output = ffmpeg.output(
        video,
        audio,
        output_path,
        vcodec='libx264',
        acodec='aac',
        preset='ultrafast'
    )

    ffmpeg.run(output, overwrite_output=True, quiet=True)

    print(f"✅ Captioned video saved: {output_path}")

    return output_path

# Example usage
if video_file and results:
    # Burn English captions
    captioned_video = burn_captions_to_video(
        video_file,
        results['captions']['en'],
        'en'
    )

    # Display result
    display(Video(captioned_video, embed=True))

    # Download
    files.download(captioned_video)
