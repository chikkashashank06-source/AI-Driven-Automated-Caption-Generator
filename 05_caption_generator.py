# ============================================================================
# CELL 5: Caption File Generator
# ============================================================================

class CaptionGenerator:
    """Generate SRT, ASS, VTT caption files"""

    @staticmethod
    def format_timestamp_srt(seconds):
        """Convert to SRT format (HH:MM:SS,mmm)"""
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = int(td.total_seconds() % 60)
        millis = int((td.total_seconds() % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def generate_srt(segments, output_path=None):
        """Generate SRT subtitle file"""
        srt_lines = []

        for i, seg in enumerate(segments, 1):
            start = CaptionGenerator.format_timestamp_srt(seg['start'])
            end = CaptionGenerator.format_timestamp_srt(seg['end'])

            srt_lines.append(f"{i}")
            srt_lines.append(f"{start} --> {end}")
            srt_lines.append(seg['text'])
            srt_lines.append("")

        srt_content = "\n".join(srt_lines)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            print(f"✅ SRT saved: {output_path}")

        return srt_content

    @staticmethod
    def format_timestamp_ass(seconds):
        """Convert to ASS format (H:MM:SS.cc)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centisecs = int((seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"

    @staticmethod
    def generate_ass(segments, style='default', output_path=None):
        """Generate ASS subtitle with styling"""

        styles = {
            'default': {
                'font': 'Arial Bold', 'size': 24,
                'color': '&H00FFFFFF', 'outline': '&H00000000'
            },
            'tiktok': {
                'font': 'Arial Bold', 'size': 28,
                'color': '&H00FFFF00', 'outline': '&H00000000'
            },
            'reels': {
                'font': 'Helvetica Bold', 'size': 26,
                'color': '&H00FFFFFF', 'outline': '&H00FF6B00'
            }
        }

        s = styles.get(style, styles['default'])

        ass_content = f"""[Script Info]
Title: CaptionCrafter AI
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, Bold, Outline, Shadow, Alignment
Style: Default,{s['font']},{s['size']},{s['color']},{s['outline']},-1,3,2,2

[Events]
Format: Layer, Start, End, Style, Text
"""

        for seg in segments:
            start = CaptionGenerator.format_timestamp_ass(seg['start'])
            end = CaptionGenerator.format_timestamp_ass(seg['end'])
            text = seg['text'].replace('\\', '\\\\')

            ass_content += f"Dialogue: 0,{start},{end},Default,{text}\n"

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(ass_content)
            print(f"✅ ASS saved: {output_path}")

        return ass_content

    @staticmethod
    def generate_text(text, output_path=None):
        """Generate plain text file with full transcript"""
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"✅ Text saved: {output_path}")

        return text


print("✅ CaptionGenerator class defined")
