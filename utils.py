import subprocess
import shutil
import os
import json

# Video2X installation path (update if installed elsewhere)
VIDEO2X_PATH = r"C:\Program Files\Video2X Qt6\video2x.exe"

def get_video_info(video_path):
    """Get video info using ffprobe"""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,duration,codec_name,r_frame_rate,nb_frames',
        '-show_entries', 'format=duration',
        '-of', 'json',
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        info = json.loads(result.stdout)
        
        video_stream = {}
        if 'streams' in info and len(info['streams']) > 0:
            video_stream = info['streams'][0]
        
        # Get duration from format if not in stream
        if 'duration' not in video_stream and 'format' in info:
            video_stream['duration'] = info['format'].get('duration', 0)
        
        return video_stream
    except Exception as e:
        return {"error": str(e)}

def check_ffmpeg_installed():
    """Check if ffmpeg is installed"""
    return shutil.which("ffmpeg") is not None

def check_video2x_installed():
    """Check if Video2X is installed"""
    # First check if it's in PATH
    if shutil.which("video2x") is not None:
        return True
    # Then check the known installation path
    return os.path.exists(VIDEO2X_PATH)

def get_video2x_path():
    """Get the Video2X executable path"""
    # First check if it's in PATH
    path_exe = shutil.which("video2x")
    if path_exe:
        return path_exe
    # Use the known installation path
    if os.path.exists(VIDEO2X_PATH):
        return VIDEO2X_PATH
    return None

def encode_video(video_path, output_path, settings):
    """Encode video with specified settings using ffmpeg CLI"""
    try:
        # Settings extraction
        start_time = settings.get('start_time', 0)
        end_time = settings.get('end_time')
        encoding_mode = settings.get('encoding_mode', 'VBR')
        bitrate = settings.get('bitrate')
        crf = settings.get('crf')
        target_codec = settings.get('target_codec', 'libx264')
        resolution = settings.get('resolution')  # (width, height) or None
        
        # Build ffmpeg command
        cmd = ['ffmpeg', '-y']
        
        # Input with seeking (faster if start_time > 0)
        if start_time > 0:
            cmd.extend(['-ss', str(start_time)])
        
        cmd.extend(['-i', video_path])
        
        # Duration/end time
        if end_time is not None and end_time > start_time:
            duration = end_time - start_time
            cmd.extend(['-t', str(duration)])
        
        # Video codec
        cmd.extend(['-c:v', target_codec])
        
        # Encoding mode settings
        if encoding_mode == 'VBR':
            if bitrate:
                cmd.extend(['-b:v', bitrate])
        elif encoding_mode == 'CBR':
            if bitrate:
                cmd.extend(['-b:v', bitrate])
                cmd.extend(['-minrate', bitrate])
                cmd.extend(['-maxrate', bitrate])
                # Calculate bufsize
                bitrate_value = bitrate.replace('M', '000000').replace('k', '000').replace('K', '000')
                try:
                    bufsize = str(int(float(bitrate_value) * 2))
                    cmd.extend(['-bufsize', bufsize])
                except:
                    pass
        elif encoding_mode == 'CRF':
            if crf is not None:
                cmd.extend(['-crf', str(crf)])
        
        # Resolution (scaling)
        if resolution:
            target_w, target_h = resolution
            # Use scale filter with proper aspect ratio handling
            cmd.extend(['-vf', f'scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2'])
        
        # Audio codec
        cmd.extend(['-c:a', 'aac'])
        
        # Output
        cmd.append(output_path)
        
        # Run ffmpeg
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            return True, "Encoding successful"
        else:
            return False, f"FFmpeg error: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return False, "Encoding timed out (exceeded 1 hour)"
    except Exception as e:
        return False, f"Error: {str(e)}"

def upscale_video_video2x(input_path, output_path, scale_factor=2, processor="realesrgan"):
    """
    Upscale video using Video2X CLI
    
    Args:
        input_path: Path to input video
        output_path: Path for output video
        scale_factor: Upscaling factor (2, 3, or 4)
        processor: Processing engine (realesrgan, realcugan, libplacebo)
    
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        video2x_exe = get_video2x_path()
        if not video2x_exe:
            return False, "Video2X not found. Please install it or update VIDEO2X_PATH in utils.py"
        
        cmd = [
            video2x_exe,
            "-i", input_path,
            "-o", output_path,
            "-p", processor,
            "-d", "0",  # Use first GPU (RTX 5090)
            "-c", "libx264"
        ]
        
        # Add processor-specific options
        if processor == "realesrgan":
            # RealESRGAN uses -s for scale factor
            cmd.extend(["-s", str(scale_factor)])
            # Use appropriate model
            if scale_factor == 4:
                cmd.extend(["--realesrgan-model", "realesrgan-plus"])
            else:
                cmd.extend(["--realesrgan-model", "realesr-animevideov3"])
        elif processor == "realcugan":
            # RealCUGAN uses -s for scale factor
            cmd.extend(["-s", str(scale_factor)])
            cmd.extend(["--realcugan-model", "models-se"])
        elif processor == "libplacebo":
            # libplacebo needs explicit width/height, so we need to calculate
            # Get input video dimensions first
            info = get_video_info(input_path)
            if "error" not in info and "width" in info and "height" in info:
                new_width = int(info["width"]) * scale_factor
                new_height = int(info["height"]) * scale_factor
                cmd.extend(["-w", str(new_width), "-h", str(new_height)])
                cmd.extend(["--libplacebo-shader", "anime4k-v4-a"])
            else:
                # Fallback to scale factor
                cmd.extend(["-s", str(scale_factor)])
        
        # Run Video2X
        cmd_str = ' '.join(f'"{c}"' if ' ' in c else c for c in cmd)
        print(f"Running Video2X command: {cmd_str}")  # Debug logging
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        
        # Check if output file was created successfully, regardless of return code
        # Video2X may segfault during cleanup but still produce valid output
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size > 0:
                # Get output resolution to verify upscaling worked
                output_info = get_video_info(output_path)
                out_w = output_info.get('width', 'unknown')
                out_h = output_info.get('height', 'unknown')
                return True, f"Upscaling complete! Output: {out_w}x{out_h} ({file_size:,} bytes)"
        
        # If we get here, upscaling failed
        if result.returncode != 0:
            error_output = result.stderr or result.stdout or "Unknown error"
            return False, f"Video2X error (code {result.returncode}): {error_output}"
        else:
            return False, "Video2X completed but output file was not created"
            
    except subprocess.TimeoutExpired:
        return False, "Upscaling timed out (exceeded 2 hours)"
    except Exception as e:
        return False, f"Error: {str(e)}"


def generate_srt_from_segments(segments):
    """
    Generate SRT subtitle content from Whisper segments
    
    Args:
        segments: List of segment dicts with 'start', 'end', 'text' keys
    
    Returns:
        str: SRT formatted subtitle content
    """
    def format_timestamp(seconds):
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    srt_lines = []
    for i, seg in enumerate(segments, 1):
        start = seg.get('start', 0)
        end = seg.get('end', 0)
        text = seg.get('text', '').strip()
        
        if text:
            srt_lines.append(str(i))
            srt_lines.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
            srt_lines.append(text)
            srt_lines.append("")  # Empty line between entries
    
    return "\n".join(srt_lines)


def burn_subtitles_to_video(video_path, srt_path, output_path, settings=None):
    """
    Burn subtitles into video using FFmpeg
    
    Args:
        video_path: Path to input video
        srt_path: Path to SRT subtitle file
        output_path: Path for output video
        settings: Optional encoding settings dict
    
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        settings = settings or {}
        target_codec = settings.get('target_codec', 'libx264')
        crf = settings.get('crf', 23)
        
        # Escape special characters in path for FFmpeg filter
        # FFmpeg subtitles filter needs escaped colons and backslashes
        srt_path_escaped = srt_path.replace('\\', '/').replace(':', '\\:')
        
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vf', f"subtitles='{srt_path_escaped}'",
            '-c:v', target_codec,
            '-crf', str(crf),
            '-c:a', 'aac',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            return True, "Subtitles burned successfully"
        else:
            return False, f"FFmpeg error: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return False, "Encoding timed out (exceeded 1 hour)"
    except Exception as e:
        return False, f"Error: {str(e)}"


def add_subtitle_track_to_video(video_path, srt_path, output_path):
    """
    Add subtitles as a separate track (soft subtitles) to video
    
    Args:
        video_path: Path to input video
        srt_path: Path to SRT subtitle file
        output_path: Path for output video (must be .mkv for best compatibility)
    
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', srt_path,
            '-c:v', 'copy',
            '-c:a', 'copy',
            '-c:s', 'srt',
            '-map', '0:v',
            '-map', '0:a?',
            '-map', '1:0',
            '-metadata:s:s:0', 'language=eng',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            return True, "Subtitle track added successfully"
        else:
            return False, f"FFmpeg error: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return False, "Encoding timed out (exceeded 1 hour)"
    except Exception as e:
        return False, f"Error: {str(e)}"


def enhance_video(input_path, output_path, options, codec='libx264'):
    """
    Enhance video quality using FFmpeg filters
    
    Args:
        input_path: Path to input video
        output_path: Path for output video
        options: Dict of enhancement options:
            - denoise: bool - Apply noise reduction
            - denoise_strength: int (1-10) - Noise reduction strength
            - sharpen: bool - Apply sharpening
            - sharpen_strength: float (0.5-2.0) - Sharpening strength
            - color_correct: bool - Auto color correction
            - deblock: bool - Remove compression artifacts
            - stabilize: bool - Video stabilization (2-pass)
        codec: Output codec (libx264 or libx265)
    
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        filters = []
        applied_filters = []
        
        # Denoise using hqdn3d (high quality 3D denoise)
        if options.get('denoise'):
            strength = options.get('denoise_strength', 5)
            # hqdn3d parameters: luma_spatial:chroma_spatial:luma_tmp:chroma_tmp
            # Scale 1-10 to appropriate values (higher = more denoising)
            luma_s = 1.5 + (strength * 0.5)  # 2.0 to 6.5
            chroma_s = luma_s * 0.8
            luma_t = luma_s * 2
            chroma_t = chroma_s * 2
            filters.append(f"hqdn3d={luma_s:.1f}:{chroma_s:.1f}:{luma_t:.1f}:{chroma_t:.1f}")
            applied_filters.append("Denoise")
        
        # Deblock to remove compression artifacts
        if options.get('deblock'):
            # deblock filter with moderate settings
            filters.append("deblock=filter=strong:block=4")
            applied_filters.append("Deblock")
        
        # Color correction using curves and eq
        if options.get('color_correct'):
            # Auto levels/normalize + slight saturation boost + contrast
            filters.append("normalize=blackpt=black:whitept=white:smoothing=50")
            filters.append("eq=contrast=1.05:saturation=1.1:brightness=0.02")
            applied_filters.append("Color Correction")
        
        # Sharpen using unsharp mask
        if options.get('sharpen'):
            strength = options.get('sharpen_strength', 1.0)
            # unsharp parameters: lx:ly:la:cx:cy:ca
            # lx, ly = luma matrix size (must be odd, 3-23)
            # la = luma amount (negative = blur, positive = sharpen)
            amount = 0.5 + (strength * 0.75)  # 1.0 to 2.0 typical
            filters.append(f"unsharp=5:5:{amount:.2f}:5:5:{amount * 0.5:.2f}")
            applied_filters.append("Sharpen")
        
        # Video stabilization (requires 2-pass)
        if options.get('stabilize'):
            # Stabilization requires a 2-pass process
            # First pass: analyze motion and create transform file
            # Use a simple filename in the same directory to avoid path escaping issues
            import tempfile
            transforms_dir = os.path.dirname(input_path)
            transforms_file = os.path.join(transforms_dir, "stab_transforms.trf")
            
            # Convert to forward slashes and escape colons for FFmpeg filter
            transforms_file_escaped = transforms_file.replace('\\', '/').replace(':', '\\:')
            
            # Pass 1: Detect motion
            detect_cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-vf', f"vidstabdetect=shakiness=5:accuracy=15:result={transforms_file_escaped}",
                '-f', 'null',
                'NUL' if os.name == 'nt' else '/dev/null'
            ]
            
            result = subprocess.run(detect_cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode != 0:
                return False, f"Stabilization analysis failed: {result.stderr[-500:] if result.stderr else 'Unknown error'}"
            
            # Verify transform file was created
            if not os.path.exists(transforms_file):
                return False, "Stabilization analysis did not produce transform file"
            
            # Add stabilization transform to filters
            filters.append(f"vidstabtransform=input={transforms_file_escaped}:smoothing=10:crop=black:zoom=1")
            applied_filters.append("Stabilize")
        
        if not filters:
            # No filters selected, just copy
            return False, "No enhancement filters selected"
        
        # Build filter chain
        filter_chain = ','.join(filters)
        
        # Build FFmpeg command
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-vf', filter_chain,
            '-c:v', codec,
            '-crf', '18',  # High quality output
            '-preset', 'medium',
            '-c:a', 'aac',
            '-b:a', '192k',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        
        if result.returncode == 0:
            # Verify output
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return True, f"Applied: {', '.join(applied_filters)}"
            else:
                return False, "Enhancement completed but output file is empty"
        else:
            return False, f"FFmpeg error: {result.stderr[:500]}"
            
    except subprocess.TimeoutExpired:
        return False, "Enhancement timed out (exceeded 2 hours)"
    except Exception as e:
        return False, f"Error: {str(e)}"
