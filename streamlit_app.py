import streamlit as st
import subprocess
import tempfile
import os
import sys
import re
import json
import time
from utils import (
    get_video_info,
    encode_video,
    upscale_video_video2x,
    check_ffmpeg_installed,
    check_video2x_installed,
    get_video2x_path,
    generate_srt_from_segments,
    burn_subtitles_to_video,
    add_subtitle_track_to_video,
    enhance_video
)

st.set_page_config(page_title="Video Describe & Editor", layout="wide")

# Initialize session state for preserving data across reruns
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'tmp_dir' not in st.session_state:
    st.session_state.tmp_dir = None
if 'srt_content' not in st.session_state:
    st.session_state.srt_content = None

st.title("Video Describe & Editor")
st.write("Upload a video to analyze it or process/encode it with various options.")

# Sidebar for Encoding Configuration
with st.sidebar:
    st.header("Encoding Settings")
    
    encoding_mode = st.selectbox("Encoding Mode", ["VBR", "CBR", "CRF"])
    
    target_codec = st.selectbox("Target Codec", ["libx264", "libx265"])
    
    if encoding_mode == "VBR":
        bitrate = st.text_input("Target Bitrate (e.g., 5M)", "5M")
        crf = None
    elif encoding_mode == "CBR":
        bitrate = st.text_input("Bitrate (e.g., 5M)", "5M")
        crf = None
    else:  # CRF
        crf = st.slider("CRF Value (Lower is better quality)", 0, 51, 23)
        bitrate = None

    st.header("Resolution")
    # Resolution options will be populated dynamically based on uploaded video
    # Store these in session state so they persist
    if 'original_width' not in st.session_state:
        st.session_state.original_width = None
        st.session_state.original_height = None
    
    # Build resolution options dynamically
    resolution_options = []
    resolution_map = {}  # Maps display string to (width, height, is_upscale, scale_factor)
    
    # Standard resolutions with their heights (for comparison)
    STANDARD_RESOLUTIONS = {
        "SD (480p)": (854, 480),
        "HD (720p)": (1280, 720),
        "Full HD (1080p)": (1920, 1080),
        "2K (1440p)": (2560, 1440),
        "4K (2160p)": (3840, 2160),
    }
    
    if st.session_state.original_width and st.session_state.original_height:
        orig_w = st.session_state.original_width
        orig_h = st.session_state.original_height
        
        # Original option first
        original_label = f"Original ({orig_w}x{orig_h})"
        resolution_options.append(original_label)
        resolution_map[original_label] = (None, None, False, 1)  # None means keep original
        
        # Add standard resolutions BELOW the original (downscaling only)
        for label, (w, h) in STANDARD_RESOLUTIONS.items():
            if h < orig_h:  # Only show if it's smaller than original
                resolution_options.append(label)
                resolution_map[label] = (w, h, False, 1)
        
        # Add upscale options
        upscale_2x_label = f"2x AI Upscale ({orig_w * 2}x{orig_h * 2})"
        upscale_4x_label = f"4x AI Upscale ({orig_w * 4}x{orig_h * 4})"
        resolution_options.append(upscale_2x_label)
        resolution_options.append(upscale_4x_label)
        resolution_map[upscale_2x_label] = (orig_w * 2, orig_h * 2, True, 2)
        resolution_map[upscale_4x_label] = (orig_w * 4, orig_h * 4, True, 4)
    else:
        # No video uploaded yet, show placeholder
        resolution_options = ["Upload a video to see options"]
        resolution_map["Upload a video to see options"] = (None, None, False, 1)
    
    resolution_option = st.selectbox("Output Resolution", resolution_options)
    
    # Get the selected resolution info
    selected_res = resolution_map.get(resolution_option, (None, None, False, 1))
    target_width, target_height, is_upscale, upscale_factor = selected_res
    
    # Auto-enable upscaling for AI upscale options
    if is_upscale:
        upscale_enabled = True
        st.header("AI Upscaling (Video2X)")
        st.info(f"AI Upscaling enabled: {upscale_factor}x")
        upscale_processor = st.selectbox("Processor", ["realesrgan", "realcugan", "libplacebo"], index=0)
        st.caption("Using GPU 0 (NVIDIA RTX) for upscaling")
    else:
        upscale_enabled = False
        upscale_factor = 2
        upscale_processor = "realesrgan"
    
    st.header("🎨 AI Enhance")
    enhance_enabled = st.checkbox("Enable Video Enhancement", value=False,
        help="Apply intelligent filters to improve video quality")
    
    enhance_options = {}
    if enhance_enabled:
        st.caption("Select enhancement filters to apply:")
        
        enhance_options['denoise'] = st.checkbox("🔇 Denoise", value=True,
            help="Remove grain and noise from the video")
        if enhance_options['denoise']:
            enhance_options['denoise_strength'] = st.slider("Denoise Strength", 1, 10, 5, 
                help="Higher = more noise removal (may lose detail)")
        
        enhance_options['sharpen'] = st.checkbox("🔍 Sharpen", value=True,
            help="Enhance edges and details")
        if enhance_options['sharpen']:
            enhance_options['sharpen_strength'] = st.slider("Sharpen Strength", 0.5, 2.0, 1.0, 0.1,
                help="Higher = sharper image (may introduce halos)")
        
        enhance_options['color_correct'] = st.checkbox("🎨 Auto Color Correction", value=False,
            help="Automatically adjust brightness, contrast, and saturation")
        
        enhance_options['deblock'] = st.checkbox("🧱 Deblock", value=False,
            help="Remove compression block artifacts (for heavily compressed videos)")
        
        enhance_options['stabilize'] = st.checkbox("📐 Stabilize", value=False,
            help="Reduce camera shake (requires 2-pass processing, slower)")
    
    st.divider()
    st.header("🔧 System Status")
    
    # FFmpeg status
    ffmpeg_available = check_ffmpeg_installed()
    if ffmpeg_available:
        st.success("✅ FFmpeg: Installed")
    else:
        st.error("❌ FFmpeg: Not found!")
        st.caption("FFmpeg is required for video encoding.")
    
    # Video2X status
    video2x_available = check_video2x_installed()
    if video2x_available:
        st.success("✅ Video2X: Installed")
        v2x_path = get_video2x_path()
        if v2x_path:
            st.caption(f"Path: {v2x_path}")
    else:
        st.warning("⚠️ Video2X: Not found")
        st.caption("Video2X is needed for AI upscaling.")

uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"], accept_multiple_files=False)

if uploaded is not None:
    # Check if we need to create a new temp dir or reuse existing one
    # Use file name + size as a simple identifier
    file_id = f"{uploaded.name}_{uploaded.size}"
    
    if st.session_state.get('file_id') != file_id:
        # New file uploaded, reset state
        st.session_state.file_id = file_id
        st.session_state.tmp_dir = tempfile.mkdtemp()
        st.session_state.video_path = os.path.join(st.session_state.tmp_dir, uploaded.name)
        with open(st.session_state.video_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.session_state.analysis_data = None
        st.session_state.srt_content = None
        
        # Get original video resolution and store in session state
        video_info = get_video_info(st.session_state.video_path)
        if "error" not in video_info and "width" in video_info and "height" in video_info:
            st.session_state.original_width = int(video_info["width"])
            st.session_state.original_height = int(video_info["height"])
        else:
            st.session_state.original_width = None
            st.session_state.original_height = None
        
        # Force a rerun to update resolution options in sidebar
        st.rerun()
    
    tmp_dir = st.session_state.tmp_dir
    video_path = st.session_state.video_path

    # Show video preview and info
    st.subheader("Video Preview")
    preview_col, info_col = st.columns([2, 1])
    with preview_col:
        st.video(video_path)
    
    with info_col:
        st.caption("Video Information")
        info = get_video_info(video_path)
        if "error" not in info:
            st.json(info)
        else:
            st.error(f"Could not read video info: {info.get('error')}")

    # Trimming section
    st.subheader("Trim Video")
    duration = float(info.get("duration", 0)) if "error" not in info else 0
    if duration > 0:
        start_time, end_time = st.slider(
            "Select Range (seconds)", 
            0.0, 
            float(duration), 
            (0.0, float(duration)),
            step=0.1
        )
    else:
        start_time, end_time = 0.0, 0.0
        st.warning("Could not determine video duration for trimming.")

    # Action buttons
    st.subheader("Actions")
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        run_button = st.button("🔍 Analyze Video", use_container_width=True)
    
    with action_col2:
        encode_button = st.button("🎬 Process & Encode", use_container_width=True)
    
    with action_col3:
        # Placeholder for future actions
        st.empty()
    
    # Handle encoding
    if encode_button:
        if not check_ffmpeg_installed():
            st.error("FFmpeg is not installed. Cannot encode video.")
        else:
            with st.spinner("Processing video..."):
                # Resolution Logic - use the selected resolution from sidebar
                resolution = None
                if not upscale_enabled and target_width is not None and target_height is not None:
                    resolution = (target_width, target_height)
                
                # Check upscaling requirements
                if upscale_enabled and not check_video2x_installed():
                    st.error("Video2X is not installed. Cannot upscale. Please disable AI Upscaling or install Video2X.")
                    st.stop()

                # Prepare Settings
                settings = {
                    "start_time": start_time,
                    "end_time": end_time,
                    "encoding_mode": encoding_mode,
                    "bitrate": bitrate,
                    "crf": crf,
                    "target_codec": target_codec,
                    "resolution": resolution
                }
                
                output_path = os.path.join(tmp_dir, "processed_video.mp4")
                
                success, message = encode_video(video_path, output_path, settings)
                
                if success:
                    final_output = output_path
                    
                    # Apply AI enhancement if enabled
                    if enhance_enabled and enhance_options:
                        enhanced_output = os.path.join(tmp_dir, "enhanced_video.mp4")
                        with st.spinner("Applying AI enhancement filters..."):
                            enh_success, enh_message = enhance_video(
                                final_output,
                                enhanced_output,
                                enhance_options,
                                target_codec
                            )
                            if enh_success:
                                st.success(f"Enhancement complete! {enh_message}")
                                final_output = enhanced_output
                            else:
                                st.warning(f"Enhancement failed: {enh_message}. Continuing with unenhanced video.")
                    
                    # Apply AI upscaling if enabled
                    if upscale_enabled:
                        upscaled_output = os.path.join(tmp_dir, "upscaled_video.mp4")
                        with st.spinner(f"Upscaling with Video2X ({upscale_processor}, {upscale_factor}x)... this may take a while"):
                            # Use final_output (which may be enhanced) as input
                            pre_upscale_info = get_video_info(final_output)
                            up_success, up_message = upscale_video_video2x(
                                final_output, 
                                upscaled_output, 
                                scale_factor=upscale_factor,
                                processor=upscale_processor
                            )
                            if up_success:
                                # Verify the upscaled file exists and check its resolution
                                if os.path.exists(upscaled_output):
                                    upscaled_info = get_video_info(upscaled_output)
                                    if "width" in upscaled_info and "width" in pre_upscale_info:
                                        st.success(f"Upscaling complete! {pre_upscale_info.get('width')}x{pre_upscale_info.get('height')} → {upscaled_info.get('width')}x{upscaled_info.get('height')}")
                                    else:
                                        st.success("Upscaling complete!")
                                    final_output = upscaled_output
                                else:
                                    st.error("Upscaled file was not created")
                            else:
                                st.error(f"Video2X failed: {up_message}")

                    st.success("Processing Complete!")
                    
                    # Show final video info
                    final_info = get_video_info(final_output)
                    if "width" in final_info:
                        st.info(f"Output resolution: {final_info.get('width')}x{final_info.get('height')}")
                    
                    st.video(final_output)
                    
                    # Read file into memory for download to ensure correct file is served
                    with open(final_output, "rb") as f:
                        video_bytes = f.read()
                    
                    st.download_button(
                        "📥 Download Processed Video", 
                        data=video_bytes,
                        file_name="processed_video.mp4",
                        mime="video/mp4",
                        key="download_processed"
                    )
                else:
                    st.error(message)

    # Handle video analysis - run the analysis
    if run_button:
        st.divider()
        st.subheader("Analysis Results")
        st.info("Processing started — models may download and this can take many minutes. Check the server logs for progress.")
        with st.spinner("Running analysis (this can be slow)..."):
            # Call main.py as a subprocess and poll a small progress file so
            # Streamlit can show progress updates while the backend runs.
            python_exe = sys.executable or "python"
            json_path = os.path.join(tmp_dir, "analysis_results.json")
            progress_path = os.path.join(tmp_dir, "analysis_progress.json")
            cmd = [python_exe, "main.py", video_path, "--json-out", json_path, "--progress-file", progress_path]

            # Start process non-blocking so we can poll progress
            try:
                proc = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE, 
                    text=True, 
                    encoding='utf-8',
                    errors='replace',  # Replace undecodable bytes instead of crashing
                    cwd=os.getcwd()
                )
            except Exception as e:
                st.error(f"Failed to start backend: {e}")
                proc = None

            stdout = ""
            stderr = ""

            progress_bar = st.progress(0)
            status_area = st.empty()

            # Poll progress file while process runs
            if proc is not None:
                while proc.poll() is None:
                    # Read progress file if it exists
                    if os.path.exists(progress_path):
                        try:
                            with open(progress_path, 'r', encoding='utf-8') as pf:
                                p = json.load(pf)
                            pct = int(p.get('percent') or 0)
                            msg = p.get('message') or p.get('step') or ''
                            try:
                                progress_bar.progress(min(max(pct, 0), 100))
                            except Exception:
                                pass
                            status_area.text(msg)
                        except Exception:
                            # ignore transient read/parse errors
                            pass
                    time.sleep(0.5)

                # Process finished; capture output
                out, err = proc.communicate()
                stdout = out or ""
                stderr = err or ""

                # If process failed and no JSON exists, write a small error JSON
                if proc.returncode != 0 and not os.path.exists(json_path):
                    err_payload = {
                        'error': f'Backend failed with return code {proc.returncode}',
                        'stderr': (stderr[:2000] + '...') if stderr and len(stderr) > 2000 else stderr,
                        'returncode': proc.returncode
                    }
                    try:
                        with open(json_path, 'w', encoding='utf-8') as jf:
                            json.dump(err_payload, jf, ensure_ascii=False, indent=2)
                    except Exception:
                        pass

                if stderr:
                    st.text_area("Backend stderr (truncated)", stderr[:1000])

                # If JSON was produced, load it into session state
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r', encoding='utf-8') as jf:
                            st.session_state.analysis_data = json.load(jf)
                        st.success("Analysis complete!")
                    except Exception as e:
                        st.error(f"Failed to load JSON results: {e}")
                else:
                    st.warning("JSON output not produced by backend.")
                    if stdout:
                        st.download_button("Download raw output", data=stdout, file_name="analysis_output.txt")

    # Display analysis results from session state (persists across reruns)
    if st.session_state.analysis_data is not None:
        data = st.session_state.analysis_data
        
        st.divider()
        st.subheader("Analysis Results")
        
        # Visual description (from video model)
        visual_description = data.get('visual_description')
        def _clean_visual_description(text: str):
            if not text:
                return text
            try:
                cleaned = text.replace('Describe what is happening visually in this video.', '')
                cleaned = re.sub(r'^(USER:|ASSISTANT:)\s*', '', cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r'^[\s\-:\u2013\u2014]+', '', cleaned)
                return cleaned.strip()
            except Exception:
                return text.strip()

        if visual_description:
            visual_description = _clean_visual_description(visual_description)
            st.subheader("Visual description")
            st.text_area("Description", value=visual_description, height=200, key="visual_desc")
        else:
            st.info("No visual description found in JSON output.")

        # Show processing time and any errors
        total_time = data.get('total_time')
        if total_time is not None:
            st.caption(f"Total processing time: {total_time:.2f}s")

        if data.get('error'):
            st.error(f"Backend reported error: {data.get('error')}")

        # Display transcript from full whisper transcription (segments)
        transcription = data.get('transcription')
        formatted_captions = data.get('formatted_captions')

        transcript_text = ''
        if transcription and isinstance(transcription, dict) and transcription.get('segments'):
            segments = transcription['segments']
            lines = []
            for seg in segments:
                start = seg.get('start', 0)
                end = seg.get('end', 0)
                text = seg.get('text', '').strip()
                start_ts = f"{int(start//60):02d}:{int(start%60):02d}"
                end_ts = f"{int(end//60):02d}:{int(end%60):02d}"
                lines.append(f"[{start_ts} - {end_ts}] {text}")
            transcript_text = "\n".join(lines)
        elif formatted_captions:
            transcript_text = formatted_captions

        if transcript_text:
            st.subheader("Transcript")
            st.text_area("Transcript", value=transcript_text, height=300, key="transcript_text")
            
            # Subtitle options
            if transcription and isinstance(transcription, dict) and transcription.get('segments'):
                st.subheader("📝 Subtitle Options")
                
                # Generate and store SRT content in session state
                if st.session_state.srt_content is None:
                    st.session_state.srt_content = generate_srt_from_segments(transcription['segments'])
                
                srt_content = st.session_state.srt_content
                
                # Download SRT file
                st.download_button(
                    "📥 Download SRT File",
                    data=srt_content,
                    file_name="subtitles.srt",
                    mime="text/plain",
                    key="download_srt"
                )
                
                # Checkbox options to add subtitles to video
                st.write("**Add subtitles to video:**")
                
                burn_subs = st.checkbox("🔥 Burn Subtitles (Hardcoded)", value=False, 
                    help="Permanently embed subtitles into the video. Always visible, cannot be turned off.",
                    key="burn_subs_checkbox")
                
                soft_subs = st.checkbox("📎 Add Subtitle Track (Soft)", value=False,
                    help="Add subtitles as a separate track (MKV). Can be toggled on/off in video players.",
                    key="soft_subs_checkbox")
                
                # Process button
                if burn_subs or soft_subs:
                    if st.button("▶️ Create Video with Subtitles", use_container_width=True, key="create_subs_btn"):
                        # Save SRT to temp file
                        srt_path = os.path.join(tmp_dir, "subtitles.srt")
                        with open(srt_path, 'w', encoding='utf-8') as f:
                            f.write(srt_content)
                        
                        if burn_subs:
                            with st.spinner("Burning subtitles into video..."):
                                output_with_subs = os.path.join(tmp_dir, "video_with_subtitles.mp4")
                                success, message = burn_subtitles_to_video(
                                    video_path, 
                                    srt_path, 
                                    output_with_subs
                                )
                                
                                if success:
                                    st.success("Subtitles burned successfully!")
                                    st.video(output_with_subs)
                                    with open(output_with_subs, 'rb') as f:
                                        st.download_button(
                                            "📥 Download Video with Burned Subtitles",
                                            data=f,
                                            file_name="video_with_subtitles.mp4",
                                            mime="video/mp4",
                                            key="download_burned"
                                        )
                                else:
                                    st.error(f"Failed to burn subtitles: {message}")
                        
                        if soft_subs:
                            with st.spinner("Adding subtitle track to video..."):
                                output_with_subs = os.path.join(tmp_dir, "video_with_subtitles.mkv")
                                success, message = add_subtitle_track_to_video(
                                    video_path, 
                                    srt_path, 
                                    output_with_subs
                                )
                                
                                if success:
                                    st.success("Subtitle track added successfully!")
                                    st.video(output_with_subs)
                                    with open(output_with_subs, 'rb') as f:
                                        st.download_button(
                                            "📥 Download Video with Subtitle Track (MKV)",
                                            data=f,
                                            file_name="video_with_subtitles.mkv",
                                            mime="video/x-matroska",
                                            key="download_soft"
                                        )
                                else:
                                    st.error(f"Failed to add subtitle track: {message}")
        else:
            st.warning("No transcript found in JSON output.")

        # Build audio events table: include AST events and transcription-derived events
        events = []

        # AST/other model events from JSON
        ast_events = data.get('audio_events') or []
        for ev in ast_events:
            events.append({
                'event': ev.get('event'),
                'confidence': ev.get('confidence'),
                'source': ev.get('source', 'AST'),
                'timestamp': ev.get('timestamp', 'N/A')
            })

        # Derive events from transcription segments using keyword matching
        if transcription and isinstance(transcription, dict) and transcription.get('segments'):
            sound_patterns = {
                'music': ['music', '♪', '♫', 'playing', 'song', 'singing'],
                'applause': ['applause', 'clapping', 'clap'],
                'laughter': ['laughter', 'laughing', 'laugh', 'haha', 'chuckle'],
                'crowd noise': ['crowd', 'cheering', 'audience'],
                'shouting': ['shouting', 'yelling', 'scream'],
                'crying': ['crying', 'weeping', 'sobbing'],
            }
            for seg in transcription['segments']:
                text = seg.get('text', '').lower()
                start = seg.get('start', 0)
                start_ts = f"{int(start//60):02d}:{int(start%60):02d}"
                for event_name, keywords in sound_patterns.items():
                    if any(k in text for k in keywords):
                        events.append({'event': event_name, 'confidence': None, 'source': 'Whisper', 'timestamp': start_ts})

        # Deduplicate events by event+timestamp
        seen = set()
        dedup = []
        for e in events:
            key = (e.get('event'), e.get('timestamp'))
            if key not in seen:
                seen.add(key)
                dedup.append(e)

        if dedup:
            st.subheader("Detected audio events")
            st.table(dedup)
        else:
            st.info("No audio events found in JSON output.")

        # Offer JSON download
        st.subheader("JSON results")
        st.download_button(
            "Download JSON results", 
            data=json.dumps(data, indent=2, ensure_ascii=False), 
            file_name='analysis_results.json',
            key="download_json"
        )

else:
    st.info("Upload a video to get started.")
