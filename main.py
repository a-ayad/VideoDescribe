import os
import sys

# Ensure UTF-8 mode and stdout encoding very early to avoid garbled
# progress bars and Unicode in Windows CP1252 consoles. Setting these
# before importing libraries like transformers/tqdm helps their
# progress renderers choose UTF-8 or fall back to ASCII properly.
os.environ.setdefault('PYTHONUTF8', '1')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import torch
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
from transformers import AutoProcessor, ASTForAudioClassification
try:
    # Force tqdm to use ASCII bars when running in consoles that don't
    # support rich Unicode characters (many Windows terminals still use
    # cp1252). We patch tqdm.auto.tqdm to default to ascii=True.
    from functools import partial
    import tqdm.auto as _tqdm_auto
    _tqdm_auto.tqdm = partial(_tqdm_auto.tqdm, ascii=True)
except Exception:
    # non-critical; proceed without forcing ASCII if tqdm isn't available
    pass
try:
    # Also patch base tqdm (some libs import tqdm.tqdm directly)
    import tqdm
    from functools import partial as _partial
    tqdm.tqdm = _partial(tqdm.tqdm, ascii=True)
except Exception:
    pass

# Prefer to disable the HF hub progress UI entirely in console apps
# to avoid any Unicode rendering issues. Users can re-enable via env var.
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS', '1')
import numpy as np
import subprocess
import json
import time
import warnings
import os
import sys
import tempfile

# Suppress warnings
warnings.filterwarnings('ignore', message='Failed to launch Triton kernels')

# Ensure UTF-8 stdout/stderr so emoji and other Unicode print correctly on Windows
# Windows default console encoding (cp1252) can't encode many emoji and causes
# UnicodeEncodeError. Try to reconfigure sys.stdout/stderr to utf-8; if that
# fails, set PYTHONUTF8 which can help in some environments.
try:
    # sys is imported at module level
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    os.environ.setdefault('PYTHONUTF8', '1')

device = "cuda" if torch.cuda.is_available() else "cpu"

def _write_progress(progress_path, step: str, percent: float = None, message: str = None):
    """Write a small JSON progress checkpoint to progress_path.
    Writes atomically by writing to a temp file and replacing.
    """
    if not progress_path:
        return
    payload = {
        'step': step,
        'percent': percent,
        'message': message,
        'timestamp': time.time()
    }
    try:
        tmp = f"{progress_path}.{os.getpid()}.tmp"
        with open(tmp, 'w', encoding='utf-8') as pf:
            json.dump(payload, pf)
        # atomic replace
        os.replace(tmp, progress_path)
    except Exception:
        # best-effort, don't fail the pipeline for progress write errors
        pass

# ========== VIDEO MODEL ==========
print("Loading video model...")
video_model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
    device_map="auto",
    dtype=torch.float16
)

video_processor = LlavaNextVideoProcessor.from_pretrained(
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
    use_fast=True
)

torch.backends.cudnn.conv.fp32_precision = 'tf32'
torch.backends.cuda.matmul.fp32_precision = 'tf32'

try:
    video_model = torch.compile(video_model, mode="reduce-overhead")
    print("Model compiled")
except:
    pass

print("Video model loaded!\n")

# ========== AUDIO MODELS (LAZY LOADING) ==========
whisper_model = None
ast_model = None
ast_processor = None

def load_whisper_model(model_size="base"):
    """Lazy load Whisper model"""
    global whisper_model
    if whisper_model is None:
        print(f"Loading Whisper model ({model_size})...")
        import whisper
        whisper_model = whisper.load_model(model_size, device=device)
        print("Whisper model loaded!")
    return whisper_model

def load_ast_model():
    """Lazy load AST (Audio Spectrogram Transformer) model"""
    global ast_model, ast_processor
    if ast_model is None:
        print("Loading AST audio classification model...")
        ast_processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        ast_model = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        ).to(device)
        print("AST model loaded!")
    return ast_model, ast_processor

# ========== AUDIO DETECTION ==========
def check_video_has_audio(video_path):
    """
    Check if video has an audio stream
    Returns: (has_audio: bool, audio_info: dict)
    """
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=codec_name,channels,sample_rate,duration',
        '-of', 'json',
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        info = json.loads(result.stdout)
        
        if 'streams' in info and len(info['streams']) > 0:
            audio_stream = info['streams'][0]
            return True, {
                'codec': audio_stream.get('codec_name', 'unknown'),
                'channels': audio_stream.get('channels', 'unknown'),
                'sample_rate': audio_stream.get('sample_rate', 'unknown'),
                'duration': audio_stream.get('duration', 'unknown')
            }
        else:
            return False, None
    except Exception as e:
        print(f"   Warning: Could not probe audio stream: {e}")
        return False, None

# ========== AUDIO EXTRACTION ==========
def extract_audio_from_video(video_path, output_audio="temp_audio.wav"):
    """Extract audio from video using FFmpeg"""
    cmd = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        output_audio
    ]
    
    result = subprocess.run(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE
    )
    
    if result.returncode != 0:
        error_msg = result.stderr.decode('utf-8', errors='ignore')
        raise Exception(f"Failed to extract audio: {error_msg}")
    
    # Verify the file was created
    if not os.path.exists(output_audio):
        raise Exception("Audio file was not created")
    
    return output_audio

# ========== AUDIO TRANSCRIPTION ==========
def transcribe_audio(audio_path, model_size="base"):
    """Transcribe audio using Whisper"""
    print("Transcribing audio...")
    start_time = time.time()
    
    model = load_whisper_model(model_size)
    
    result = model.transcribe(
        audio_path,
        task="transcribe",
        language=None,
        verbose=False,
        word_timestamps=True
    )
    
    elapsed = time.time() - start_time
    print(f"   Transcription time: {elapsed:.3f}s")
    
    return result, elapsed

def format_transcription(result):
    """Format transcription with timestamps"""
    if not result or not result.get('segments'):
        return "[No speech detected]"
    
    formatted = []
    for segment in result['segments']:
        start = segment['start']
        end = segment['end']
        text = segment['text'].strip()
        
        start_time = f"{int(start//60):02d}:{int(start%60):02d}"
        end_time = f"{int(end//60):02d}:{int(end%60):02d}"
        
        formatted.append(f"[{start_time} - {end_time}] {text}")
    
    return "\n".join(formatted) if formatted else "[No speech detected]"

# ========== AUDIO EVENT DETECTION (COMBINED APPROACH) ==========
def detect_audio_events_ast(audio_path):
    """Detect audio events using AST model"""
    model, processor = load_ast_model()
    
    import librosa
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    
    top_probs, top_indices = torch.topk(probs, k=8)
    
    detected_events = []
    for prob, idx in zip(top_probs, top_indices):
        if prob > 0.05:  # Lower threshold for better detection
            label = model.config.id2label[idx.item()]
            detected_events.append({
                'event': label,
                'confidence': prob.item(),
                'source': 'AST Model'
            })
    
    return detected_events

def analyze_transcription_for_sounds(transcription_result):
    """Extract sound events from Whisper transcription"""
    if not transcription_result or not transcription_result.get('text'):
        return []
    
    text = transcription_result['text'].lower()
    detected_sounds = []
    
    # Sound patterns that Whisper might capture
    sound_patterns = {
        'music': ['music', '♪', '♫', 'playing', 'song', 'singing'],
        'applause': ['applause', 'clapping', 'clap'],
        'laughter': ['laughter', 'laughing', 'laugh', 'haha', 'chuckle'],
        'crowd noise': ['crowd', 'cheering', 'audience'],
        'silence': ['silence', 'quiet', '...'],
        'speech': ['speaking', 'talking', 'voice'],
        'background music': ['background music', 'music in background'],
        'shouting': ['shouting', 'yelling', 'scream'],
        'crying': ['crying', 'weeping', 'sobbing'],
    }
    
    for event_type, keywords in sound_patterns.items():
        if any(keyword in text for keyword in keywords):
            detected_sounds.append({
                'event': event_type,
                'confidence': 0.7,
                'source': 'Whisper Transcription'
            })
    
    return detected_sounds

def detect_audio_events_combined(audio_path, transcription_result=None):
    """
    Combined audio event detection using AST model + Whisper analysis
    This approach gives better accuracy than single-method detection
    """
    print("Detecting audio events (combined approach)...")
    start_time = time.time()
    
    all_events = []
    
    # 1. Get events from AST model (primary source)
    try:
        ast_events = detect_audio_events_ast(audio_path)
        all_events.extend(ast_events)
    except Exception as e:
        print(f"   Warning: AST detection failed: {e}")
    
    # 2. Analyze Whisper transcription for sound cues (secondary source)
    if transcription_result:
        whisper_sounds = analyze_transcription_for_sounds(transcription_result)
        all_events.extend(whisper_sounds)
    
    # 3. Deduplicate and merge events
    unique_events = {}
    for event in all_events:
        # Normalize event names
        key = event['event'].lower().strip()
        
        # Merge similar events
        similar_keys = {
            'speech': ['speech', 'speaking', 'voice', 'talking'],
            'music': ['music', 'background music', 'song'],
            'applause': ['applause', 'clapping', 'clap'],
            'laughter': ['laughter', 'laughing', 'laugh'],
        }
        
        # Check if this event matches a known category
        normalized_key = key
        for main_key, variants in similar_keys.items():
            if any(variant in key for variant in variants):
                normalized_key = main_key
                break
        
        # Keep highest confidence for each event type
        if normalized_key not in unique_events or event['confidence'] > unique_events[normalized_key]['confidence']:
            unique_events[normalized_key] = event
    
    # Sort by confidence
    final_events = sorted(unique_events.values(), key=lambda x: x['confidence'], reverse=True)
    
    elapsed = time.time() - start_time
    print(f"   Audio event detection time: {elapsed:.3f}s")
    print(f"   Found {len(final_events)} distinct audio events")
    
    return final_events, elapsed

# ========== VIDEO READING ==========
def get_video_info(video_path):
    """Get video info"""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_frames,duration,r_frame_rate',
        '-of', 'json',
        video_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    stream = info['streams'][0]
    
    if 'nb_frames' in stream:
        total_frames = int(stream['nb_frames'])
    else:
        duration = float(stream.get('duration', 0))
        fps_str = stream['r_frame_rate'].split('/')
        fps = float(fps_str[0]) / float(fps_str[1])
        total_frames = int(duration * fps)
    
    return total_frames

def read_video_ffmpeg(video_path, max_frames=8, resolution=336):
    """Extract video frames using FFmpeg"""
    start_time = time.time()
    
    try:
        total_frames = get_video_info(video_path)
    except:
        total_frames = 1000
    
    frame_indices = np.linspace(0, total_frames - 1, max_frames).astype(int)
    select_expr = '+'.join([f'eq(n,{idx})' for idx in frame_indices])
    
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'select=\'{select_expr}\',scale={resolution}:{resolution}',
        '-vsync', '0',
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-'
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw_frames, _ = process.communicate()
    
    frame_size = resolution * resolution * 3
    num_frames = len(raw_frames) // frame_size
    
    frames = []
    for i in range(num_frames):
        start = i * frame_size
        end = start + frame_size
        frame_data = np.frombuffer(raw_frames[start:end], dtype=np.uint8)
        frame = frame_data.reshape((resolution, resolution, 3))
        frames.append(frame)
    
    elapsed = time.time() - start_time
    return frames, elapsed

# ========== VIDEO DESCRIPTION ==========
def describe_video_visual(video_path, max_frames=8, resolution=336):
    """Generate visual description"""
    print("\nAnalyzing video (visual)...")
    frames, video_read_time = read_video_ffmpeg(video_path, max_frames, resolution)
    print(f"   Frame extraction: {video_read_time:.3f}s ({len(frames)} frames)")
    
    prompt = "USER: <video>\nDescribe what is happening visually in this video. ASSISTANT:"
    
    preprocess_start = time.time()
    inputs = video_processor(text=prompt, videos=frames, return_tensors="pt")
    
    processed_inputs = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            if k in ['input_ids', 'attention_mask']:
                processed_inputs[k] = v.long().to(device)
            else:
                processed_inputs[k] = v.to(device, dtype=torch.float16)
        else:
            processed_inputs[k] = v
    
    preprocess_time = time.time() - preprocess_start
    
    inference_start = time.time()
    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output = video_model.generate(
                **processed_inputs,
                max_new_tokens=300,
                do_sample=False,
                use_cache=True
            )
    
    inference_time = time.time() - inference_start
    description = video_processor.decode(output[0], skip_special_tokens=True)

    # Clean out any prompt markers so the returned description begins with
    # the model's natural language output. Some model checkpoints echo the
    # prompt (e.g. "USER: <video>... ASSISTANT:") — strip those prefixes.
    def _clean_visual_description(text):
        if not text:
            return text
        # Remove exact prompt occurrences first
        try:
            # Common prompt used earlier
            candidate_prompts = [
                "USER: <video>\nDescribe what is happening visually in this video. ASSISTANT:",
                "USER: <video>\nDescribe what is happening visually in this video.\nASSISTANT:",
                "USER: <video>\nASSISTANT:",
                "USER:",
                "ASSISTANT:"
            ]
            desc = text
            for p in candidate_prompts:
                desc = desc.replace(p, "")
            import re as _re
            # Remove any leading label like 'USER:' or 'ASSISTANT:' (case-insensitive)
            desc = _re.sub(r'^(USER:|ASSISTANT:)\s*', '', desc, flags=_re.IGNORECASE)
            # Trim leading punctuation/whitespace leftover
            desc = _re.sub(r'^[\s\-:\u2013\u2014]+', '', desc)
            return desc.strip()
        except Exception:
            return text.strip()

    description = _clean_visual_description(description)
    
    print(f"   Video inference: {preprocess_time + inference_time:.3f}s")
    
    return description

# ========== COMPLETE ANALYSIS ==========
def analyze_video_complete(
    video_path, 
    max_frames=8, 
    resolution=336,
    enable_transcription=True,
    enable_audio_events=True,
    whisper_model_size="base"
):
    """
    Complete video analysis with audio stream detection
    """
    print("="*70)
    print("PROCESSING VIDEO")
    print("="*70)
    # If caller provided a progress file via env or arg, it will be handled
    # by the outer scope. This function accepts an optional `progress_file`
    # via closure or argument if integrated; for now leave hook points.
    
    # Check that the video file exists before doing any heavy work
    if not os.path.exists(video_path):
        print(f"\nError: video file not found: {video_path}\nPlease check the path and try again.")
        return {
            'visual_description': None,
            'has_audio': False,
            'audio_info': None,
            'transcription': None,
            'formatted_captions': None,
            'audio_events': None,
            'total_time': 0.0,
            'error': 'video_not_found',
        }

    total_start = time.time()
    
    # Check for audio stream FIRST
    has_audio = False
    audio_info = None
    
    if enable_transcription or enable_audio_events:
        print(f"\nChecking for audio stream...")
        _write_progress(progress_file if 'progress_file' in globals() else None, 'checking_audio', 5, 'Checking for audio stream')
        has_audio, audio_info = check_video_has_audio(video_path)
        if has_audio:
            print("   Audio stream detected:")
            print(f"     - Codec: {audio_info['codec']}")
            print(f"     - Channels: {audio_info['channels']}")
            print(f"     - Sample Rate: {audio_info['sample_rate']} Hz")
        else:
            print("   Warning: No audio stream detected in video")
            print("   Skipping audio transcription and event detection")
    
    # 1. Visual Description (always enabled)
    _write_progress(progress_file if 'progress_file' in globals() else None, 'visual_description_start', 10, 'Starting visual description')
    visual_description = describe_video_visual(video_path, max_frames, resolution)
    _write_progress(progress_file if 'progress_file' in globals() else None, 'visual_description_done', 45, 'Visual description complete')
    
    # Initialize results
    transcription_result = None
    formatted_captions = None
    audio_events = None
    audio_path = None
    
    # 2. Audio Processing (ONLY if audio exists and features enabled)
    if has_audio and (enable_transcription or enable_audio_events):
        try:
            print(f"\nExtracting audio track...")
            _write_progress(progress_file if 'progress_file' in globals() else None, 'extract_audio', 50, 'Extracting audio track')
            audio_path = extract_audio_from_video(video_path)
            
            # 3. Transcription (run first to help with audio event detection)
            if enable_transcription:
                transcription_result, transcribe_time = transcribe_audio(audio_path, whisper_model_size)
                _write_progress(progress_file if 'progress_file' in globals() else None, 'transcription_done', 75, 'Transcription complete')
                formatted_captions = format_transcription(transcription_result)
            
            # 4. Audio Events (combined approach using AST + Whisper)
            if enable_audio_events:
                audio_events, event_time = detect_audio_events_combined(
                    audio_path, 
                    transcription_result
                )
                _write_progress(progress_file if 'progress_file' in globals() else None, 'audio_events_done', 85, 'Audio event detection complete')
            
        except Exception as e:
            print(f"   Warning: Error processing audio: {e}")
            has_audio = False
        
        finally:
            # Clean up temp audio file
            if audio_path:
                try:
                    os.remove(audio_path)
                except:
                    pass
    
    total_time = time.time() - total_start
    _write_progress(progress_file if 'progress_file' in globals() else None, 'done', 100, 'Analysis complete')
    
    # ========== PRINT RESULTS ==========
    print("\n" + "="*70)
    print("ANALYSIS RESULTS")
    print("="*70)
    
    print(f"\nVISUAL DESCRIPTION:")
    print("-"*70)
    print(visual_description)
    
    if enable_transcription:
        print(f"\n\nCLOSED CAPTIONS (Speech Transcription):")
        print("-"*70)
        if not has_audio:
            print("[Video has no audio stream]")
        elif formatted_captions:
            print(formatted_captions)
        else:
            print("[No speech detected]")
    
    if enable_audio_events:
        print(f"\n\nDETECTED AUDIO EVENTS:")
        print("-"*70)
        if not has_audio:
            print("  [Video has no audio stream]")
        elif audio_events:
            for event in audio_events:
                source_icon = "[AST]" if event['source'] == 'AST Model' else "[WHISPER]"
                print(f"  {source_icon} {event['event'].upper()}: {event['confidence']*100:.1f}% ({event['source']})")
        else:
            print("  [No significant audio events detected]")
    
    print(f"\n\nTIMING:")
    print("-"*70)
    print(f"  Total processing time: {total_time:.2f}s")
    
    print("="*70)
    
    return {
        'visual_description': visual_description,
        'has_audio': has_audio,
        'audio_info': audio_info,
        'transcription': transcription_result,
        'formatted_captions': formatted_captions,
        'audio_events': audio_events,
        'total_time': total_time
    }

# ========== MAIN ==========
if __name__ == "__main__":
    import argparse
    
    # ========== CONFIGURATION ==========
    DEFAULT_CONFIG = {
        'enable_transcription': True,
        'enable_audio_events': True,
        'whisper_model_size': 'base',
        'max_frames': 8,
        'resolution': 336
    }
    
    try:
        parser = argparse.ArgumentParser(
            description="Video Analysis for Accessibility",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python main.py video.mp4
  python main.py video.mp4 --no-transcription
  python main.py video.mp4 --no-audio-events
  python main.py video.mp4 --whisper-model small --frames 12
            """
        )

        parser.add_argument('video_path', type=str, help='Path to video file')
        parser.add_argument('--no-transcription', action='store_true',
                            help='Disable transcription')
        parser.add_argument('--no-audio-events', action='store_true',
                            help='Disable audio event detection')
        parser.add_argument('--whisper-model', type=str, default='base',
                            choices=['tiny', 'base', 'small', 'medium', 'large'],
                            help='Whisper model size (default: base)')
        parser.add_argument('--frames', type=int, default=8,
                            help='Number of frames to extract (default: 8)')
        parser.add_argument('--resolution', type=int, default=336,
                            help='Frame resolution (default: 336)')
        parser.add_argument('--json-out', type=str, default=None,
                            help='Path to write structured JSON results')
        parser.add_argument('--progress-file', type=str, default=None,
                            help='Path to write small JSON progress updates')

        args = parser.parse_args()

        video_path = args.video_path
        enable_transcription = not args.no_transcription
        enable_audio_events = not args.no_audio_events
        whisper_model_size = args.whisper_model
        max_frames = args.frames
        resolution = args.resolution
        json_out = args.json_out
        progress_file = args.progress_file
        # expose progress_file to functions via module globals
        globals()['progress_file'] = progress_file
        # write initial progress
        try:
            _write_progress(progress_file, 'started', 0, 'Started analysis')
        except Exception:
            pass

    except Exception:
        # No command-line args
        print("Using default configuration")
        video_path = "input2.mp4"  # CHANGE THIS!
        enable_transcription = DEFAULT_CONFIG['enable_transcription']
        enable_audio_events = DEFAULT_CONFIG['enable_audio_events']
        whisper_model_size = DEFAULT_CONFIG['whisper_model_size']
        max_frames = DEFAULT_CONFIG['max_frames']
        resolution = DEFAULT_CONFIG['resolution']
        json_out = None
    
    # Show configuration
    print("="*70)
    print("VIDEO ANALYSIS FOR ACCESSIBILITY")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Video: {video_path}")
    print(f"  Transcription: {'Enabled' if enable_transcription else 'Disabled'}")
    print(f"  Audio Events: {'Enabled' if enable_audio_events else 'Disabled'}")
    if enable_transcription:
        print(f"  Whisper Model: {whisper_model_size}")
    print(f"  Video Frames: {max_frames}")
    print(f"  Resolution: {resolution}x{resolution}")
    print("="*70)
    
    # Check the video file exists before running analysis
    if not os.path.exists(video_path):
        print(f"Error: video file '{video_path}' does not exist. Exiting.")
        sys.exit(1)

    # Run analysis (wrap in try/except so we always produce a JSON file
    # even if something goes wrong; Streamlit expects a JSON file).
    try:
        result = analyze_video_complete(
            video_path,
            max_frames=max_frames,
            resolution=resolution,
            enable_transcription=enable_transcription,
            enable_audio_events=enable_audio_events,
            whisper_model_size=whisper_model_size
        )
    except Exception as e:
        # Capture the exception and return a minimal result dict with error
        print(f"Fatal error during analysis: {e}")
        import traceback as _traceback
        _traceback.print_exc()
        result = {
            'visual_description': None,
            'has_audio': False,
            'audio_info': None,
            'transcription': None,
            'formatted_captions': None,
            'audio_events': None,
            'total_time': 0.0,
            'error': str(e)
        }

    # Write structured JSON output if requested
    try:
        if json_out:
            def _make_serializable(o):
                """Recursively convert numpy/torch objects to native Python types for JSON."""
                # Lazy import torch here to avoid top-level dependency if not installed
                try:
                    import torch as _torch
                except Exception:
                    _torch = None

                if isinstance(o, dict):
                    return {k: _make_serializable(v) for k, v in o.items()}
                if isinstance(o, list):
                    return [_make_serializable(v) for v in o]
                if isinstance(o, tuple):
                    return tuple(_make_serializable(v) for v in o)
                # numpy types
                try:
                    import numpy as _np
                    if isinstance(o, _np.ndarray):
                        return o.tolist()
                    # numpy scalar types
                    if isinstance(o, (_np.floating,)):
                        return float(o)
                    if isinstance(o, (_np.integer,)):
                        return int(o)
                except Exception:
                    pass
                # torch tensors
                if _torch is not None:
                    try:
                        if _torch.is_tensor(o):
                            return _make_serializable(o.cpu().numpy())
                    except Exception:
                        pass
                # basic types
                if isinstance(o, (str, int, float, bool)) or o is None:
                    return o
                # fallback to string
                try:
                    return str(o)
                except Exception:
                    return None

            serializable = _make_serializable(result)
            with open(json_out, 'w', encoding='utf-8') as jf:
                json.dump(serializable, jf, ensure_ascii=False, indent=2)
            print(f"Wrote JSON results to: {json_out}")
    except Exception as e:
        print(f"Warning: failed to write JSON output: {e}")
