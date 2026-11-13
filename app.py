import os
import shutil
import tempfile

import gradio as gr
import requests

import pipeline as P

MAX_MINUTES = int(os.environ.get("MAX_MINUTES", "90"))
MAX_CLIPS = 5
GAME = "Gameplay (action detection)"
TALK = "Talking video (speech + AI)"


def fetch_url(url, workdir):
    low = url.lower()
    if "youtube.com" in low or "youtu.be" in low:
        raise RuntimeError("YouTube links are not supported (downloading breaks YouTube's terms). "
                           "Upload the video file instead, or paste a direct .mp4 link.")
    path = os.path.join(workdir, "source.mp4")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        if "text/html" in r.headers.get("content-type", ""):
            raise RuntimeError("That link is a web page, not a direct video file.")
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return path


def process(video_file, url, mode, n_clips, length, captions, language, progress=gr.Progress()):
    empty = [None] * MAX_CLIPS
    workdir = tempfile.mkdtemp(prefix="clipforge_")
    try:
        if video_file:
            src = os.path.join(workdir, "source" + os.path.splitext(video_file)[1])
            shutil.copy(video_file, src)
        elif url and url.strip():
            progress(0.02, desc="Downloading video")
            src = fetch_url(url.strip(), workdir)
        else:
            return ["Upload a video or paste a direct video link.", None, *empty]

        total = P.probe_duration(src)
        if total > MAX_MINUTES * 60:
            return [f"Video is {total / 60:.0f} min. Limit is {MAX_MINUTES} min.", None, *empty]

        n = int(n_clips)
        lang = None if language == "auto" else language
        segments = []

        if mode == GAME:
            secs = 30 if length == "30 sec" else 60
            progress(0.1, desc="Scanning loudness and motion (takes a few minutes)")
            picks = P.gameplay_clips(src, total, n, secs)
            if captions:
                progress(0.45, desc="Transcribing commentary")
                chunks, _ = P.extract_audio_chunks(src, workdir)
                segments = P.transcribe(chunks, lang)
            layout = "blur"
        else:
            min_s, max_s = (25, 40) if length == "30 sec" else (50, 70)
            progress(0.1, desc="Extracting audio")
            chunks, _ = P.extract_audio_chunks(src, workdir)
            progress(0.25, desc="Transcribing")
            segments = P.transcribe(chunks, lang)
            if not segments:
                return ["No speech found. For gameplay without talking, choose Gameplay mode.", None, *empty]
            progress(0.5, desc="Finding engaging moments")
            picks = P.pick_clips(segments, n, min_s, max_s)
            picks = P.snap_and_filter(picks, segments, n, min_s, max_s, total)
            layout = "crop"

        if not picks:
            return ["Could not find good clips. Try a different video or length.", None, *empty]

        outdir = tempfile.mkdtemp(prefix="clips_")
        files, log = [], []
        for i, clip in enumerate(picks):
            progress(0.55 + 0.4 * i / len(picks), desc=f"Cutting clip {i + 1}/{len(picks)}")
            out = P.cut_clip(src, segments, clip, workdir, i, bool(captions), layout)
            final = os.path.join(outdir, os.path.basename(out))
            shutil.move(out, final)
            files.append(final)
            log.append(f"Clip {i + 1}: {clip['title']}  (score {clip['score']}, "
                       f"{clip['start']:.0f}s-{clip['end']:.0f}s)\n   {clip['reason']}")
        return ["\n".join(log), files, *(files + empty)[:MAX_CLIPS]]
    except Exception as e:
        return [f"Error: {e}", None, *empty]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


with gr.Blocks(title="ClipForge") as demo:
    gr.Markdown("# ✂️ ClipForge\nTurn a long video into 9:16 clips. "
                "Gameplay mode finds the loudest, most intense moments with no API keys needed. "
                "Only use videos you own or have permission to clip.")
    with gr.Row():
        with gr.Column():
            video = gr.File(label="Upload video", file_types=["video"], type="filepath")
            url = gr.Textbox(label="...or direct video link (.mp4)", placeholder="https://...")
            mode = gr.Radio([GAME, TALK], value=GAME, label="Mode")
            n_clips = gr.Slider(1, MAX_CLIPS, value=3, step=1, label="Number of clips")
            length = gr.Radio(["30 sec", "60 sec"], value="30 sec", label="Clip length")
            captions = gr.Checkbox(value=False, label="Burn in captions (needs GROQ_API_KEY)")
            language = gr.Dropdown(["auto", "en", "hi", "ta", "te", "bn", "mr", "gu", "kn", "ml", "ur"],
                                   value="auto", label="Spoken language")
            btn = gr.Button("Create clips", variant="primary")
        with gr.Column():
            log = gr.Textbox(label="Result", lines=8)
            files = gr.Files(label="Download clips")
    players = [gr.Video(label=f"Clip {i + 1}") for i in range(MAX_CLIPS)]
    btn.click(process, [video, url, mode, n_clips, length, captions, language], [log, files, *players])

demo.queue(max_size=5)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
