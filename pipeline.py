import json
import math
import os
import re
import subprocess
import time

import requests

GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL = "whisper-large-v3-turbo"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
CHUNK_SECONDS = 1200  # 20 min audio chunks keep each upload far below Groq's size limit


def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd[:4])}...\n{p.stderr[-800:]}")
    return p.stdout


def probe_duration(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "csv=p=0", path])
    return float(out.strip())


def extract_audio_chunks(video, workdir):
    dur = probe_duration(video)
    chunks = []
    for i in range(math.ceil(dur / CHUNK_SECONDS)):
        offset = i * CHUNK_SECONDS
        out = os.path.join(workdir, f"audio_{i}.mp3")
        run(["ffmpeg", "-y", "-ss", str(offset), "-t", str(CHUNK_SECONDS), "-i", video,
             "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", out])
        chunks.append((out, offset))
    return chunks, dur


def transcribe(chunks, language=None):
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it in Space Settings -> Secrets.")
    segments = []
    for path, offset in chunks:
        for attempt in range(4):
            with open(path, "rb") as f:
                data = {"model": GROQ_MODEL, "response_format": "verbose_json",
                        "timestamp_granularities[]": "segment"}
                if language:
                    data["language"] = language
                r = requests.post(GROQ_URL, headers={"Authorization": f"Bearer {key}"},
                                  files={"file": f}, data=data, timeout=300)
            if r.status_code == 429:
                time.sleep(15 * (attempt + 1))
                continue
            if r.status_code != 200:
                raise RuntimeError(f"Groq error {r.status_code}: {r.text[:300]}")
            for s in r.json().get("segments", []):
                text = s["text"].strip()
                if text:
                    segments.append({"start": s["start"] + offset,
                                     "end": s["end"] + offset, "text": text})
            break
        else:
            raise RuntimeError("Groq rate limit hit repeatedly. Wait a minute and retry.")
    return segments


def pick_clips(segments, n, min_s, max_s):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it in Space Settings -> Secrets.")
    transcript = "\n".join(f"[{int(s['start'])}-{int(s['end'])}] {s['text']}" for s in segments)
    prompt = f"""You are an expert short-form video editor (Reels, Shorts, TikTok).
From the timestamped transcript below, choose the {n} most engaging moments.
Rules:
- Each clip must be between {min_s} and {max_s} seconds long.
- Start on a strong hook (a bold claim, question, story start or surprise).
- Must be a complete, self-contained thought that ends on a payoff, not mid-sentence.
- Clips must not overlap.
- Timestamps are in seconds and must come from the transcript.
Return ONLY a JSON array: [{{"start": number, "end": number, "title": "short catchy title", "score": 1-10, "reason": "why it is engaging"}}]

TRANSCRIPT:
{transcript}"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.4}}
    for attempt in range(3):
        r = requests.post(url, headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                          json=body, timeout=180)
        if r.status_code == 429:
            time.sleep(20 * (attempt + 1))
            continue
        if r.status_code != 200:
            raise RuntimeError(f"Gemini error {r.status_code}: {r.text[:300]}")
        break
    else:
        raise RuntimeError("Gemini rate limit hit. Wait a minute and retry.")
    text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    text = re.sub(r"^```(?:json)?|```$", "", text.strip()).strip()
    return json.loads(text)


def snap_and_filter(clips, segments, n, min_s, max_s, total):
    starts = [s["start"] for s in segments]
    ends = [s["end"] for s in segments]
    result = []
    for c in clips:
        try:
            st = min(starts, key=lambda x: abs(x - float(c["start"])))
            en = min(ends, key=lambda x: abs(x - float(c["end"])))
        except (KeyError, ValueError, TypeError):
            continue
        if en - st > max_s + 10:
            valid = [e for e in ends if st + min_s - 5 <= e <= st + max_s + 5]
            if not valid:
                continue
            en = valid[-1]
        en = min(en, total)
        if en - st < min_s - 10:
            continue
        if any(st < r["end"] and en > r["start"] for r in result):
            continue
        result.append({"start": st, "end": en, "title": str(c.get("title", "Clip")),
                       "score": c.get("score", 0), "reason": str(c.get("reason", ""))})
    result.sort(key=lambda x: float(x["score"] or 0), reverse=True)
    return result[:n]


def srt_time(t):
    t = max(t, 0)
    h, m = int(t // 3600), int(t % 3600 // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{int(s):02d},{int((s - int(s)) * 1000):03d}"


def build_srt(segments, start, end, path):
    lines, idx = [], 1
    for s in segments:
        if s["end"] <= start or s["start"] >= end:
            continue
        words = s["text"].split()
        groups = [words[i:i + 5] for i in range(0, len(words), 5)] or [[]]
        span = (min(s["end"], end) - max(s["start"], start))
        t0 = max(s["start"], start) - start
        total_words = max(len(words), 1)
        pos = 0
        for g in groups:
            d = span * len(g) / total_words
            lines.append(f"{idx}\n{srt_time(t0 + pos)} --> {srt_time(t0 + pos + d)}\n{' '.join(g)}\n")
            pos += d
            idx += 1
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def cut_clip(video, seg_list, clip, workdir, index, captions=True, layout="crop"):
    """layout: 'crop' = centre crop to 9:16, 'blur' = full gameplay over blurred background."""
    start, end = clip["start"], clip["end"]
    out_name = f"clip_{index + 1}.mp4"
    srt_filter = ""
    if captions and seg_list:
        srt = f"subs_{index + 1}.srt"
        build_srt(seg_list, start, end, os.path.join(workdir, srt))
        style = "FontName=DejaVu Sans,FontSize=11,Bold=1,Outline=2,Alignment=2,MarginV=%d" % (
            70 if layout == "blur" else 45)
        srt_filter = f",subtitles={srt}:force_style='{style}'"
    base = ["ffmpeg", "-y", "-ss", str(start), "-i", os.path.abspath(video), "-t", str(end - start)]
    tail = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-r", "30",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", out_name]
    if layout == "blur":
        fc = ("[0:v]split=2[a][b];"
              "[a]scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,boxblur=20:5[bg];"
              "[b]scale=720:-2[fg];"
              f"[bg][fg]overlay=(W-w)/2:(H-h)/2{srt_filter}[v]")
        cmd = base + ["-filter_complex", fc, "-map", "[v]", "-map", "0:a?"] + tail
    else:
        vf = ("crop='min(iw,ih*9/16)':ih,scale=720:1280:force_original_aspect_ratio=increase,"
              "crop=720:1280" + srt_filter)
        cmd = base + ["-vf", vf] + tail
    run(cmd, cwd=workdir)
    return os.path.join(workdir, out_name)


# ---------------------------------------------------------------- gameplay mode
def _norm(x):
    import numpy as np
    top = np.percentile(x, 95) if len(x) else 0
    return np.clip(x / (top + 1e-9), 0, 1) if top > 0 else np.zeros_like(x, dtype=float)


def audio_energy(video, total_sec):
    """Loudness (RMS) for every second of the video."""
    import numpy as np
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vn", "-ac", "1", "-ar", "8000",
                        "-f", "s16le", "-"], capture_output=True)
    data = np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32)
    n = min(len(data) // 8000, total_sec)
    out = np.zeros(total_sec, dtype=np.float32)
    if n > 0:
        out[:n] = np.sqrt((data[:n * 8000].reshape(n, 8000) ** 2).mean(axis=1))
    return out


def motion_energy(video, total_sec):
    """How much the picture changes every second (chases, shootouts, explosions)."""
    import numpy as np
    w, h = 64, 36
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-an", "-sn",
                        "-vf", f"fps=2,scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
                       capture_output=True)
    frames = np.frombuffer(p.stdout, dtype=np.uint8)
    nf = len(frames) // (w * h)
    out = np.zeros(total_sec, dtype=np.float32)
    if nf < 2:
        return out
    frames = frames[:nf * w * h].reshape(nf, w * h).astype(np.int16)
    d = np.abs(np.diff(frames, axis=0)).mean(axis=1)
    idx = ((np.arange(len(d)) + 1) * 0.5).astype(int)
    ok = idx < total_sec
    np.maximum.at(out, idx[ok], d[ok])
    return out


def gameplay_clips(video, total, n, length):
    """Pick the n most intense non-overlapping windows using loudness + motion."""
    import numpy as np
    total_sec = int(total)
    aud = _norm(audio_energy(video, total_sec))
    mot = _norm(motion_energy(video, total_sec))
    score = 0.6 * aud + 0.4 * mot
    score = np.convolve(score, np.ones(3) / 3, mode="same")
    L = int(length)
    if total_sec <= L:
        return [{"start": 0, "end": total, "title": "Action moment 1", "score": 10,
                 "reason": "Video is shorter than the clip length."}]
    cs = np.concatenate([[0], np.cumsum(score ** 2)])
    win = (cs[L:] - cs[:-L]) / L
    win = win.astype(float).copy()
    picks = []
    for k in range(n):
        i = int(np.argmax(win))
        if not np.isfinite(win[i]) or win[i] <= 0:
            break
        a, m = float(aud[i:i + L].mean()), float(mot[i:i + L].mean())
        picks.append({"start": float(i), "end": float(min(i + L, total)),
                      "title": f"Action moment {k + 1}", "score": round(float(win[i]) * 10, 1),
                      "reason": f"loudness {a:.2f}, motion {m:.2f} (0-1 scale)"})
        win[max(0, i - L + 1): i + L] = -np.inf
    return picks
