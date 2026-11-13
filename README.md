---
title: ClipForge
emoji: ✂️
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
---

# ClipForge

Upload a long video you own and get 9:16 short clips (30 or 60 sec) with captions.

Pipeline: ffmpeg extracts audio -> Groq Whisper transcribes -> Gemini picks the most engaging
moments -> ffmpeg cuts, crops to vertical and burns in captions.

## Secrets (Space Settings -> Variables and secrets)
- `GROQ_API_KEY`   (free: https://console.groq.com)
- `GEMINI_API_KEY` (free: https://aistudio.google.com/apikey)

Optional variables: `GEMINI_MODEL` (default `gemini-2.5-flash`), `MAX_MINUTES` (default 90).

Only clip videos you own or have permission to use.
