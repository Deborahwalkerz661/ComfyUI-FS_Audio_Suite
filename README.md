# The Fixed Seed Company Audio Suite

**FS_Audio Suite** — modular YuE2 audio generation for ComfyUI, by Make the Robot Do It.

![FS_Audio Suite in ComfyUI](docs/fs_audio_suite.png)
 Six nodes, no stock nodes needed. Hum-to-song is the first feature; the suite is built to grow.

| node | role |
|---|---|
| **🎤 FS_Audio Model Loader** | YuE2 checkpoint (`yue2_3b_bf16.safetensors` from Comfy-Org/YuE2) + optional melody transcriber (`sheetsage2_bf16.safetensors` in `models/audio_encoders/`). Left-side inputs take chained **loras** and **adapters** and apply them. Outputs one **pipe**. |
| **🧩 FS_Audio Lora Loader** | Any LoRA in `models/loras/` with model / clip strengths (AR-side LoRAs such as the instrumental LoRA act on CLIP; decoder LoRAs act on MODEL). Chain several by feeding one into the next. |
| **🎚 FS_Audio Adapter Loader** | A conditioning adapter and its behaviour. Today: the hum adapter (`humsong_yue2_adapter_v1_comfy.safetensors`) with **hum influence** (0–3, 1.0 as trained), **melody** mode (continue from hum / hum only / ignore hum) and **hum offset**. Chainable. |
| **🎙 Hum Input** | Record in the node, upload, or pick a file. Play / scrub the waveform, download the take. |
| **🎵 FS_Audio Sampler** | style, lyrics (bare tags: `[intro] [verse] [chorus] [bridge] [outro]`), seed, length cap, **score mode** (full / melody / off), steps, sampler, scheduler (ComfyUI's native lists; defaults `dpm_2` + `sgm_uniform` match the official template). **hum is optional**: without it this is a plain YuE2 generator with the pipe's LoRAs; with a hum and a hum adapter it is hum-to-song. |
| **💿 FS_Audio Output** | FLAC to `output/`, in-node player, download, and the score with the hummed bars highlighted. |

Wiring: Lora Loader(s) → Model Loader `loras`; Adapter Loader(s) → Model Loader `adapters`; Model Loader → Sampler `pipe`; Hum Input → Sampler `hum` (optional); Sampler → Output (`song`, `score`, `info`).

Rules the sampler follows: a connected hum with an adapter whose melody mode is not *ignore* forces a melody-mode score continued from the hum; otherwise **score mode** decides how the planner writes its score. Hum influence 0 keeps the adapter's decoder LoRA but skips the injection. If no adapter is loaded but a hum is connected, the hum still opens the score (continuation) and no injection happens.

Install: clone into `ComfyUI/custom_nodes/ComfyUI-FS_Audio_Suite`, `pip install -r requirements.txt` (librosa, scipy, soundfile). Weights: the hum adapter and other LoRAs from the Make the Robot Do It model cards.
