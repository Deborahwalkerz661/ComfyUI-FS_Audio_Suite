# The Fixed Seed Company Audio Suite

**FS_Audio Suite** — modular YuE2 audio generation for ComfyUI, by Fixed Seed LLC and Make the Robot Do It
**Special thanks to @machinedelusions for the support in making this possible**
Check him out at www.fixedseed.com

![FS_Audio Suite in ComfyUI](docs/fs_audio_suite.png)
 Six nodes, no stock nodes needed. Hum-to-song is the first feature; the suite is built to grow.

| node | role |
|---|---|
| **🎤 FS_Audio Model Loader** | YuE2 checkpoint (`yue2_3b_bf16.safetensors` from Comfy-Org/YuE2) + optional melody transcriber (`sheetsage2_bf16.safetensors` in `models/audio_encoders/`). Left-side inputs take chained **loras** and **adapters** and apply them. Outputs one **pipe**. |
| **🧩 FS_Audio Lora Loader** | Any LoRA in `models/loras/` with model / clip strengths (AR-side LoRAs such as the instrumental LoRA act on CLIP; decoder LoRAs act on MODEL). Chain several by feeding one into the next. |
| **🎚 FS_Audio Adapter Loader** | A conditioning adapter and its behaviour. Today: the hum adapter (`humsong_yue2_adapter_v1_comfy.safetensors`) with **hum influence** (0–3, 1.0 as trained), **melody** mode (continue from hum / hum only / ignore hum) and **hum offset**. Chainable. |
| **🎙 Hum Input** | Record in the node, upload, or pick a file. Play / scrub the waveform, download the take. |
| **🎵 FS_Audio Sampler** | style, lyrics (bare tags: `[intro] [verse] [chorus] [bridge] [outro]`), seed, length cap, **score mode** (full / melody / off), steps, sampler, scheduler (ComfyUI's native lists; defaults `dpm_2` + `sgm_uniform` match the official template), **repetition_penalty** (1.2 reference; 1.0 lets a memorized song replay), **Weirdness (cfg)** (1.0 = YuE2's native single-pass decoding, the recommended setting; above 1 the decoder is pushed away from an unconditioned pass for stranger, less safe renders at twice the render time). **hum is optional**: without it this is a plain YuE2 generator with the pipe's LoRAs; with a hum and a hum adapter it is hum-to-song. |
| **💿 FS_Audio Output** | FLAC to `output/`, in-node player, download, and the score with the hummed bars highlighted. |

Wiring: Model Loader (with `sheetsage2` if you want scores) → Dataset Builder and Artist Trainer `pipe`; Dataset Builder → Artist Trainer `dataset`; Regularizer → Artist Trainer `regularizer`; Artist Trainer `loras` → a second Model Loader `loras` → Sampler. Load the current joint decoder LoRA (`nar_lora_joint_v9_comfyui`, from Training Assets) into the *first* Model Loader so both dataset tokens and training run on the real-audio stack.

Captions: the builder does not caption for you. Put `<song>.txt` (style) and `<song>.lyrics.txt` next to each song, or rely on the defaults. Style text is flattened to one line with straight quotes; lyrics are normalised to YuE2's native section layout (`[Intro]`, `[Verse 2]`, `[Pre-Chorus]`, `[Chorus]`, `[Bridge]`, `[Outro]`, Title case with optional numbers; any other bracketed line is treated as a production note and dropped; literal `\n` becomes a line break). Scores that fail to export are repaired by retrying melody-only and on the first four minutes before a track is given up on. With `transcribe_scores` on, `auto_tempo_key` appends the transcribed tempo and key to captions that lack them. Memory: 8192 tokens fits a 24 GB card (~15 GB peak); use 12288 on 40 GB+.

## Credits

- **[Ostris](https://github.com/ostris/ai-toolkit)**'s ai-toolkit YuE2 extension is the model for the Artist Trainer: one loop for both experts with the decoder conditioned on the detached planner context, the KL(base ‖ lora) trust region on the planner, chunked + checkpointed next-token logits so whole songs fit in memory, YuE2-native caption/section normalisation, SheetSage repair instead of dropping tracks, and the EMA warm-up form. Thank you.

- **[Ostris](https://github.com/ostris/ai-toolkit)**'s YuE2 extension for ai-toolkit carries a fix we had missed: MERT-v2's rotary `inv_freq` buffer is non-persistent and newer transformers leave it uninitialised, so the Dataset Builder's audio → token step was running MERT with no positional encoding (only ~37% of tokens matched the correct ones). Fixed in the tokenizer by rebuilding the buffer after load, the same way ai-toolkit does. Datasets built before this fix should be rebuilt.

- **[@AIWarper](https://x.com/AIWarper)** found and reported the Decoder Adapter Trainer bug where the `vae2llm` / `llm2vae` layers received no gradient under ComfyUI's dynamic-VRAM weight casting, so every exported `.diff` was zero while the LoRA tensors trained normally. Fixed in bf4ba64: the trainer now runs those layers through its own fp32 parameters and warns if they ever export unchanged. Thank you.
