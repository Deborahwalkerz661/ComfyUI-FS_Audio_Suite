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
| **🎵 FS_Audio Sampler** | style, lyrics (YuE2's section tags: `[Intro] [Verse 1] [Pre-Chorus] [Chorus] [Bridge] [Outro]`), seed, length cap, **score mode** (full / melody / off), steps, sampler, scheduler (ComfyUI's native lists; defaults `dpm_2` + `sgm_uniform` match the official template), **repetition_penalty** (1.2 reference; 1.0 lets a memorized song replay), **Weirdness (cfg)** (1.0 = YuE2's native single-pass decoding, the recommended setting; above 1 the decoder is pushed away from an unconditioned pass for stranger, less safe renders at twice the render time). **hum is optional**: without it this is a plain YuE2 generator with the pipe's LoRAs; with a hum and a hum adapter it is hum-to-song. |
| **💿 FS_Audio Output** | FLAC to `output/`, in-node player, download, and the score with the hummed bars highlighted. |

Wiring: Lora Loader(s) → Model Loader `loras`; Adapter Loader(s) → Model Loader `adapters`; Model Loader → Sampler `pipe`; Hum Input → Sampler `hum` (optional); Sampler → Output (`song`, `score`, `info`).

Rules the sampler follows: a connected hum with an adapter whose melody mode is not *ignore* forces a melody-mode score continued from the hum; otherwise **score mode** decides how the planner writes its score. Hum influence 0 keeps the adapter's decoder LoRA but skips the injection. If no adapter is loaded but a hum is connected, the hum still opens the score (continuation) and no injection happens.

Install: clone into `ComfyUI/custom_nodes/ComfyUI-FS_Audio_Suite`, `pip install -r requirements.txt` (librosa, scipy, soundfile). Weights: the hum adapter and other LoRAs from the Make the Robot Do It model cards.

## Training your own LoRA (FS_Audio / Training)

Four nodes turn a folder of songs into **one artist LoRA** (planner + decoder in a single file) that plugs straight back into the **Model Loader**. Everything runs on ComfyUI's own YuE2 weights; no other checkpoint is downloaded.

| node | role |
|---|---|
| **⬇ FS_Audio Training Assets** | standalone; downloads the audio → token head (v9 current, trained with an audio-domain loss against real recordings; v8 and v5 also offered, v4 as legacy), the matching decoder LoRA `nar_lora_joint_v9_comfyui` (into `models/loras/`), and the minted regularizer pack into `models/fs_audio/`. Run once per asset. Load the joint LoRA in the Model Loader chain whenever you train or render from real-audio tokens. |
| **📦 FS_Audio Dataset Builder** | folder of songs (`<song>.txt` = style caption, `<song>.lyrics.txt` = lyrics; defaults for missing ones) → real-audio tokens (public MERT-v2-FullSong layer 20 + our head), lyrics normalized to bare tags, optional chord-annotated ABC scores from the pipe's melody transcriber, hash-based hold-out. |
| **🧪 FS_Audio Regularizer** | the minted regularizer pack (`models/fs_audio/*.pt`): YuE2's own songs, coin-flipped against the artist so the LoRA keeps the base model's range. |
| **🏋 FS_Audio Artist Trainer** | trains a whole artist in one run: the **planner** LoRA (what they write) and the **decoder** LoRA (how they sound) in one loop on ComfyUI's own YuE2 weights, exported as **one file** that a single FS_Audio Lora Loader applies to both halves (`text_encoders.*` + `diffusion_model.*`). Saves `<name>_best` and `<name>_step…` into `models/loras/`, plus a JSON report and `<name>_log.json`. |

The recipe inside the Artist Trainer (all switchable):

- **Planner**: whole-song next-token CE from the song's start (so lyrics and music never learn to start at arbitrary offsets), **batch_songs** whole songs per step with an **artist_fraction** coin flip against the minted regularizer (artist and regularizer songs share one update), **score_first_fraction** of songs with the transcribed score in front of the music while the planner *always* also learns to write the score, **end_token_weight**, **kl_weight**: a trust region KL(base ‖ lora) on the next-token distributions (base = the same sequence with the LoRA switched off). Logits are computed 512 positions at a time inside a checkpoint, so whole 8k-token songs fit on 24 GB.
- **Decoder**: **decoder_steps** flow-matching steps spread across the run, on **window_seconds** windows of the artist's VAE latents, conditioned on the *current* planner's context (recomputed every step, detached so the flow loss never trains the planner); rank-**rank_decoder** LoRA on the decoder plus the full vae2llm / llm2vae layers at **io_lr**.
- **EMA** (**ema_decay**, warm-up form so the decay never overshoots): every exported checkpoint is the EMA of both halves.
- **resume_from**: continue from any earlier Artist Trainer checkpoint (same ranks); the ladder is the record, so stopping a run never loses it.
- **Ladder**: **eval_every** (held-out artist CE, regularizer CE and held-out decoder flow, all plotted live), **checkpoint_from / checkpoint_every**, **seed**. Pick rungs **by ear**: held-out CE on a handful of songs tells you about memorization, not about whether the artist comes through, and on our runs the rungs the metric called overfit were the ones that sounded most like the band. Keep the whole ladder and listen.

Wiring: Model Loader (with `sheetsage2` if you want scores) → Dataset Builder and Artist Trainer `pipe`; Dataset Builder → Artist Trainer `dataset`; Regularizer → Artist Trainer `regularizer`; Artist Trainer `loras` → a second Model Loader `loras` → Sampler. Load the current joint decoder LoRA (`nar_lora_joint_v9_comfyui`, from Training Assets) into the *first* Model Loader so both dataset tokens and training run on the real-audio stack.

Captions: the builder does not caption for you. Put `<song>.txt` (style) and `<song>.lyrics.txt` next to each song, or rely on the defaults. Style text is flattened to one line with straight quotes; lyrics are normalised to YuE2's native section layout (`[Intro]`, `[Verse 2]`, `[Pre-Chorus]`, `[Chorus]`, `[Bridge]`, `[Outro]`, Title case with optional numbers; any other bracketed line is treated as a production note and dropped; literal `\n` becomes a line break). Scores that fail to export are repaired by retrying melody-only and on the first four minutes before a track is given up on. With `transcribe_scores` on, `auto_tempo_key` appends the transcribed tempo and key to captions that lack them. Memory: 8192 tokens fits a 24 GB card (~15 GB peak); use 12288 on 40 GB+.

## Credits

- **[Ostris](https://github.com/ostris/ai-toolkit)**'s ai-toolkit YuE2 extension is the model for the Artist Trainer: one loop for both experts with the decoder conditioned on the detached planner context, the KL(base ‖ lora) trust region on the planner, chunked + checkpointed next-token logits so whole songs fit in memory, YuE2-native caption/section normalisation, SheetSage repair instead of dropping tracks, and the EMA warm-up form. Thank you.

- **[Ostris](https://github.com/ostris/ai-toolkit)**'s YuE2 extension for ai-toolkit carries a fix we had missed: MERT-v2's rotary `inv_freq` buffer is non-persistent and newer transformers leave it uninitialised, so the Dataset Builder's audio → token step was running MERT with no positional encoding (only ~37% of tokens matched the correct ones). Fixed in the tokenizer by rebuilding the buffer after load, the same way ai-toolkit does. Datasets built before this fix should be rebuilt.

- **[@AIWarper](https://x.com/AIWarper)** found and reported the Decoder Adapter Trainer bug where the `vae2llm` / `llm2vae` layers received no gradient under ComfyUI's dynamic-VRAM weight casting, so every exported `.diff` was zero while the LoRA tensors trained normally. Fixed in bf4ba64: the trainer now runs those layers through its own fp32 parameters and warns if they ever export unchanged. Thank you.
