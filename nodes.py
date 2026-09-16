"""HumSong: hum in, lyrics + style in, finished song out. Four nodes, no stock nodes required.
Stage 1 (score continuation): the hum's melody is transcribed to ABC and placed as the OPEN beginning of YuE2's score; the AR writes the rest
of the score, then the song's semantic tokens. Stage 2 (prosody adapter): the hum's pitch/timing, as a sine carrier encoded by the VAE, is injected
into the NAR decoder at several depths so the rendered vocal follows the actual hum where it is present."""
import os, logging, re, json, types, hashlib, numpy as np, torch, torch.nn.functional as F, torchaudio
import folder_paths, comfy.sd, comfy.utils, comfy.sample, comfy.samplers, comfy.model_management, comfy.model_prefetch
import comfy.audio_encoders.audio_encoders
from comfy.ldm.yue2 import model as yue2_model
from comfy_extras.nodes_audio import load as load_audio_file
from server import PromptServer
from .prosody import prosody_sine

FPS = 25
def _status(node_id, stage, pct=None, detail=""):
    try: PromptServer.instance.send_sync("humsong.stage", {"node": node_id, "stage": stage, "pct": pct, "detail": detail})
    except Exception: pass

# ---------------------------------------------------------------- decoder forward with hum injection
def _hum_forward(self, x, timestep, context, yue2_chunks, transformer_options={}, **kwargs):
    H = self._humsong; batch, channels, frames = x.shape
    if frames != yue2_chunks[-1][1]: raise ValueError("YuE2 latent duration must match the conditioning.")
    config = self.config; time = self.time_embedder(timestep.to(x.dtype), x.dtype)[:, None]
    attention = yue2_model.optimized_attention_for_device(x.device)
    condp = F.pad(H["cond"].to(x.device, x.dtype), (0, 0, 1, 1))                      # [frames+2, 64], zero where there is no hum
    projs = [(w.to(x.device, x.dtype), b.to(x.device, x.dtype)) for w, b in H["proj"]]
    def run(use_hum):
        output = torch.empty_like(x)
        for start, end, kv_start, kv_end in yue2_chunks:
            comfy.model_management.throw_exception_if_processing_interrupted()
            ar_length = kv_end - kv_start; length = end - start + 2
            state = F.pad(x[..., start:end].transpose(1, 2), (0, 0, 1, 1))
            state = self.vae2llm(state) + time + self.latent_pos_embed(length, x)[None]
            hc = condp[start:end + 2][None] if use_hum else torch.zeros(1, length, 64, device=x.device, dtype=x.dtype)
            adds = {H["inject"][i]: F.linear(hc, w, b) for i, (w, b) in enumerate(projs)}
            if 0 in adds: state = state + adds[0]
            positions = torch.arange(ar_length, ar_length + length, device=x.device)[None]
            rope = yue2_model.precompute_freqs_cis(config.head_dim, positions, config.rope_theta, device=x.device)
            prefix = context[:, kv_start:kv_end].reshape(batch, ar_length, config.num_hidden_layers, 2, config.num_key_value_heads, config.head_dim).permute(2, 3, 0, 4, 1, 5)
            prefetch = comfy.model_prefetch.make_prefetch_queue(list(self.model.layers), x.device, transformer_options)
            for index, layer in enumerate(self.model.layers):
                comfy.model_prefetch.prefetch_queue_pop(prefetch, x.device, layer, state.dtype)
                if index > 0 and index in adds: state = state + adds[index]
                state, _ = layer(state, freqs_cis=rope, optimized_attention=attention, past_key_value=(prefix[index, 0], prefix[index, 1], ar_length))
            comfy.model_prefetch.prefetch_queue_pop(prefetch, x.device, None)
            output[..., start:end] = self.llm2vae(self.model.norm(state))[:, 1:-1].transpose(1, 2)
        return output
    g = H["g"]
    if g == 1.0: return run(True)
    o0 = run(False); return o0 + g * (run(True) - o0)          # classifier-free guidance on the hum channel

# ---------------------------------------------------------------- nodes (FS_Audio suite)
CAT = "🎤 FS_Audio"
def _lora_files(): return folder_paths.get_filename_list("loras")

class FSAudioLoraLoader:
    """A LoRA for the pipe (AR-side LoRAs act on CLIP, decoder-side LoRAs on MODEL). Chain several by feeding one loader into the next."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"lora_name": (_lora_files(),), "strength_model": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 3.0, "step": 0.05}), "strength_clip": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 3.0, "step": 0.05})},
                "optional": {"loras": ("FS_AUDIO_LORAS",)}}
    RETURN_TYPES = ("FS_AUDIO_LORAS",); RETURN_NAMES = ("loras",); FUNCTION = "add"; CATEGORY = CAT
    def add(self, lora_name, strength_model, strength_clip, loras=None):
        return (list(loras or []) + [{"name": lora_name, "path": folder_paths.get_full_path_or_raise("loras", lora_name), "model": strength_model, "clip": strength_clip}],)

class FSAudioAdapterLoader:
    """A conditioning adapter for the pipe. Today: the hum adapter (decoder LoRA + hum injection) and its behaviour."""
    @classmethod
    def INPUT_TYPES(cls):
        files = _lora_files(); hum = [f for f in files if "hum" in f.lower()] or files
        return {"required": {"adapter_name": (hum,),
                             "hum_influence": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1, "tooltip": "How hard the decoder follows the hum's pitch and timing where it is present. 0 = injection off (the adapter's decoder LoRA still applies)."}),
                             "melody": (["continue from hum", "hum only", "ignore hum"], {"tooltip": "continue: the hum opens the score and the model finishes it. hum only: the score is exactly the hum. ignore: the model writes its own melody."}),
                             "hum_offset_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 120.0, "step": 0.5, "tooltip": "Where in the song the hum's phrasing is applied."})},
                "optional": {"adapters": ("FS_AUDIO_ADAPTERS",)}}
    RETURN_TYPES = ("FS_AUDIO_ADAPTERS",); RETURN_NAMES = ("adapters",); FUNCTION = "add"; CATEGORY = CAT
    def add(self, adapter_name, hum_influence, melody, hum_offset_seconds, adapters=None):
        return (list(adapters or []) + [{"type": "hum", "name": adapter_name, "path": folder_paths.get_full_path_or_raise("loras", adapter_name), "influence": hum_influence, "melody": melody, "offset": hum_offset_seconds}],)

class FSAudioModelLoader:
    """YuE2 checkpoint (+ optional melody transcriber) with any chained LoRAs and adapters applied -> one pipe."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"yue2_checkpoint": (folder_paths.get_filename_list("checkpoints"),), "melody_transcriber": (["none"] + folder_paths.get_filename_list("audio_encoders"),)},
                "optional": {"loras": ("FS_AUDIO_LORAS",), "adapters": ("FS_AUDIO_ADAPTERS",)}}
    RETURN_TYPES = ("FS_AUDIO_PIPE",); RETURN_NAMES = ("pipe",); FUNCTION = "load"; CATEGORY = CAT
    def load(self, yue2_checkpoint, melody_transcriber, loras=None, adapters=None):
        model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(folder_paths.get_full_path_or_raise("checkpoints", yue2_checkpoint), output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        applied = []
        for l in loras or []:
            sd = comfy.utils.load_torch_file(l["path"], safe_load=True); model, clip = comfy.sd.load_lora_for_models(model, clip, sd, l["model"], l["clip"]); applied.append(f"{l['name']} (model {l['model']}, clip {l['clip']})")
        ads = []
        for a in adapters or []:
            sd = comfy.utils.load_torch_file(a["path"], safe_load=True)
            lora_sd = {k: v for k, v in sd.items() if k.startswith("diffusion_model.")}; hum = {k[len("yue2_hum."):]: v for k, v in sd.items() if k.startswith("yue2_hum.")}
            if lora_sd: model, clip = comfy.sd.load_lora_for_models(model, clip, lora_sd, 1.0, 0.0)
            inject = hum["inject_layers"].tolist() if "inject_layers" in hum else [0]
            proj = [(hum[f"hum_proj.{i}.weight"].float(), hum[f"hum_proj.{i}.bias"].float()) for i in range(len(inject))] if hum else []
            ads.append({**a, "inject": inject, "proj": proj}); applied.append(f"adapter {a['name']}")
        encoder = None
        if melody_transcriber != "none":
            enc_sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("audio_encoders", melody_transcriber), safe_load=True)
            encoder = comfy.audio_encoders.audio_encoders.load_audio_encoder_from_sd(enc_sd)
            if encoder is None: raise RuntimeError("melody_transcriber is not a valid audio encoder (expected sheetsage2_bf16.safetensors)")
        return ({"model": model, "clip": clip, "vae": vae, "encoder": encoder, "adapters": ads, "applied": applied, "ckpt_path": folder_paths.get_full_path_or_raise("checkpoints", yue2_checkpoint)},)

class HumInput:
    """Your hum. Pick a file, upload, or press Record in the node."""
    @classmethod
    def INPUT_TYPES(cls):
        d = folder_paths.get_input_directory()
        files = sorted(f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)) and f.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm", ".aac", ".opus")))
        return {"required": {"audio": (files or ["(record or upload a hum)"], {}), "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 600.0, "step": 0.1}), "max_seconds": ("FLOAT", {"default": 30.0, "min": 3.0, "max": 120.0, "step": 0.5})}}
    RETURN_TYPES = ("AUDIO",); RETURN_NAMES = ("audio_conditioning",); FUNCTION = "load"; CATEGORY = CAT
    def load(self, audio, start_seconds, max_seconds):
        wav, sr = load_audio_file(folder_paths.get_annotated_filepath(audio))
        wav = wav.mean(0, keepdim=True)[:, int(start_seconds * sr):int((start_seconds + max_seconds) * sr)]
        return ({"waveform": wav[None], "sample_rate": sr},)
    @classmethod
    def IS_CHANGED(cls, audio, start_seconds, max_seconds):
        m = hashlib.sha256(open(folder_paths.get_annotated_filepath(audio), "rb").read()); return f"{m.hexdigest()}:{start_seconds}:{max_seconds}"
    @classmethod
    def VALIDATE_INPUTS(cls, audio, **kw):
        return True if folder_paths.exists_annotated_filepath(audio) else f"Hum file not found: {audio}"

class FSAudioSampler:
    """Style + lyrics (+ optional hum) -> song. Without a hum it is a plain YuE2 generator with the pipe's LoRAs; with a hum and a hum adapter it is hum-to-song."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "pipe": ("FS_AUDIO_PIPE",),
            "style": ("STRING", {"multiline": True, "default": "groove funk, tight syncopated bass, clavinet, wah guitar, punchy horn stabs, live drums, upbeat, male vocals, 104 BPM"}),
            "lyrics": ("STRING", {"multiline": True, "default": "[intro]\n\n[verse]\n...\n\n[chorus]\n...\n\n[outro]\n"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "song_length_cap": ("FLOAT", {"default": 240.0, "min": 30.0, "max": 360.0, "step": 1.0, "tooltip": "Upper bound in seconds; the song ends when the score ends."}),
            "score_mode": (["full", "melody", "off"], {"tooltip": "How the planner writes its score when no hum melody is used: full (melody + chords), melody, or off (no score)."}),
            "steps": ("INT", {"default": 32, "min": 8, "max": 64}), "sampler": (cls._samplers(), {"default": "dpm_2"}), "scheduler": (cls._schedulers(), {"default": "sgm_uniform"})},
            "optional": {"audio_conditioning": ("AUDIO",), "score_temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.5, "step": 0.05}), "music_temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.5, "step": 0.05})},
            "hidden": {"unique_id": "UNIQUE_ID"}}
    RETURN_TYPES = ("AUDIO", "STRING", "STRING"); RETURN_NAMES = ("song", "score", "info"); FUNCTION = "sample"; CATEGORY = CAT
    @staticmethod
    def _samplers():
        names = list(comfy.samplers.KSampler.SAMPLERS); return ["dpm_2"] + [n for n in names if n != "dpm_2"]
    @staticmethod
    def _schedulers():
        names = list(comfy.samplers.KSampler.SCHEDULERS); return ["sgm_uniform"] + [n for n in names if n != "sgm_uniform"]
    def sample(self, pipe, style, lyrics, seed, song_length_cap, score_mode, steps, sampler, scheduler, audio_conditioning=None, score_temperature=0.7, music_temperature=1.0, unique_id=None):
        hum = audio_conditioning
        model, clip, vae, enc = pipe["model"], pipe["clip"], pipe["vae"], pipe.get("encoder"); nid = unique_id
        lyrics = lyrics.replace("\\n", "\n").replace("\r", "").strip(); style = " ".join(style.replace("\\n", " ").split())
        ad = next((a for a in pipe.get("adapters", []) if a.get("type") == "hum"), None)
        melody = (ad["melody"] if ad else "continue from hum") if hum is not None else "none"; influence = float(ad["influence"]) if (ad and hum is not None) else 0.0; offset = float(ad["offset"]) if ad else 0.0
        # -- hum melody -> ABC
        abc_hum = ""
        if hum is not None and melody != "ignore hum":
            if enc is None: raise ValueError("A hum is connected but the pipe has no melody_transcriber; pick sheetsage2 in the Model Loader.")
            _status(nid, "Listening to your hum", 5)
            abc_hum = enc.generate_abc(hum["waveform"], hum["sample_rate"], melody_only=True)[0].strip(); lines = abc_hum.split("\n")
            while lines and re.fullmatch(r"(V: (Vocal|Ins))|(Z\d*\|)|", lines[-1].strip()): lines.pop()
            abc_hum = "\n".join(lines) + "\n"
        hum_lines = abc_hum.count("\n")
        # -- score
        use_hum_melody = bool(abc_hum); cot = "melody" if use_hum_melody else score_mode
        if cot == "off": full_abc = ""; _status(nid, "Composing the score", 15, "score off")
        elif melody == "hum only": full_abc = abc_hum; _status(nid, "Composing the score", 15, "hum only")
        else:
            _status(nid, "Composing the score", 15, f"{hum_lines} lines from your hum" if use_hum_melody else f"{cot} score")
            tokens = clip.tokenize(style, lyrics=lyrics, cot=cot, seed=seed, max_tokens=8192, penalty_window=100)
            if use_hum_melody: tokens["prefix"] = tokens["prefix"] + clip.tokenizer.tokenizer.encode(abc_hum).ids      # open score: the planner keeps writing it
            ids = clip.generate(tokens, max_length=8192, temperature=score_temperature, top_p=0.9, top_k=30, repetition_penalty=1.005, seed=seed)
            full_abc = abc_hum + clip.decode(ids)
        # -- semantic tokens + acoustic conditioning
        _status(nid, "Writing the song", 35, f"score {full_abc.count(chr(10))} lines" if full_abc else "no score")
        kw = {"abc": full_abc} if (full_abc and cot != "off") else {}
        tokens2 = clip.tokenize(style, lyrics=lyrics, cot=cot, seed=seed, max_tokens=max(200, int(song_length_cap * FPS)), temperature=music_temperature, top_p=0.95, top_k=100, repetition_penalty=1.2, **kw)
        cond = clip.encode_from_tokens_scheduled(tokens2); frames = cond[0][1]["yue2_frames"]
        negative = [[torch.zeros_like(cond[0][0]), dict(cond[0][1])]]
        # -- render (with hum injection when an adapter and a hum are present)
        _status(nid, "Rendering", 60, f"{frames / FPS:.0f} s")
        dev = comfy.model_management.intermediate_device()
        latent = torch.zeros((1, 64, frames), device=dev, dtype=comfy.model_management.intermediate_dtype()); noise = comfy.sample.prepare_noise(latent, seed)
        dm = model.model.diffusion_model; patched = False
        if hum is not None and ad and ad.get("proj") and influence > 0:
            y = hum["waveform"][0].mean(0).float().cpu().numpy(); sr = hum["sample_rate"]
            if sr != 48000: y = torchaudio.functional.resample(torch.from_numpy(y), sr, 48000).numpy()
            carrier, voiced, onset, sing_s = prosody_sine(y.astype(np.float32), 48000)
            if carrier is not None:
                vsr = getattr(vae, "audio_sample_rate", 48000); w = torch.from_numpy(np.stack([carrier, carrier], 0))[None]
                if vsr != 48000: w = torchaudio.functional.resample(w, 48000, vsr)
                zh = vae.encode(w.movedim(1, -1))[0].T.float().cpu(); condl = torch.zeros(frames, 64); o = int(offset * FPS); L = min(len(zh), frames - o)
                if L > 0: condl[o:o + L] = zh[:L]
                dm._humsong = {"cond": condl, "proj": ad["proj"], "inject": ad["inject"], "g": influence}; dm.forward = types.MethodType(_hum_forward, dm); patched = True
        try: samples = comfy.sample.sample(model, noise, steps, 1.0, sampler, scheduler, cond, negative, latent, denoise=1.0, seed=seed)
        finally:
            if patched: del dm.forward; del dm._humsong
        _status(nid, "Decoding audio", 90)
        audio = vae.decode(samples).movedim(-1, 1); std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0; std[std < 1.0] = 1.0; audio = audio / std
        out_sr = getattr(vae, "audio_sample_rate_output", getattr(vae, "audio_sample_rate", 48000))
        score_view = (abc_hum + ("% ---- your hum ends here; the model continues ----\n" if melody == "continue from hum" and use_hum_melody else "") + full_abc[len(abc_hum):]) if full_abc else "(score off)"
        info = json.dumps({"seconds": round(frames / FPS, 1), "audio_conditioning": hum is not None, "hum": hum is not None, "hum_score_lines": hum_lines, "score_lines": full_abc.count("\n"), "melody": melody, "score_mode": cot, "hum_influence": influence, "hum_offset": offset,
                           "adapter": ad["name"] if ad else None, "loras": pipe.get("applied", []), "seed": seed, "sampler": sampler, "scheduler": scheduler, "steps": steps}, indent=1)
        _status(nid, "Done", 100, f"{frames / FPS:.0f} s")
        return ({"waveform": audio, "sample_rate": out_sr}, score_view, info)

class FSAudioOutput:
    """Saves the song as FLAC (plus .abc score and .json info sidecars) and shows a player + the score."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"song": ("AUDIO",), "filename_prefix": ("STRING", {"default": "FS_Audio/song"})}, "optional": {"score": ("STRING", {"forceInput": True}), "info": ("STRING", {"forceInput": True})}}
    RETURN_TYPES = (); FUNCTION = "save"; OUTPUT_NODE = True; CATEGORY = CAT
    def save(self, song, filename_prefix="FS_Audio/song", score="", info=""):
        full_out, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory()); results = []
        for i in range(song["waveform"].shape[0]):
            name = f"{filename}_{counter + i:05}_.flac"; path = os.path.join(full_out, name)
            try:
                import soundfile as sf; sf.write(path, song["waveform"][i].float().cpu().numpy().T, song["sample_rate"], subtype="PCM_24")
            except ImportError: torchaudio.save(path, song["waveform"][i].float().cpu(), song["sample_rate"], format="flac")
            results.append({"filename": name, "subfolder": subfolder, "type": "output"})
            # sidecars: the planner's score and the generation info land next to the FLAC
            try:
                if score: open(path[:-5] + ".abc", "w").write(score)
                if info: open(path[:-5] + ".json", "w").write(info)
            except OSError as e: logging.warning(f"FS_Audio Output: could not write sidecars for {name}: {e}")
        return {"ui": {"audio": results, "humsong_score": [score], "humsong_info": [info]}}

ADAPTERS = {   # name -> (direct URL, destination folder type)
    "hum_adapter_v1 (hum-to-song)": ("https://huggingface.co/Mothersuperior/YuE2-hum-to-song/resolve/main/humsong_yue2_adapter_v1_comfy.safetensors", "loras"),
}
BASE_MODELS = [("https://huggingface.co/Comfy-Org/YuE2/resolve/main/checkpoints/yue2_3b_bf16.safetensors", "checkpoints"),
               ("https://huggingface.co/Comfy-Org/YuE2/resolve/main/audio_encoders/sheetsage2_bf16.safetensors", "audio_encoders")]

def _download(url, dest, node_id=None):
    """Streaming HTTPS download to dest (.part then rename), with progress messages."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "FS_Audio/1.0"}); name = os.path.basename(dest)
    with urllib.request.urlopen(req) as r, open(dest + ".part", "wb") as f:
        total = int(r.headers.get("Content-Length") or 0); got = 0; last = -1; pbar = comfy.utils.ProgressBar(total or 1)
        while True:
            chunk = r.read(8 << 20)
            if not chunk: break
            f.write(chunk); got += len(chunk)
            pct = int(got * 100 / total) if total else None
            if pct is not None and pct != last: last = pct; pbar.update_absolute(got); _status(node_id, "Downloading", pct, f"{name} {got / 2**20:.0f} / {total / 2**20:.0f} MB")
    os.replace(dest + ".part", dest)

class FSAudioAdapterDownloader:
    """Standalone: downloads an FS_Audio adapter (and optionally the YuE2 base checkpoint + melody transcriber) into the ComfyUI models folders."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"adapter": (list(ADAPTERS.keys()),), "download_base_models": ("BOOLEAN", {"default": False, "tooltip": "Also fetch Comfy-Org's yue2_3b_bf16 checkpoint (7.8 GB) and sheetsage2 transcriber (1.3 GB) if missing."})},
                "hidden": {"unique_id": "UNIQUE_ID"}}
    RETURN_TYPES = (); FUNCTION = "download"; OUTPUT_NODE = True; CATEGORY = CAT
    @classmethod
    def IS_CHANGED(cls, adapter, download_base_models, unique_id=None): return float("nan")   # always re-check
    def download(self, adapter, download_base_models, unique_id=None):
        jobs = [ADAPTERS[adapter]] + (BASE_MODELS if download_base_models else []); done = []
        for url, folder in jobs:
            dest_dir = folder_paths.get_folder_paths(folder)[0]; os.makedirs(dest_dir, exist_ok=True); name = url.rsplit("/", 1)[-1]; dest = os.path.join(dest_dir, name)
            if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000: done.append(f"{name} — already in models/{folder}"); continue
            _status(unique_id, "Downloading", 0, name)
            try: _download(url, dest, unique_id)
            except Exception as e:
                try: os.remove(dest + ".part")
                except OSError: pass
                raise RuntimeError(f"Could not download {name}: {type(e).__name__}: {str(e)[:200]}") from e
            done.append(f"{name} -> models/{folder}")
        _status(unique_id, "Done", 100, f"{len(done)} file(s)")
        return {"ui": {"fs_download": ["\n".join(done)]}}

NODE_CLASS_MAPPINGS = {"FSAudioModelLoader": FSAudioModelLoader, "FSAudioLoraLoader": FSAudioLoraLoader, "FSAudioAdapterLoader": FSAudioAdapterLoader, "HumInput": HumInput, "FSAudioSampler": FSAudioSampler, "FSAudioOutput": FSAudioOutput, "FSAudioAdapterDownloader": FSAudioAdapterDownloader}
NODE_DISPLAY_NAME_MAPPINGS = {"FSAudioModelLoader": "🎤 FS_Audio Model Loader", "FSAudioLoraLoader": "🧩 FS_Audio Lora Loader", "FSAudioAdapterLoader": "🎚 FS_Audio Adapter Loader", "HumInput": "🎙 Hum Input", "FSAudioSampler": "🎵 FS_Audio Sampler", "FSAudioOutput": "💿 FS_Audio Output", "FSAudioAdapterDownloader": "⬇ FS_Audio Adapter Downloader"}
