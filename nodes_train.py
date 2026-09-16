"""FS_Audio Suite — training nodes (additive; the inference nodes are untouched)."""
import os, re, json, time, glob, numpy as np, torch
import folder_paths, comfy.model_management
from server import PromptServer
from comfy_extras.nodes_audio import load as load_audio_file
from .nodes import _download, CAT
from .fs_train.tokenizer import AudioTokenizer, assets_dir
from .fs_train.data import list_audio, sidecar, normalize_lyrics, held, load_regularizer
from .fs_train import trainer as fs_trainer
from .fs_train import decoder as fs_decoder
from .fs_train import artist as fs_artist
TCAT = CAT + "/Training"
def _msg(node_id, **kw):
    try: PromptServer.instance.send_sync("fsaudio.train", {"node": node_id, **kw})
    except Exception: pass
def _datasets_dir(): d = os.path.join(folder_paths.get_output_directory(), "fs_audio", "datasets"); os.makedirs(d, exist_ok=True); return d
def _fs_files(ext): return sorted(f for f in os.listdir(assets_dir()) if f.endswith(ext))

class FSAudioTrainAssets:
    """Standalone: downloads the audio->token head (and other training assets) into models/fs_audio."""
    ASSETS = {"tokenizer_head_joint_v9 (audio -> YuE2 tokens, current; audio-loss trained)": ("https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4/resolve/main/tokenizer_head_joint_v9.safetensors", "fs_audio"),
              "nar_lora_joint_v9_comfyui (decoder LoRA paired with the v9 head)": ("https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4/resolve/main/nar_lora_joint_v9_comfyui.safetensors", "loras"),
              "tokenizer_head_joint_v8 (audio-loss weight 2.0)": ("https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4/resolve/main/tokenizer_head_joint_v8.safetensors", "fs_audio"),
              "nar_lora_joint_v8_comfyui (decoder LoRA paired with the v8 head)": ("https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4/resolve/main/nar_lora_joint_v8_comfyui.safetensors", "loras"),
              "tokenizer_head_joint_v5 (latent-loss only)": ("https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4/resolve/main/tokenizer_head_joint_v5.safetensors", "fs_audio"),
              "nar_lora_joint_v5_comfyui (decoder LoRA paired with the v5 head)": ("https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4/resolve/main/nar_lora_joint_v5_comfyui.safetensors", "loras"),
              "tokenizer_head_joint_v4 (audio -> YuE2 tokens, previous)": ("https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4/resolve/main/tokenizer_head_joint_v4.safetensors", "fs_audio"),
              "minted_regularizer_pack_v2 (12,247 songs, for the trainer)": ("https://huggingface.co/Mothersuperior/YuE2-hum-to-song/resolve/main/minted_regularizer_pack_v2.pt", "fs_audio")}
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"asset": (list(cls.ASSETS.keys()),)}, "hidden": {"unique_id": "UNIQUE_ID"}}
    RETURN_TYPES = (); FUNCTION = "download"; OUTPUT_NODE = True; CATEGORY = TCAT
    @classmethod
    def IS_CHANGED(cls, asset, unique_id=None): return float("nan")
    def download(self, asset, unique_id=None):
        url, folder = self.ASSETS[asset]; ddir = folder_paths.get_folder_paths("loras")[0] if folder == "loras" else assets_dir(); dest = os.path.join(ddir, url.rsplit("/", 1)[-1])
        if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000: return {"ui": {"fs_download": [f"{os.path.basename(dest)} — already in models/{folder}"]}}
        _download(url, dest, unique_id); return {"ui": {"fs_download": [f"{os.path.basename(dest)} -> models/{folder}"]}}

class FSAudioDatasetBuilder:
    """A folder of songs -> training dataset: real-audio tokens (public MERT + our head), captions/lyrics from sidecars, optional scores from the pipe's melody transcriber."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pipe": ("FS_AUDIO_PIPE",), "audio_folder": ("STRING", {"default": "train/my_artist", "tooltip": "Folder of songs. Relative paths resolve under ComfyUI/input. Per-song sidecars: <song>.txt = style caption, <song>.lyrics.txt = lyrics."}),
                             "dataset_name": ("STRING", {"default": "my_artist"}), "tokenizer_head": (_fs_files(".safetensors") or ["(run FS_Audio Training Assets first)"],),
                             "default_style": ("STRING", {"multiline": True, "default": "", "tooltip": "Used for songs without a .txt caption sidecar."}),
                             "default_lyrics": ("STRING", {"multiline": True, "default": "[instrumental]", "tooltip": "Used for songs without a .lyrics.txt sidecar."}),
                             "transcribe_scores": ("BOOLEAN", {"default": True, "tooltip": "Transcribe each song to a chord-annotated ABC score with the pipe's melody transcriber, enabling the score-first training recipe."}),
                             "hold_out_percent": ("INT", {"default": 5, "min": 0, "max": 50}), "max_minutes": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 16.0, "step": 0.5}),
                             "store_latents": ("BOOLEAN", {"default": True, "tooltip": "Also store each song's VAE latents (needed by the Decoder Adapter Trainer)."}),
                             "auto_tempo_key": ("BOOLEAN", {"default": True, "tooltip": "When scores are transcribed, append 'N BPM, key of X' to captions that don't state them."})},
                "hidden": {"unique_id": "UNIQUE_ID"}}
    RETURN_TYPES = ("FS_AUDIO_DATASET", "STRING"); RETURN_NAMES = ("dataset", "report"); FUNCTION = "build"; CATEGORY = TCAT
    def build(self, pipe, audio_folder, dataset_name, tokenizer_head, default_style, default_lyrics, transcribe_scores, hold_out_percent, max_minutes, store_latents=True, auto_tempo_key=True, unique_id=None):
        folder = audio_folder if os.path.isabs(audio_folder) else os.path.join(folder_paths.get_input_directory(), audio_folder); files = list_audio(folder)
        if not files: raise ValueError(f"No audio files in {folder}")
        head = os.path.join(assets_dir(), tokenizer_head)
        if not os.path.exists(head): raise ValueError("Tokenizer head missing: run the FS_Audio Training Assets node first.")
        enc = pipe.get("encoder") if transcribe_scores else None; repaired = []
        if transcribe_scores and enc is None: raise ValueError("transcribe_scores needs a melody_transcriber in the Model Loader (sheetsage2).")
        _msg(unique_id, stage="Loading tokenizer", pct=0); tk = AudioTokenizer(head); items = []; t0 = time.time()
        for i, f in enumerate(files):
            comfy.model_management.throw_exception_if_processing_interrupted(); name = os.path.splitext(os.path.basename(f))[0]
            _msg(unique_id, stage="Tokenizing", pct=int(100 * i / len(files)), detail=name)
            wav, sr = load_audio_file(f); wav = wav[:, :int(max_minutes * 60 * sr)]
            style = re.sub("[‘’]", "'", re.sub("[“”]", '"', " ".join((sidecar(f, ".txt") or default_style).replace("\\n", " ").split())))[:1500]; lyrics = normalize_lyrics(sidecar(f, ".lyrics.txt", ".lyrics") or default_lyrics)
            codec = tk.tokenize(wav, sr); abc = None; lat = None
            if store_latents:
                import torchaudio; vae = pipe["vae"]; vsr = getattr(vae, "audio_sample_rate", 48000); w = wav if wav.shape[0] == 2 else wav.expand(2, -1)
                if sr != vsr: w = torchaudio.functional.resample(w, sr, vsr)
                lat = vae.encode(w[None].movedim(1, -1))[0].T.to(torch.float16).cpu().numpy()                     # [T,64]
            if enc is not None:
                _msg(unique_id, stage="Transcribing score", pct=int(100 * i / len(files)), detail=name)
                abc = None; note = ""
                for how, kw, w_ in (("full", {"melody_only": False}, wav), ("melody-only", {"melody_only": True}, wav), ("first 4 min, full", {"melody_only": False}, wav[..., :sr * 240]), ("first 4 min, melody-only", {"melody_only": True}, wav[..., :sr * 240])):
                    try:
                        cand = enc.generate_abc(w_[None], sr, **kw)[0].strip()
                        if cand and cand.count("|") >= 4: abc = cand; note = "" if how == "full" else f" (score repaired: {how})"; break
                    except Exception as e: continue                            # repair chain instead of dropping the track (ai-toolkit does row-level repair; this is the coarse version)
                if note: repaired.append(name + note)
                if abc and auto_tempo_key:      # our captions always carried tempo + key; read them from the transcribed score header when the caption lacks them
                    q = re.search(r"^Q:\s*1/4\s*=\s*(\d+)", abc, re.M); k = re.search(r"^K:\s*([A-G][#b]?)(m|min|maj|dor|mix)?", abc, re.M)
                    if q and not re.search(r"\bBPM\b", style, re.I): style = f"{style}, {q.group(1)} BPM"
                    if k and not re.search(r"\bkey of\b", style, re.I): style = f"{style}, key of {k.group(1)} {'minor' if k.group(2) in ('m', 'min') else 'major' if k.group(2) in (None, 'maj') else k.group(2)}"
            items.append({"name": name, "style": style, "lyrics": lyrics, "abc": abc, "codec": codec, "latents": lat, "held": held(name, hold_out_percent)})
        tk.release(); path = os.path.join(_datasets_dir(), f"{dataset_name}.pt"); torch.save(items, path)
        rep = {"dataset": path, "songs": len(items), "held_out": sum(x["held"] for x in items), "with_scores": sum(1 for x in items if x["abc"]), "with_latents": sum(1 for x in items if x.get("latents") is not None), "minutes": round(sum(len(x["codec"]) for x in items) / 25 / 60, 1), "seconds_elapsed": round(time.time() - t0)}
        _msg(unique_id, stage="Done", pct=100, detail=f"{rep['songs']} songs, {rep['minutes']} min"); return ({"path": path, **rep}, json.dumps(rep, indent=1))

class FSAudioRegularizer:
    """The minted-song regularizer pack (YuE2's own generations + captions), coin-flipped against the artist during training so the LoRA keeps the base model's range."""
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"pack": (_fs_files(".pt") or ["(put a regularizer pack in models/fs_audio)"],)}}
    RETURN_TYPES = ("FS_AUDIO_REGULARIZER",); RETURN_NAMES = ("regularizer",); FUNCTION = "load"; CATEGORY = TCAT
    def load(self, pack):
        p = os.path.join(assets_dir(), pack)
        if not os.path.exists(p): raise ValueError(f"Regularizer pack not found: {p}")
        return ({"path": p},)

class FSAudioArtistTrainer:
    """Trains a whole artist in one run on ComfyUI's own YuE2: the planner LoRA (what they write) and the decoder LoRA (how they sound) in one
    loop, exported as ONE file that a single FS_Audio Lora Loader applies to both halves. Recipe: whole-song planner CE from the song's start with a
    KL(base||lora) trust region, artist/regularizer mixed batches, score-first half the time, decoder flow-matching on the artist's latents conditioned
    on the live planner, EMA weights, dense checkpoint ladder, held-out eval for both halves."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pipe": ("FS_AUDIO_PIPE",), "dataset": ("FS_AUDIO_DATASET",), "lora_name": ("STRING", {"default": "my_artist"}),
                             "steps": ("INT", {"default": 600, "min": 50, "max": 20000, "step": 50, "tooltip": "Planner optimizer steps. Rule of thumb: ~10 passes over the artist songs (steps x batch_songs x artist_fraction / songs); more memorizes."}),
                             "decoder_steps": ("INT", {"default": 1500, "min": 50, "max": 20000, "step": 50, "tooltip": "Decoder flow steps, spread evenly across the planner steps. Converges in ~1000 on a few dozen songs."}),
                             "rank_planner": ("INT", {"default": 64, "min": 4, "max": 256, "step": 4}), "rank_decoder": ("INT", {"default": 32, "min": 4, "max": 128, "step": 4}),
                             "planner_lr": ("FLOAT", {"default": 3e-5, "min": 1e-6, "max": 1e-2, "step": 1e-6}), "decoder_lr": ("FLOAT", {"default": 4e-5, "min": 1e-6, "max": 1e-3, "step": 1e-6}), "io_lr": ("FLOAT", {"default": 2e-5, "min": 1e-6, "max": 1e-3, "step": 1e-6, "tooltip": "For the decoder's full vae2llm / llm2vae layers."}),
                             "artist_fraction": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 1.0, "step": 0.05, "tooltip": "Coin flip per song: artist song vs regularizer song. 1.0 = artist only (memorizes fast)."}),
                             "batch_songs": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1, "tooltip": "Whole songs per planner step, gradients accumulated; artist and regularizer songs share one update (mixed batch)."}),
                             "kl_weight": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 5.0, "step": 0.01, "tooltip": "Trust region: KL(base || lora) on the planner's next-token distributions, base = LoRA switched off. Keeps the planner from drifting; 0 = off."}),
                             "score_first_fraction": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Fraction of planner songs with the transcribed score in front of the music (needs transcribe_scores). The planner always also learns to write the score."}),
                             "end_token_weight": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 50.0, "step": 1.0, "tooltip": "Up-weights the song-end token; ~20 for instrumental / long-song data that runs to the cap."}),
                             "max_tokens": ("INT", {"default": 8192, "min": 2048, "max": 16384, "step": 512, "tooltip": "Whole-song planner context. Songs that would not contain an ending are dropped from the planner set."}),
                             "window_seconds": ("FLOAT", {"default": 30.0, "min": 10.0, "max": 60.0, "step": 5.0, "tooltip": "Decoder training window."}),
                             "ema_decay": ("FLOAT", {"default": 0.99, "min": 0.0, "max": 0.9999, "step": 0.0001, "tooltip": "EMA of both LoRAs; exported checkpoints are the EMA weights. 0 = off."}),
                             "eval_every": ("INT", {"default": 50, "min": 25, "max": 1000, "step": 25}), "checkpoint_from": ("INT", {"default": 200, "min": 0, "max": 20000, "step": 50}), "checkpoint_every": ("INT", {"default": 100, "min": 50, "max": 5000, "step": 50}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})},
                "optional": {"regularizer": ("FS_AUDIO_REGULARIZER",), "strength_model": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}), "strength_clip": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05})},
                "hidden": {"unique_id": "UNIQUE_ID"}}
    RETURN_TYPES = ("FS_AUDIO_LORAS", "STRING"); RETURN_NAMES = ("loras", "report"); FUNCTION = "train"; CATEGORY = TCAT
    def train(self, pipe, dataset, lora_name, steps, decoder_steps, rank_planner, rank_decoder, planner_lr, decoder_lr, io_lr, artist_fraction, batch_songs, kl_weight, score_first_fraction, end_token_weight, max_tokens, window_seconds, ema_decay, eval_every, checkpoint_from, checkpoint_every, seed, regularizer=None, strength_model=1.0, strength_clip=1.0, unique_id=None):
        artist = torch.load(dataset["path"], weights_only=False); reg = load_regularizer(regularizer["path"]) if regularizer else None
        out_dir = folder_paths.get_folder_paths("loras")[0]; status = lambda **k: _msg(unique_id, **k)
        cfg = {"name": lora_name, "out_dir": out_dir, "rank_planner": rank_planner, "rank_decoder": rank_decoder, "steps": steps, "decoder_steps": decoder_steps, "lr_planner": planner_lr, "lr_decoder": decoder_lr, "lr_io": io_lr,
               "artist_fraction": artist_fraction, "batch_songs": batch_songs, "kl_weight": kl_weight, "score_first_fraction": score_first_fraction, "end_weight": end_token_weight, "max_tokens": max_tokens, "window_frames": int(window_seconds * 25),
               "ema_decay": ema_decay, "eval_every": eval_every, "ckpt_from": checkpoint_from, "ckpt_every": checkpoint_every, "seed": seed}
        # ComfyUI executes nodes under torch.inference_mode(); weights loaded there are inference tensors and cannot enter an autograd graph, so the
        # trainer loads its own copy of the checkpoint (planner + decoder) with inference mode OFF. The pipe's copy is untouched.
        import comfy.sd
        with torch.inference_mode(False), torch.enable_grad():
            status(stage="Loading YuE2 for training", detail=os.path.basename(pipe["ckpt_path"]))
            model, clip, _v, _cv = comfy.sd.load_checkpoint_guess_config(pipe["ckpt_path"], output_vae=False, output_clip=True, output_model=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
            try: res = fs_artist.train(model, clip, artist, reg, cfg, status)
            finally:
                del model, clip; comfy.model_management.soft_empty_cache()
        final = res["final"] or os.path.join(out_dir, f"{lora_name}_best.safetensors"); status(stage="Done", detail=os.path.basename(final))
        rep = {"lora": final, "best_artist_loss": round(res["best_artist_loss"], 4), "checkpoints": [os.path.basename(c) for c in res["checkpoints"]], "artist_songs": res["artist_train"], "regularizer_songs": res["regularizer_train"], "decoder_songs": res["decoder_songs"], "score_first": res["score_first"], "config": cfg, "log": res["log"]}
        return ([{"name": os.path.basename(final), "path": final, "model": strength_model, "clip": strength_clip}], json.dumps(rep, indent=1))

NODE_CLASS_MAPPINGS = {"FSAudioTrainAssets": FSAudioTrainAssets, "FSAudioDatasetBuilder": FSAudioDatasetBuilder, "FSAudioRegularizer": FSAudioRegularizer, "FSAudioArtistTrainer": FSAudioArtistTrainer}
NODE_DISPLAY_NAME_MAPPINGS = {"FSAudioTrainAssets": "⬇ FS_Audio Training Assets", "FSAudioDatasetBuilder": "📦 FS_Audio Dataset Builder", "FSAudioRegularizer": "🧪 FS_Audio Regularizer", "FSAudioArtistTrainer": "🏋 FS_Audio Artist Trainer"}
