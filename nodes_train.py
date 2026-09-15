"""FS_Audio Suite — training nodes (additive; the inference nodes are untouched)."""
import os, json, time, glob, numpy as np, torch
import folder_paths, comfy.model_management
from server import PromptServer
from comfy_extras.nodes_audio import load as load_audio_file
from .nodes import _download, CAT
from .fs_train.tokenizer import AudioTokenizer, assets_dir
from .fs_train.data import list_audio, sidecar, normalize_lyrics, held, load_regularizer
from .fs_train import trainer as fs_trainer
TCAT = CAT + "/Training"
def _msg(node_id, **kw):
    try: PromptServer.instance.send_sync("fsaudio.train", {"node": node_id, **kw})
    except Exception: pass
def _datasets_dir(): d = os.path.join(folder_paths.get_output_directory(), "fs_audio", "datasets"); os.makedirs(d, exist_ok=True); return d
def _fs_files(ext): return sorted(f for f in os.listdir(assets_dir()) if f.endswith(ext))

class FSAudioTrainAssets:
    """Standalone: downloads the audio->token head (and other training assets) into models/fs_audio."""
    ASSETS = {"tokenizer_head_joint_v4 (audio -> YuE2 tokens)": ("https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4/resolve/main/tokenizer_head_joint_v4.safetensors", "fs_audio")}
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"asset": (list(cls.ASSETS.keys()),)}, "hidden": {"unique_id": "UNIQUE_ID"}}
    RETURN_TYPES = (); FUNCTION = "download"; OUTPUT_NODE = True; CATEGORY = TCAT
    @classmethod
    def IS_CHANGED(cls, asset, unique_id=None): return float("nan")
    def download(self, asset, unique_id=None):
        url, folder = self.ASSETS[asset]; dest = os.path.join(assets_dir(), url.rsplit("/", 1)[-1])
        if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000: return {"ui": {"fs_download": [f"{os.path.basename(dest)} — already in models/fs_audio"]}}
        _download(url, dest, unique_id); return {"ui": {"fs_download": [f"{os.path.basename(dest)} -> models/fs_audio"]}}

class FSAudioDatasetBuilder:
    """A folder of songs -> training dataset: real-audio tokens (public MERT + our head), captions/lyrics from sidecars, optional scores from the pipe's melody transcriber."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pipe": ("FS_AUDIO_PIPE",), "audio_folder": ("STRING", {"default": "train/my_artist", "tooltip": "Folder of songs. Relative paths resolve under ComfyUI/input. Per-song sidecars: <song>.txt = style caption, <song>.lyrics.txt = lyrics."}),
                             "dataset_name": ("STRING", {"default": "my_artist"}), "tokenizer_head": (_fs_files(".safetensors") or ["(run FS_Audio Training Assets first)"],),
                             "default_style": ("STRING", {"multiline": True, "default": "", "tooltip": "Used for songs without a .txt caption sidecar."}),
                             "default_lyrics": ("STRING", {"multiline": True, "default": "[instrumental]", "tooltip": "Used for songs without a .lyrics.txt sidecar."}),
                             "transcribe_scores": ("BOOLEAN", {"default": True, "tooltip": "Transcribe each song to a chord-annotated ABC score with the pipe's melody transcriber, enabling the score-first training recipe."}),
                             "hold_out_percent": ("INT", {"default": 5, "min": 0, "max": 50}), "max_minutes": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 16.0, "step": 0.5})},
                "hidden": {"unique_id": "UNIQUE_ID"}}
    RETURN_TYPES = ("FS_AUDIO_DATASET", "STRING"); RETURN_NAMES = ("dataset", "report"); FUNCTION = "build"; CATEGORY = TCAT
    def build(self, pipe, audio_folder, dataset_name, tokenizer_head, default_style, default_lyrics, transcribe_scores, hold_out_percent, max_minutes, unique_id=None):
        folder = audio_folder if os.path.isabs(audio_folder) else os.path.join(folder_paths.get_input_directory(), audio_folder); files = list_audio(folder)
        if not files: raise ValueError(f"No audio files in {folder}")
        head = os.path.join(assets_dir(), tokenizer_head)
        if not os.path.exists(head): raise ValueError("Tokenizer head missing: run the FS_Audio Training Assets node first.")
        enc = pipe.get("encoder") if transcribe_scores else None
        if transcribe_scores and enc is None: raise ValueError("transcribe_scores needs a melody_transcriber in the Model Loader (sheetsage2).")
        _msg(unique_id, stage="Loading tokenizer", pct=0); tk = AudioTokenizer(head); items = []; t0 = time.time()
        for i, f in enumerate(files):
            comfy.model_management.throw_exception_if_processing_interrupted(); name = os.path.splitext(os.path.basename(f))[0]
            _msg(unique_id, stage="Tokenizing", pct=int(100 * i / len(files)), detail=name)
            wav, sr = load_audio_file(f); wav = wav[:, :int(max_minutes * 60 * sr)]
            style = " ".join((sidecar(f, ".txt") or default_style).replace("\\n", " ").split())[:1500]; lyrics = normalize_lyrics(sidecar(f, ".lyrics.txt", ".lyrics") or default_lyrics)
            codec = tk.tokenize(wav, sr); abc = None
            if enc is not None:
                _msg(unique_id, stage="Transcribing score", pct=int(100 * i / len(files)), detail=name)
                try: abc = enc.generate_abc(wav[None], sr, melody_only=False)[0].strip()
                except Exception as e: abc = None
            items.append({"name": name, "style": style, "lyrics": lyrics, "abc": abc, "codec": codec, "held": held(name, hold_out_percent)})
        tk.release(); path = os.path.join(_datasets_dir(), f"{dataset_name}.pt"); torch.save(items, path)
        rep = {"dataset": path, "songs": len(items), "held_out": sum(x["held"] for x in items), "with_scores": sum(1 for x in items if x["abc"]), "minutes": round(sum(len(x["codec"]) for x in items) / 25 / 60, 1), "seconds_elapsed": round(time.time() - t0)}
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

class FSAudioLoraTrainer:
    """Trains a planner (AR) LoRA on ComfyUI's own YuE2 and hands it back as a loras chain entry for the Model Loader."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pipe": ("FS_AUDIO_PIPE",), "dataset": ("FS_AUDIO_DATASET",), "lora_name": ("STRING", {"default": "my_artist_lora"}),
                             "rank": ("INT", {"default": 64, "min": 4, "max": 256, "step": 4}), "steps": ("INT", {"default": 1600, "min": 50, "max": 20000, "step": 50}),
                             "learning_rate": ("FLOAT", {"default": 1e-4, "min": 1e-6, "max": 1e-2, "step": 1e-5}),
                             "artist_fraction": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 1.0, "step": 0.05, "tooltip": "Coin flip: chance a step trains on the artist instead of the regularizer. 1.0 = artist only (memorizes fast)."}),
                             "score_first_fraction": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Fraction of steps that put the transcribed score in front of the music (needs transcribe_scores in the dataset). The planner always learns to write scores."}),
                             "end_token_weight": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 50.0, "step": 1.0, "tooltip": "Up-weights the song-end token. Use ~20 for instrumental / long-song data that tends to run to the cap."}),
                             "max_tokens": ("INT", {"default": 8192, "min": 2048, "max": 16384, "step": 512, "tooltip": "Whole-song context. 8192 fits 24 GB; 12288 for 40+ GB. Songs longer than this are dropped so every example contains an ending."}),
                             "eval_every": ("INT", {"default": 100, "min": 25, "max": 1000, "step": 25}), "checkpoint_from": ("INT", {"default": 600, "min": 0, "max": 20000, "step": 50}), "checkpoint_every": ("INT", {"default": 200, "min": 50, "max": 5000, "step": 50}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})},
                "optional": {"regularizer": ("FS_AUDIO_REGULARIZER",), "strength_clip": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05})},
                "hidden": {"unique_id": "UNIQUE_ID"}}
    RETURN_TYPES = ("FS_AUDIO_LORAS", "STRING"); RETURN_NAMES = ("loras", "report"); FUNCTION = "train"; CATEGORY = TCAT
    def train(self, pipe, dataset, lora_name, rank, steps, learning_rate, artist_fraction, score_first_fraction, end_token_weight, max_tokens, eval_every, checkpoint_from, checkpoint_every, seed, regularizer=None, strength_clip=1.0, unique_id=None):
        artist = torch.load(dataset["path"], weights_only=False); reg = load_regularizer(regularizer["path"]) if regularizer else None
        out_dir = folder_paths.get_folder_paths("loras")[0]; status = lambda **k: _msg(unique_id, **k)
        cfg = {"rank": rank, "steps": steps, "lr": learning_rate, "artist_fraction": artist_fraction, "score_first_fraction": score_first_fraction, "end_weight": end_token_weight, "max_tokens": max_tokens, "eval_every": eval_every, "ckpt_from": checkpoint_from, "ckpt_every": checkpoint_every, "seed": seed, "out_dir": out_dir, "name": lora_name}
        # ComfyUI executes nodes under torch.inference_mode(); weights loaded there are inference tensors and cannot enter an autograd graph.
        # So the trainer loads its own text-encoder copy with inference mode OFF (the pipe's copy is untouched) and trains with grad enabled.
        import comfy.sd
        with torch.inference_mode(False), torch.enable_grad():
            status(stage="Loading planner for training", detail=os.path.basename(pipe["ckpt_path"]))
            _m, clip, _v, _cv = comfy.sd.load_checkpoint_guess_config(pipe["ckpt_path"], output_vae=False, output_clip=True, output_model=False, embedding_directory=folder_paths.get_folder_paths("embeddings"))
            try: res = fs_trainer.train(clip, artist, reg, cfg, status)
            finally:
                del clip; comfy.model_management.soft_empty_cache()
        final = res["final"] or os.path.join(out_dir, f"{lora_name}_best.safetensors"); status(stage="Done", detail=os.path.basename(final))
        rep = {"lora": final, "best_artist_loss": round(res["best_artist_loss"], 4), "checkpoints": [os.path.basename(c) for c in res["checkpoints"]], "artist_songs": res["artist_train"], "regularizer_songs": res["regularizer_train"], "score_first": res["score_first"], "log": res["log"]}
        return ([{"name": os.path.basename(final), "path": final, "model": 0.0, "clip": strength_clip}], json.dumps(rep, indent=1))

NODE_CLASS_MAPPINGS = {"FSAudioTrainAssets": FSAudioTrainAssets, "FSAudioDatasetBuilder": FSAudioDatasetBuilder, "FSAudioRegularizer": FSAudioRegularizer, "FSAudioLoraTrainer": FSAudioLoraTrainer}
NODE_DISPLAY_NAME_MAPPINGS = {"FSAudioTrainAssets": "⬇ FS_Audio Training Assets", "FSAudioDatasetBuilder": "📦 FS_Audio Dataset Builder", "FSAudioRegularizer": "🧪 FS_Audio Regularizer", "FSAudioLoraTrainer": "🏋 FS_Audio LoRA Trainer"}
