"""Datasets for the planner LoRA. Item = {name, style, lyrics, abc (str|None), codec (int32 tokens), held (bool)}.
Sequence layouts (YuE2 protocol, built with ComfyUI's own YuE2 tokenizer):
  score-free : [EOD] text(off)  [ABC_START] [ABC_END] [MUSIC_START] codec [MUSIC_END]         loss on codec
  score-first: [EOD] text(full) [ABC_START] abc [ABC_END] [MUSIC_START] codec [MUSIC_END]     loss on abc + codec"""
import os, re, json, glob, hashlib, numpy as np, torch
ABC_START, ABC_END, MUSIC_START, MUSIC_END, CODEC_OFFSET = 151847, 151848, 151851, 151852, 151853
AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus")
TAGS = {"intro", "verse", "pre-chorus", "chorus", "bridge", "outro"}
def normalize_lyrics(text):
    """bare section tags only; drop production notes in brackets; literal \\n -> newline; curly quotes -> straight"""
    t = (text or "").replace("\\n", "\n").replace("\r", ""); t = re.sub("[‘’]", "'", t); t = re.sub("[“”]", '"', t); out = []
    for ln in t.split("\n"):
        m = re.match(r"^\s*\[([^\]]+)\]\s*$", ln)
        if m:
            low = m.group(1).lower(); tag = next((k for k in ("pre-chorus", "chorus", "verse", "bridge", "intro", "outro") if k in low), None)
            if low.strip() == "instrumental": out.append("[instrumental]")
            elif tag: out.append(f"[{tag}]")
            continue
        out.append(ln.rstrip())
    s = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip(); return s if s else "[instrumental]"
def sidecar(path, *exts):
    base = os.path.splitext(path)[0]
    for e in exts:
        if os.path.exists(base + e): return open(base + e, encoding="utf-8", errors="ignore").read().strip()
    return None
def list_audio(folder): return sorted(p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith(AUDIO_EXT))
def held(name, pct): return (int(hashlib.md5(name.encode()).hexdigest(), 16) % 100) < pct
def build_sequences(item, tok, cot_layout):
    """-> (ids list, loss_start) using ComfyUI's YuE2 tokenizer (tok = clip.tokenizer)."""
    if cot_layout == "full" and item.get("abc"):
        t = tok.tokenize_with_weights(item["style"], lyrics=item["lyrics"], cot="full", abc=item["abc"]); pre = t["prefix"] + t["abc_ids"] + [ABC_END, MUSIC_START]; loss_start = len(t["prefix"])       # predict abc[0] onward
    else:
        t = tok.tokenize_with_weights(item["style"], lyrics=item["lyrics"], cot="off"); pre = t["prefix"] + [ABC_END, MUSIC_START]; loss_start = len(pre)
    return pre, loss_start
def load_regularizer(path):
    """our minted pack: list of {name, src, style, lyrics, codec, (abc)} -> items"""
    pack = torch.load(path, weights_only=False); items = []
    for x in pack:
        items.append({"name": x["name"], "style": x["style"], "lyrics": x["lyrics"], "abc": x.get("abc"), "codec": np.asarray(x["codec"], dtype=np.int32), "held": x.get("src") == "minted_val"})
    return items
