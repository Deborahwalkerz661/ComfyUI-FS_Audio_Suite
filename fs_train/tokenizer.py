"""Audio -> YuE2 semantic tokens: public MERT-v2-FullSong (layer 20, 25 Hz, 30 s chunks, per-track instance norm) -> our tokenizer head."""
import os, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchaudio
import folder_paths
from safetensors.torch import load_file
VOCAB, WIN, D, L, H = 32768, 512, 512, 8, 8
def assets_dir():
    d = os.path.join(folder_paths.models_dir, "fs_audio"); os.makedirs(d, exist_ok=True); return d
class Tok(nn.Module):
    def __init__(s, din=1024):
        super().__init__(); s.inp = nn.Linear(din, D); s.pos = nn.Parameter(torch.zeros(1, WIN, D))
        layer = nn.TransformerEncoderLayer(D, H, 4 * D, dropout=0.1, batch_first=True, norm_first=True, activation="gelu"); s.enc = nn.TransformerEncoder(layer, L); s.norm = nn.LayerNorm(D); s.head = nn.Linear(D, VOCAB)
    def forward(s, x): return s.head(s.norm(s.enc(s.inp(x) + s.pos[:, :x.shape[1]])))
class AudioTokenizer:
    """Loads MERT (through transformers, cached under models/fs_audio/hf) and the head; tokenize(waveform, sr) -> int32 tokens @25 Hz."""
    def __init__(self, head_file, device="cuda"):
        from transformers import AutoModel, AutoFeatureExtractor
        self.dev = device; cache = os.path.join(assets_dir(), "hf")
        self.proc = AutoFeatureExtractor.from_pretrained("m-a-p/MERT-v2-FullSong", trust_remote_code=True, cache_dir=cache)
        self.mert = AutoModel.from_pretrained("m-a-p/MERT-v2-FullSong", trust_remote_code=True, cache_dir=cache).to(device).eval()
        self.head = Tok().to(device).eval(); sd = load_file(head_file); self.head.load_state_dict(sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd)
    @torch.no_grad()
    def features(self, wav, sr):
        m24 = torchaudio.functional.resample(wav.float().mean(0), sr, 24000).numpy(); CH = 24000 * 30
        chunks = [m24[s:s + CH] for s in range(0, len(m24), CH)]; chunks = [c for c in chunks if len(c) >= 24000]; full = [c for c in chunks if len(c) == CH]; tail = [c for c in chunks if len(c) < CH]; feats = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for group in ([full] if full else []) + [[c] for c in tail]:
                inp = {k: v.to(self.dev) for k, v in self.proc(group, sampling_rate=24000, return_tensors="pt").items()}; feats.append(self.mert(**inp, output_hidden_states=True).hidden_states[20].reshape(-1, 1024))
        Hh = torch.cat(feats, 0).float(); T25 = int(round(len(m24) / 24000 * 25)); return F.interpolate(Hh.T[None], size=T25, mode="linear", align_corners=False)[0].T.cpu().numpy()
    @torch.no_grad()
    def tokenize(self, wav, sr):
        x = self.features(wav, sr); x = (x - x.mean(0)) / (x.std(0) + 1e-5); T = len(x); out = np.zeros(T, dtype=np.int64); starts = list(range(0, max(1, T - WIN + 1), WIN // 2))
        if starts[-1] + WIN < T: starts.append(max(0, T - WIN))
        for s0 in starts:
            xw = x[s0:s0 + WIN]; n = len(xw)
            if n < WIN: xw = np.pad(xw, ((0, WIN - n), (0, 0)))
            with torch.autocast("cuda", dtype=torch.bfloat16): pred = self.head(torch.tensor(xw[None], dtype=torch.float32, device=self.dev))[0, :n].float().argmax(-1).cpu().numpy()
            lo = s0 + (0 if s0 == 0 else WIN // 4); hi = s0 + n - (0 if s0 + n >= T else WIN // 4); out[lo:hi] = pred[lo - s0:hi - s0]
        return out.astype(np.int32)
    def release(self):
        del self.mert, self.head; torch.cuda.empty_cache()
