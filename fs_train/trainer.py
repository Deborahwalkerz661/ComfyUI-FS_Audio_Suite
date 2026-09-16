"""Planner (AR) LoRA training inside ComfyUI, on ComfyUI's own YuE2 text encoder.
Tricks (all switchable): whole-song sequences up to max_tokens, artist/regularizer coin flip, END-token up-weighting, score-first layout for a
fraction of steps (Ostris recipe: the planner always learns to write the score, music sees it half the time), cosine schedule + warmup,
held-out eval, ladder checkpoints. LoRA sits on the fused qkv_proj / o_proj / gate_up_proj / down_proj so the file loads straight into ComfyUI."""
import os, math, time, random, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from safetensors.torch import save_file
import comfy.model_management
from .data import build_sequences, ABC_START, MUSIC_END, CODEC_OFFSET
class LoRAHook:
    """Adds (x @ A^T) @ B^T to a frozen comfy Linear via a forward hook (comfy reads .weight directly in places, so no module wrapping)."""
    def __init__(s, base, r, dev):
        s.A = nn.Parameter(torch.randn(r, base.in_features, device=dev) * (1 / math.sqrt(base.in_features))); s.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
        s.h = base.register_forward_hook(lambda mod, inp, out: out + ((inp[0].float() @ s.A.T) @ s.B.T).to(out.dtype))
    def remove(s): s.h.remove()
TARGETS = (("self_attn", "qkv_proj"), ("self_attn", "o_proj"), ("mlp", "gate_up_proj"), ("mlp", "down_proj"))
class PlannerTrainer:
    def __init__(self, clip, rank, dev="cuda"):
        self.clip = clip; self.dev = dev; comfy.model_management.load_models_gpu([clip.patcher], force_full_load=True)
        self.L2 = clip.cond_stage_model.model; self.L2.requires_grad_(False); cfg = self.L2.config
        self.HD, self.NH, self.NKV, self.THETA = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.rope_theta
        self.hooks, self.params, self.saved_act = [], [], []
        for layer in self.L2.layers:
            self.saved_act.append(layer.mlp.merged_input_act); layer.mlp.merged_input_act = None          # plain module-call MLP path so hooks fire on down_proj
            for blk, proj in TARGETS: hk = LoRAHook(getattr(getattr(layer, blk), proj), rank, dev); self.hooks.append(hk); self.params += [hk.A, hk.B]
        self.rank = rank; self.tok = clip.tokenizer
    def close(self):
        for hk in self.hooks: hk.remove()
        for layer, act in zip(self.L2.layers, self.saved_act): layer.mlp.merged_input_act = act
        self.hooks = []; torch.cuda.empty_cache()
    # ---- functional forward (comfy's block uses an out= add and a compiled rope kernel, neither differentiable)
    def _rope(self, S):
        inv = 1.0 / (self.THETA ** (torch.arange(0, self.HD, 2, device=self.dev).float() / self.HD)); f = torch.outer(torch.arange(S, device=self.dev).float(), inv); emb = torch.cat([f, f], -1); return emb.cos()[None, None], emb.sin()[None, None]
    @staticmethod
    def _rot(x): h = x.shape[-1] // 2; return torch.cat([-x[..., h:], x[..., :h]], -1)
    def _attn(self, sa, x, cos, sin):
        B, S, _ = x.shape; q, k, v = sa.qkv_proj(x).split((sa.inner_size, sa.kv_size, sa.kv_size), -1)
        q = q.view(B, S, self.NH, self.HD).transpose(1, 2); k = k.view(B, S, self.NKV, self.HD).transpose(1, 2); v = v.view(B, S, self.NKV, self.HD).transpose(1, 2)
        if sa.q_norm is not None: q = sa.q_norm(q)
        if sa.k_norm is not None: k = sa.k_norm(k)
        q = (q * cos + self._rot(q) * sin).to(v.dtype); k = (k * cos + self._rot(k) * sin).to(v.dtype)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True); return sa.o_proj(o.transpose(1, 2).reshape(B, S, sa.inner_size))
    def _blk(self, layer, x, cos, sin):
        x = x + self._attn(layer.self_attn, layer.input_layernorm(x), cos, sin); return x + layer.mlp(layer.post_attention_layernorm(x))
    def hidden(self, ids, grad=True):
        x = self.L2.embed_tokens(ids, out_dtype=torch.bfloat16); cos, sin = self._rope(ids.shape[1])
        for layer in self.L2.layers: x = checkpoint(self._blk, layer, x, cos, sin, use_reentrant=False) if grad else self._blk(layer, x, cos, sin)
        return self.L2.norm(x)
    def loss(self, ids, loss_start, end_w=1.0, grad=True):
        h = self.hidden(ids, grad)[0, loss_start - 1:-1]; tgt = ids[0, loss_start:]; w = torch.ones_like(tgt, dtype=torch.float32); w[tgt == MUSIC_END] = end_w; tot = 0.
        for s in range(0, h.shape[0], 1024): tot = tot + (F.cross_entropy(self.L2.lm_head(h[s:s + 1024]).float(), tgt[s:s + 1024], reduction="none") * w[s:s + 1024]).sum()
        return tot / w.sum()
    # ---- sequences
    def seq(self, item, layout, max_tokens):
        pre, ls = build_sequences(item, self.tok, layout); cod = [int(c) + CODEC_OFFSET for c in item["codec"]]; room = max_tokens - len(pre) - 1
        body = cod[:room] + ([MUSIC_END] if len(cod) <= room else []); return torch.tensor([pre + body], device=self.dev), ls
    def fits(self, item, layout, max_tokens):
        pre, _ = build_sequences(item, self.tok, layout); return len(pre) + len(item["codec"]) + 1 <= max_tokens
    # ---- export in ComfyUI's native LoRA layout (AR = text_encoders.*)
    def export(self, path, meta):
        out = {}; i = 0
        for l in range(len(self.L2.layers)):
            for blk, proj in TARGETS: hk = self.hooks[i]; i += 1; out[f"text_encoders.model.layers.{l}.{blk}.{proj}.lora_down.weight"] = hk.A.detach().to(torch.bfloat16).cpu().contiguous(); out[f"text_encoders.model.layers.{l}.{blk}.{proj}.lora_up.weight"] = hk.B.detach().to(torch.bfloat16).cpu().contiguous()
        save_file(out, path, metadata={"format": "pt", "fs_audio": "planner LoRA", "rank": str(self.rank), "scale": "1.0 (no alpha)", **{k: str(v) for k, v in meta.items()}})
def train(clip, artist, regularizer, cfg, status=lambda **k: None):
    """cfg: rank steps lr artist_fraction batch_songs score_first_fraction end_weight max_tokens eval_every ckpt_from ckpt_every seed out_dir name warmup"""
    torch.backends.cuda.matmul.allow_tf32 = True; random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    tr = PlannerTrainer(clip, cfg["rank"])
    try:
        mt = cfg["max_tokens"]; require_end = cfg.get("require_end", True)
        a_train = [x for x in artist if not x["held"] and (tr.fits(x, "off", mt) or not require_end)]; a_val = [x for x in artist if x["held"]] or a_train[:4]
        r_train = [x for x in (regularizer or []) if not x["held"] and (tr.fits(x, "off", mt) or not require_end)]; r_val = [x for x in (regularizer or []) if x["held"]][:6]
        has_abc = any(x.get("abc") for x in a_train); sf = cfg["score_first_fraction"] if has_abc else 0.0
        status(stage="Training", detail=f"{len(a_train)} artist / {len(r_train)} regularizer songs, rank {cfg['rank']}, {cfg['steps']} steps x {cfg.get('batch_songs', 1)} songs" + (f", score-first {sf:.0%}" if sf else ""))
        opt = torch.optim.AdamW(tr.params, lr=cfg["lr"], weight_decay=0.0, betas=(0.9, 0.95)); STEPS = cfg["steps"]; WARM = cfg.get("warmup", 50)
        sched = lambda st: cfg["lr"] * min(1, st / WARM) * (0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * min(st, STEPS) / STEPS)))
        @torch.no_grad()
        def evaluate():
            r = {}
            for tag, items in (("artist", a_val[:6]), ("regularizer", r_val)):
                if not items: continue
                tot = 0
                for it in items: ids, ls = tr.seq(it, "off", mt); tot += tr.loss(ids, ls, grad=False).item()
                r[tag] = tot / len(items)
            return r
        log = []; e = evaluate(); log.append({"step": 0, **e}); status(step=0, evals=e, total=STEPS); best = e.get("artist", 9e9); t0 = time.time(); os.makedirs(cfg["out_dir"], exist_ok=True); ckpts = []
        for st in range(1, STEPS + 1):
            comfy.model_management.throw_exception_if_processing_interrupted()
            for g in opt.param_groups: g["lr"] = sched(st)
            K = max(1, int(cfg.get("batch_songs", 1))); loss_sum = 0.0
            for _k in range(K):                                                                              # mixed batch: K whole songs per optimizer step, each coin-flipped artist/regularizer
                it = random.choice(a_train) if (not r_train or random.random() < cfg["artist_fraction"]) else random.choice(r_train)
                layout = "full" if (it.get("abc") and random.random() < sf) else "off"
                ids, ls = tr.seq(it, layout, mt); loss = tr.loss(ids, ls, cfg["end_weight"]); (loss / K).backward(); loss_sum += float(loss) / K
                if layout == "off" and sf and it.get("abc"):                                                # always learn to write the score
                    pre, _ = build_sequences(it, tr.tok, "full"); aids = torch.tensor([pre[:-1]], device=tr.dev)    # text + [ABC_START] abc [ABC_END]: score-writing only
                    (tr.loss(aids, pre.index(ABC_START) + 1) * 0.5 / K).backward()
            loss = torch.tensor(loss_sum)
            torch.nn.utils.clip_grad_norm_(tr.params, 1.0); opt.step(); opt.zero_grad(set_to_none=True)
            if st % 5 == 0 or st <= 3: status(step=st, loss=float(loss), total=STEPS, eta=(time.time() - t0) / st * (STEPS - st), seq=int(ids.shape[1]))
            if st % cfg["eval_every"] == 0 or st == STEPS:
                e = evaluate(); log.append({"step": st, **e}); status(step=st, evals=e, total=STEPS)
                if e.get("artist", 9e9) < best: best = e["artist"]; tr.export(os.path.join(cfg["out_dir"], f"{cfg['name']}_best.safetensors"), {**e, "step": st, "config": json.dumps(cfg)})
            if st >= cfg["ckpt_from"] and (st - cfg["ckpt_from"]) % cfg["ckpt_every"] == 0 or st == STEPS:
                p = os.path.join(cfg["out_dir"], f"{cfg['name']}_step{st}.safetensors"); tr.export(p, {"step": st, "config": json.dumps(cfg)}); ckpts.append(p)
        json.dump(log, open(os.path.join(cfg["out_dir"], f"{cfg['name']}_log.json"), "w"), indent=1)
        return {"best_artist_loss": best, "checkpoints": ckpts, "log": log, "final": ckpts[-1] if ckpts else None, "artist_train": len(a_train), "regularizer_train": len(r_train), "score_first": sf}
    finally:
        tr.close()
