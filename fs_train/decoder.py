"""Decoder (NAR) adapter training inside ComfyUI: the step that made off-genre real audio render faithfully for us.
Trains a LoRA on the decoder's fused projections plus the full vae2llm / llm2vae layers with the flow-matching loss on the artist's own VAE
latents, conditioned on the artist's tokens through ComfyUI's own AR prefix cache. Exports diffusion_model.* LoRA + .diff for the MODEL slot."""
import os, math, time, random, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from safetensors.torch import save_file
import comfy.model_management
from .trainer import LoRAHook, TARGETS
from .data import build_sequences, MUSIC_END
class DecoderTrainer:
    def __init__(self, model, clip, rank, dev="cuda"):
        clip.load_model(clip.tokenize("x", lyrics="[instrumental]", cot="off")); comfy.model_management.load_models_gpu([model, clip.patcher], force_full_load=True)   # sets the text encoder's execution_device (YuE2's memory estimate needs real tokens)
        self.dm = model.model.diffusion_model; self.ms = model.model.model_sampling; self.te = clip.cond_stage_model; self.tok = clip.tokenizer; self.dev = dev; self.rank = rank
        self.te.execution_device = torch.device(dev); self.te.model.to(dev)     # the combined load can leave the text encoder's device tag on CPU; the KV cache is allocated on this device
        self.dm.requires_grad_(False); cfg = self.dm.config; self.HD, self.NH, self.NKV, self.THETA = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.rope_theta; self.L = cfg.num_hidden_layers
        self.hooks, self.params, self.saved_act = [], [], []
        for layer in self.dm.model.layers:
            self.saved_act.append(layer.mlp.merged_input_act); layer.mlp.merged_input_act = None
            for blk, proj in TARGETS: hk = LoRAHook(getattr(getattr(layer, blk), proj), rank, dev); self.hooks.append(hk); self.params += [hk.A, hk.B]
        self.io_orig = {n: getattr(self.dm, n).weight.detach().clone() for n in ("vae2llm", "llm2vae")}; self.io_orig_b = {n: getattr(self.dm, n).bias.detach().clone() for n in ("vae2llm", "llm2vae")}
        self.io_params = []
        for n in ("vae2llm", "llm2vae"):
            lin = getattr(self.dm, n); lin.weight.data = lin.weight.data.float(); lin.bias.data = lin.bias.data.float(); lin.weight.requires_grad_(True); lin.bias.requires_grad_(True); self.io_params += [lin.weight, lin.bias]
    def close(self):
        for hk in self.hooks: hk.remove()
        for layer, act in zip(self.dm.model.layers, self.saved_act): layer.mlp.merged_input_act = act
        for n in ("vae2llm", "llm2vae"):
            lin = getattr(self.dm, n); lin.weight.requires_grad_(False); lin.bias.requires_grad_(False); lin.weight.data = self.io_orig[n].clone(); lin.bias.data = self.io_orig_b[n].clone()
        torch.cuda.empty_cache()
    # ---- AR prefix cache via ComfyUI's own conditioning builder
    @torch.no_grad()
    def prefix_cache(self, item, tokens):
        pre, _ = build_sequences(item, self.tok, "off"); dtype = torch.bfloat16
        ctx, chunks = self.te._acoustic_conditioning(pre, [int(t) for t in tokens], dtype); start, end, ks, ke = chunks[0]
        ar_len = ke - ks; prefix = ctx[:, ks:ke].to(self.dev).reshape(1, ar_len, self.L, 2, self.NKV, self.HD).permute(2, 3, 0, 4, 1, 5)     # [L,2,B,kv,ar_len,hd]
        return prefix, ar_len
    # ---- functional decoder forward with hooks-friendly ops (their block uses out= adds and a compiled rope kernel)
    def _rope(self, start, N):
        inv = 1.0 / (self.THETA ** (torch.arange(0, self.HD, 2, device=self.dev).float() / self.HD)); f = torch.outer(torch.arange(start, start + N, device=self.dev).float(), inv); emb = torch.cat([f, f], -1); return emb.cos()[None, None], emb.sin()[None, None]
    @staticmethod
    def _rot(x): h = x.shape[-1] // 2; return torch.cat([-x[..., h:], x[..., :h]], -1)
    def _attn(self, sa, x, cos, sin, pk, pv):
        B, S, _ = x.shape; q, k, v = sa.qkv_proj(x).split((sa.inner_size, sa.kv_size, sa.kv_size), -1)
        q = q.view(B, S, self.NH, self.HD).transpose(1, 2); k = k.view(B, S, self.NKV, self.HD).transpose(1, 2); v = v.view(B, S, self.NKV, self.HD).transpose(1, 2)
        if sa.q_norm is not None: q = sa.q_norm(q)
        if sa.k_norm is not None: k = sa.k_norm(k)
        q = (q * cos + self._rot(q) * sin).to(v.dtype); k = (k * cos + self._rot(k) * sin).to(v.dtype)
        o = F.scaled_dot_product_attention(q, torch.cat([pk.to(k.dtype), k], 2), torch.cat([pv.to(v.dtype), v], 2), is_causal=False, enable_gqa=True)
        return sa.o_proj(o.transpose(1, 2).reshape(B, S, sa.inner_size))
    def _blk(self, layer, x, cos, sin, pk, pv):
        x = x + self._attn(layer.self_attn, layer.input_layernorm(x), cos, sin, pk, pv); return x + layer.mlp(layer.post_attention_layernorm(x))
    def velocity(self, xt, sigma, prefix, ar_len, grad=True):
        """xt: [N,64] window latents at noise level sigma -> predicted v [N,64] (ComfyUI FLOW convention: v = noise - x1)"""
        N = xt.shape[0] + 2; state = F.pad(xt, (0, 0, 1, 1))[None].to(torch.bfloat16)
        t = self.ms.timestep(torch.tensor([sigma], device=self.dev)).to(torch.bfloat16); time = self.dm.time_embedder(t, torch.bfloat16)[:, None]
        state = self.dm.vae2llm(state) + time + self.dm.latent_pos_embed(N, state)[None]; cos, sin = self._rope(ar_len, N)
        for i, layer in enumerate(self.dm.model.layers):
            pk, pv = prefix[i, 0], prefix[i, 1]
            state = checkpoint(self._blk, layer, state, cos, sin, pk, pv, use_reentrant=False) if grad else self._blk(layer, state, cos, sin, pk, pv)
        return self.dm.llm2vae(self.dm.model.norm(state))[0, 1:-1].float()
    def flow_loss(self, x1, prefix, ar_len, sigma, noise, grad=True):
        xt = self.ms.noise_scaling(torch.tensor(sigma, device=self.dev), noise, x1); target = noise - x1
        return F.mse_loss(self.velocity(xt, sigma, prefix, ar_len, grad), target)
    def export(self, path, meta):
        out = {}; i = 0
        for l in range(self.L):
            for blk, proj in TARGETS: hk = self.hooks[i]; i += 1; out[f"diffusion_model.model.layers.{l}.{blk}.{proj}.lora_down.weight"] = hk.A.detach().to(torch.bfloat16).cpu().contiguous(); out[f"diffusion_model.model.layers.{l}.{blk}.{proj}.lora_up.weight"] = hk.B.detach().to(torch.bfloat16).cpu().contiguous()
        for n in ("vae2llm", "llm2vae"):
            lin = getattr(self.dm, n); out[f"diffusion_model.{n}.diff"] = (lin.weight.detach().float() - self.io_orig[n].float()).cpu().contiguous(); out[f"diffusion_model.{n}.diff_b"] = (lin.bias.detach().float() - self.io_orig_b[n].float()).cpu().contiguous()
        save_file(out, path, metadata={"format": "pt", "fs_audio": "decoder adapter (NAR LoRA + vae2llm/llm2vae diffs)", "rank": str(self.rank), "scale": "1.0 (no alpha)", **{k: str(v) for k, v in meta.items()}})
def train(model, clip, artist, cfg, status=lambda **k: None):
    """cfg: rank steps lr io_lr window_frames eval_every ckpt_from ckpt_every seed out_dir name"""
    torch.backends.cuda.matmul.allow_tf32 = True; random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    tr = DecoderTrainer(model, clip, cfg["rank"])
    try:
        items = [x for x in artist if x.get("latents") is not None]
        if not items: raise ValueError("Dataset has no latents: rebuild it with store_latents enabled.")
        train_items = [x for x in items if not x["held"]] or items; val_items = [x for x in items if x["held"]] or train_items[:2]; WIN = cfg["window_frames"]
        status(stage="Training decoder", detail=f"{len(train_items)} songs, rank {cfg['rank']}, {cfg['steps']} steps, {WIN / 25:.0f} s windows")
        opt = torch.optim.AdamW([{"params": tr.params, "lr": cfg["lr"]}, {"params": tr.io_params, "lr": cfg["io_lr"]}], weight_decay=0.0, betas=(0.9, 0.95)); base = [g["lr"] for g in opt.param_groups]; STEPS = cfg["steps"]
        cache = {}
        def window(it, s=None):
            n = min(len(it["codec"]), len(it["latents"])); s = random.randint(0, max(0, n - WIN)) if s is None else s; toks = it["codec"][s:s + WIN]
            key = (it["name"], s)
            if key not in cache:
                if len(cache) > 64: cache.clear()
                cache[key] = tr.prefix_cache(it, toks)
            return torch.tensor(np.asarray(it["latents"][s:s + WIN], dtype=np.float32), device=tr.dev), cache[key]
        @torch.no_grad()
        def evaluate():
            g = torch.Generator(device="cpu").manual_seed(123); tot = 0; k = 0
            for it in val_items[:4]:
                n = min(len(it["codec"]), len(it["latents"])); x1, (pre, al) = window(it, max(0, (n - WIN) // 2)); noise = torch.randn(x1.shape, generator=g).to(tr.dev)
                for s in (0.2, 0.5, 0.8): tot += tr.flow_loss(x1, pre, al, s, noise, grad=False).item(); k += 1
            return tot / k
        e = evaluate(); log = [{"step": 0, "artist": e}]; status(step=0, evals={"artist": e}, total=STEPS); best = e; t0 = time.time(); os.makedirs(cfg["out_dir"], exist_ok=True); ckpts = []
        for st in range(1, STEPS + 1):
            comfy.model_management.throw_exception_if_processing_interrupted()
            mult = min(1, st / 50) * (0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * st / STEPS)))
            for g_, b in zip(opt.param_groups, base): g_["lr"] = b * mult
            it = random.choice(train_items); x1, (pre, al) = window(it); sigma = float(np.clip(np.random.beta(2, 2), 0.02, 0.98)); noise = torch.randn_like(x1)
            loss = tr.flow_loss(x1, pre, al, sigma, noise); loss.backward(); torch.nn.utils.clip_grad_norm_(tr.params + tr.io_params, 1.0); opt.step(); opt.zero_grad(set_to_none=True)
            if st % 5 == 0 or st <= 3: status(step=st, loss=float(loss.detach()), total=STEPS, eta=(time.time() - t0) / st * (STEPS - st), seq=int(x1.shape[0]))
            if st % cfg["eval_every"] == 0 or st == STEPS:
                e = evaluate(); log.append({"step": st, "artist": e}); status(step=st, evals={"artist": e}, total=STEPS)
                if e < best: best = e; tr.export(os.path.join(cfg["out_dir"], f"{cfg['name']}_best.safetensors"), {"artist": e, "step": st, "config": json.dumps(cfg)})
            if (st >= cfg["ckpt_from"] and (st - cfg["ckpt_from"]) % cfg["ckpt_every"] == 0) or st == STEPS:
                p = os.path.join(cfg["out_dir"], f"{cfg['name']}_step{st}.safetensors"); tr.export(p, {"step": st, "config": json.dumps(cfg)}); ckpts.append(p)
        json.dump(log, open(os.path.join(cfg["out_dir"], f"{cfg['name']}_log.json"), "w"), indent=1)
        return {"best_artist_loss": best, "checkpoints": ckpts, "log": log, "final": ckpts[-1] if ckpts else None, "songs": len(train_items)}
    finally:
        tr.close()
