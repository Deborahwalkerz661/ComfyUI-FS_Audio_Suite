"""One training loop for both YuE2 experts (ai-toolkit's layout, on ComfyUI's own weights):
  planner LoRA  : whole-song next-token CE from the song's start (+ END up-weighting, score-first layout half the time, artist/regularizer mixed
                  batches) with a KL(base || lora) trust region on the next-token distributions;
  decoder LoRA  : flow-matching on the artist's VAE latents in windows, conditioned on the (LoRA'd) planner's context, recomputed every step
                  and detached so the flow loss never trains the planner.
Both halves get an EMA copy (decay warmed up as min(decay, (1+n)/(10+n))) that is what gets exported; checkpoints are ONE file carrying both
halves (text_encoders.* + diffusion_model.*) so a single LoRA loader applies the whole artist."""
import os, math, time, random, json, numpy as np, torch
from safetensors.torch import save_file
import comfy.model_management
from .trainer import PlannerTrainer
from .decoder import DecoderTrainer
from .data import build_sequences, ABC_START
from safetensors.torch import load_file
from .trainer import TARGETS as P_TARGETS
class EMA:
    def __init__(s, params, decay): s.params = list(params); s.decay = decay; s.shadow = [p.detach().clone() for p in s.params]; s.n = 0
    @torch.no_grad()
    def update(s):
        s.n += 1; d = min(s.decay, (1 + s.n) / (10 + s.n))                       # warm-up form; never exceeds `decay`
        for sh, p in zip(s.shadow, s.params): sh.mul_(d).add_(p.detach(), alpha=1 - d)
    @torch.no_grad()
    def swap(s):
        for sh, p in zip(s.shadow, s.params): tmp = p.detach().clone(); p.copy_(sh); sh.copy_(tmp)
def train(model, clip, artist, regularizer, cfg, status=lambda **k: None):
    """cfg: name out_dir rank_planner rank_decoder steps decoder_steps lr_planner lr_decoder lr_io artist_fraction batch_songs kl_weight
    score_first_fraction end_weight max_tokens window_frames ema_decay eval_every ckpt_from ckpt_every seed"""
    torch.backends.cuda.matmul.allow_tf32 = True; random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    pl = PlannerTrainer(clip, cfg["rank_planner"]); dc = DecoderTrainer(model, clip, cfg["rank_decoder"])
    try:
        if cfg.get("resume_from"):
            sd = load_file(cfg["resume_from"]); n_p = n_d = 0
            with torch.no_grad():
                i = 0
                for l in range(len(pl.L2.layers)):
                    for blk, proj in P_TARGETS:
                        hk = pl.hooks[i]; i += 1; kd = f"text_encoders.model.layers.{l}.{blk}.{proj}.lora_down.weight"; ku = f"text_encoders.model.layers.{l}.{blk}.{proj}.lora_up.weight"
                        if kd in sd and sd[kd].shape == hk.A.shape: hk.A.copy_(sd[kd].to(hk.A)); hk.B.copy_(sd[ku].to(hk.B)); n_p += 1
                i = 0
                for l in range(dc.L):
                    for blk, proj in dc.TARGETS if hasattr(dc, "TARGETS") else P_TARGETS:
                        hk = dc.hooks[i]; i += 1; kd = f"diffusion_model.model.layers.{l}.{blk}.{proj}.lora_down.weight"; ku = f"diffusion_model.model.layers.{l}.{blk}.{proj}.lora_up.weight"
                        if kd in sd and sd[kd].shape == hk.A.shape: hk.A.copy_(sd[kd].to(hk.A)); hk.B.copy_(sd[ku].to(hk.B)); n_d += 1
                for n in ("vae2llm", "llm2vae"):
                    if f"diffusion_model.{n}.diff" in sd: dc.io_w[n].add_(sd[f"diffusion_model.{n}.diff"].to(dc.io_w[n])); dc.io_b[n].add_(sd[f"diffusion_model.{n}.diff_b"].to(dc.io_b[n]))
            status(stage="Resumed", detail=f"{os.path.basename(cfg['resume_from'])}: {n_p} planner + {n_d} decoder LoRA pairs loaded")
        mt = cfg["max_tokens"]; a_train = [x for x in artist if not x["held"] and pl.fits(x, "off", mt)]; a_val = [x for x in artist if x["held"]] or a_train[:4]
        r_train = [x for x in (regularizer or []) if not x["held"] and pl.fits(x, "off", mt)]; r_val = [x for x in (regularizer or []) if x["held"]][:6]
        d_train = [x for x in artist if not x["held"] and x.get("latents") is not None] or [x for x in artist if x.get("latents") is not None]; d_val = [x for x in artist if x["held"] and x.get("latents") is not None] or d_train[:2]
        if not d_train: raise ValueError("Dataset has no latents: rebuild it with store_latents enabled.")
        has_abc = any(x.get("abc") for x in a_train); sf_ = cfg["score_first_fraction"] if has_abc else 0.0; WIN = cfg["window_frames"]; STEPS = cfg["steps"]
        dec_per = max(1, round(cfg["decoder_steps"] / STEPS)); K = max(1, int(cfg["batch_songs"])); klw = cfg["kl_weight"]
        status(stage="Training artist", detail=f"{len(a_train)} artist / {len(r_train)} regularizer songs | planner r{cfg['rank_planner']} {STEPS} steps x {K} songs, KL {klw} | decoder r{cfg['rank_decoder']} {dec_per}/step ({dec_per * STEPS} total), {WIN / 25:.0f} s windows" + (f" | score-first {sf_:.0%}" if sf_ else ""))
        opt_p = torch.optim.AdamW(pl.params, lr=cfg["lr_planner"], weight_decay=0.0, betas=(0.9, 0.95))
        opt_d = torch.optim.AdamW([{"params": dc.params, "lr": cfg["lr_decoder"]}, {"params": dc.io_params, "lr": cfg["lr_io"]}], weight_decay=0.0, betas=(0.9, 0.95)); base_d = [g["lr"] for g in opt_d.param_groups]
        ema = EMA(pl.params + dc.params + dc.io_params, cfg["ema_decay"]) if cfg["ema_decay"] > 0 else None
        WARM = 50; sched = lambda st: min(1, st / WARM) * (0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * min(st, STEPS) / STEPS)))
        def window(it, s=None):
            n = min(len(it["codec"]), len(it["latents"])); s = random.randint(0, max(0, n - WIN)) if s is None else s
            with torch.no_grad(): pre, al = dc.prefix_cache(it, it["codec"][s:s + WIN])           # LoRA'd planner context, recomputed each call, no grad
            return torch.tensor(np.asarray(it["latents"][s:s + WIN], dtype=np.float32), device=dc.dev), pre, al
        @torch.no_grad()
        def evaluate():
            r = {}
            for tag, items in (("artist", a_val[:6]), ("regularizer", r_val)):
                if items: r[tag] = sum(pl.loss(*pl.seq(it, "off", mt), grad=False)[0].item() for it in items) / len(items)
            g = torch.Generator(device="cpu").manual_seed(123); tot = 0; k = 0
            for it in d_val[:4]:
                n = min(len(it["codec"]), len(it["latents"])); x1, pre, al = window(it, max(0, (n - WIN) // 2)); noise = torch.randn(x1.shape, generator=g).to(dc.dev)
                for sg in (0.2, 0.5, 0.8): tot += dc.flow_loss(x1, pre, al, sg, noise, grad=False).item(); k += 1
            r["decoder"] = tot / max(1, k); return r
        def export(path, meta):
            if ema: ema.swap()
            try: save_file({**pl.state(), **dc.state()}, path, metadata={"format": "pt", "fs_audio": "artist LoRA (planner text_encoders.* + decoder diffusion_model.*)", "rank_planner": str(cfg["rank_planner"]), "rank_decoder": str(cfg["rank_decoder"]), "scale": "1.0 (no alpha)", **{k: str(v) for k, v in meta.items()}})
            finally:
                if ema: ema.swap()
        log = []; e = evaluate(); log.append({"step": 0, **e}); status(step=0, evals=e, total=STEPS); best = e.get("artist", 9e9); t0 = time.time(); os.makedirs(cfg["out_dir"], exist_ok=True); ckpts = []
        for st in range(1, STEPS + 1):
            comfy.model_management.throw_exception_if_processing_interrupted(); m = sched(st)
            for g in opt_p.param_groups: g["lr"] = cfg["lr_planner"] * m
            for g, b in zip(opt_d.param_groups, base_d): g["lr"] = b * m
            # ---- planner: K coin-flipped whole songs, CE (+ score-writing pass) + KL trust region
            ce_sum = kl_sum = 0.0
            for _k in range(K):
                it = random.choice(a_train) if (not r_train or random.random() < cfg["artist_fraction"]) else random.choice(r_train)
                layout = "full" if (it.get("abc") and random.random() < sf_) else "off"
                ids, ls = pl.seq(it, layout, mt); ce, kl = pl.loss(ids, ls, cfg["end_weight"], kl_w=klw); ((ce + klw * kl) / K).backward(); ce_sum += float(ce) / K; kl_sum += float(kl) / K
                if layout == "off" and sf_ and it.get("abc"):
                    pre, _ = build_sequences(it, pl.tok, "full"); aids = torch.tensor([pre[:-1]], device=pl.dev); (pl.loss(aids, pre.index(ABC_START) + 1)[0] * 0.5 / K).backward()
            torch.nn.utils.clip_grad_norm_(pl.params, 1.0); opt_p.step(); opt_p.zero_grad(set_to_none=True)
            # ---- decoder: dec_per flow steps on artist windows, conditioned on the current planner (detached)
            dl = 0.0
            for _d in range(dec_per):
                it = random.choice(d_train); x1, pre, al = window(it); sigma = float(np.clip(np.random.beta(2, 2), 0.02, 0.98)); noise = torch.randn_like(x1)
                fl = dc.flow_loss(x1, pre, al, sigma, noise); (fl / dec_per).backward(); dl += float(fl) / dec_per
            torch.nn.utils.clip_grad_norm_(dc.params + dc.io_params, 1.0); opt_d.step(); opt_d.zero_grad(set_to_none=True)
            if ema: ema.update()
            if st % 5 == 0 or st <= 3: status(step=st, loss=ce_sum, kl=kl_sum, decoder_loss=dl, total=STEPS, eta=(time.time() - t0) / st * (STEPS - st))
            if st % cfg["eval_every"] == 0 or st == STEPS:
                if ema: ema.swap()
                try: e = evaluate()
                finally:
                    if ema: ema.swap()
                log.append({"step": st, **e}); status(step=st, evals=e, total=STEPS)
                if e.get("artist", 9e9) < best: best = e["artist"]; export(os.path.join(cfg["out_dir"], f"{cfg['name']}_best.safetensors"), {**e, "step": st, "config": json.dumps(cfg)})
            if (st >= cfg["ckpt_from"] and (st - cfg["ckpt_from"]) % cfg["ckpt_every"] == 0) or st == STEPS:
                p = os.path.join(cfg["out_dir"], f"{cfg['name']}_step{st}.safetensors"); export(p, {"step": st, "config": json.dumps(cfg)}); ckpts.append(p)
        json.dump(log, open(os.path.join(cfg["out_dir"], f"{cfg['name']}_log.json"), "w"), indent=1)
        return {"best_artist_loss": best, "checkpoints": ckpts, "log": log, "final": ckpts[-1] if ckpts else None, "artist_train": len(a_train), "regularizer_train": len(r_train), "decoder_songs": len(d_train), "score_first": sf_}
    finally:
        pl.close(); dc.close()
