// FS_Audio Suite — Make the Robot Do It edition.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const MINT = "#62f0da", MINT_DIM = "#3fbfa9", INK = "#0b1116", PANEL = "#111820", LINE = "#22303c", TEXT = "#e9f3f1";
const CSS = `
.hs-wrap{font-family:"Inter",ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:${TEXT};padding:10px 12px 12px;box-sizing:border-box;width:100%;
  background:${PANEL};border:1px solid ${LINE};border-left:3px solid ${MINT};position:relative;
  background-image:radial-gradient(rgba(98,240,218,.10) 1px,transparent 1px);background-size:14px 14px;background-position:right top;}
.hs-wrap::before{content:"";position:absolute;right:8px;top:6px;width:38px;height:2px;background:linear-gradient(90deg,transparent,${MINT});transform:skewX(-35deg);opacity:.9;}
.hs-brand{display:flex;align-items:flex-start;gap:8px;font-weight:900;letter-spacing:1.5px;font-size:10px;text-transform:uppercase;margin-bottom:6px;min-height:52px;padding-top:2px;}
.hs-brand .grow{flex:1} .hs-avatar{width:50px;height:50px;border-radius:12px;object-fit:cover;border:3px solid ${MINT};box-shadow:0 0 0 2px ${INK},0 0 18px rgba(98,240,218,.35);background:${INK};margin-top:-2px;}
.hs-brand .w{color:#fff} .hs-brand .m{color:${MINT}} .hs-brand .sl{color:${MINT};font-weight:900;margin-left:4px;transform:skewX(-20deg);display:inline-block}
.hs-row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;}
.hs-btn{display:inline-flex;align-items:center;gap:8px;border:0;padding:8px 16px;font-weight:900;letter-spacing:1px;text-transform:uppercase;font-size:11px;cursor:pointer;
  background:${MINT};color:${INK};clip-path:polygon(9px 0,100% 0,100% calc(100% - 9px),calc(100% - 9px) 100%,0 100%,0 9px);transition:filter .12s,transform .12s;}
.hs-btn:hover{filter:brightness(1.12);transform:translateY(-1px);} .hs-btn.ghost{background:${LINE};color:${MINT};} .hs-btn.rec{background:#ff5e6c;color:#fff;animation:hs-pulse 1.1s ease-in-out infinite;}
.hs-btn.icon{padding:8px 12px;min-width:40px;justify-content:center;}
@keyframes hs-pulse{0%,100%{filter:brightness(1)}50%{filter:brightness(1.35)}}
.hs-wave{width:100%;height:64px;display:block;margin-top:10px;background:${INK};border:1px solid ${LINE};cursor:pointer;}
.hs-meta{font-size:11px;color:#9fb3ae;margin-top:6px;} .hs-meta b{color:${MINT};font-weight:700}
.hs-stages{display:flex;gap:5px;margin:8px 0 2px;} .hs-stage{flex:1;height:7px;background:${LINE};position:relative;overflow:hidden;clip-path:polygon(4px 0,100% 0,calc(100% - 4px) 100%,0 100%);}
.hs-stage.on{background:linear-gradient(90deg,${MINT_DIM},${MINT},${MINT_DIM});background-size:200% 100%;animation:hs-flow 1.4s linear infinite;} .hs-stage.done{background:${MINT};}
@keyframes hs-flow{0%{background-position:0% 0}100%{background-position:200% 0}}
.hs-label{font-size:12px;font-weight:800;text-transform:uppercase;letter-spacing:.8px;color:#fff;} .hs-detail{font-size:11px;color:${MINT};margin-left:8px;}
.hs-player{width:100%;margin-top:8px;filter:hue-rotate(115deg) saturate(1.1) brightness(1.05);}
.hs-score{margin-top:8px;max-height:220px;overflow:auto;background:${INK};border:1px solid ${LINE};padding:8px 10px;font:11px/1.35 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre;color:#a9bdb8;}
.hs-score .hum{color:${MINT};background:rgba(98,240,218,.08);} .hs-score .mark{color:#fff;font-weight:800;}
.hs-tag{display:inline-block;font-size:10px;font-weight:800;letter-spacing:.6px;text-transform:uppercase;padding:3px 9px;background:${LINE};color:${MINT};margin:0 6px 6px 0;clip-path:polygon(6px 0,100% 0,calc(100% - 6px) 100%,0 100%);}
`;
if (!document.getElementById("humsong-css")) { const s = document.createElement("style"); s.id = "humsong-css"; s.textContent = CSS; document.head.appendChild(s); }

const STAGES = ["Listening to your hum", "Composing the score", "Writing the song", "Rendering", "Decoding audio"];
const NORA = new Image(); NORA.src = new URL("./nora.png", import.meta.url).href;
const brand = () => { const d = document.createElement("div"); d.className = "hs-brand"; d.innerHTML = `<span class="w">Make the</span><span class="m">Robot Do It</span><span class="sl">//</span><span class="grow"></span><img class="hs-avatar" src="${NORA.src}" alt="Nora">`; return d; };
const esc = (l) => l.replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

function paint(node) {
  node.color = "#131b24"; node.bgcolor = "#0e141b"; node.title_text_color = "#ffffff";
  const prev = node.onDrawForeground;
  node.onDrawForeground = function (ctx) { prev?.apply(this, arguments); if (this.flags?.collapsed) return; ctx.save(); ctx.strokeStyle = MINT; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(this.size[0], 0); ctx.stroke(); ctx.restore(); };
}
// fixed-height DOM panels so the node always encloses its content
function addPanel(node, name, el, height, minWidth) {
  let h = height;
  const w = node.addDOMWidget(name, "div", el, { serialize: false, getMinHeight: () => h, getMaxHeight: () => h, getHeight: () => h });
  w.computeSize = () => [node.size[0], h];
  const PAD = 20;   // breathing room under the panel
  w.hsResize = (nh) => { h = nh; node.setSize([Math.max(node.size[0], minWidth), node.computeSize()[1] + PAD]); node.setDirtyCanvas(true, true); };
  node.setSize([Math.max(node.size[0], minWidth), node.computeSize()[1] + PAD]);
  return w;
}

function drawWave(canvas, samples, cursor = -1) {
  const ctx = canvas.getContext("2d"), W = canvas.width = Math.max(200, canvas.clientWidth) * 2, H = canvas.height = 128;
  ctx.fillStyle = INK; ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = "rgba(98,240,218,.12)"; for (let x = 8; x < W; x += 28) for (let y = 8; y < H; y += 28) ctx.fillRect(x, y, 2, 2);
  ctx.fillStyle = "rgba(98,240,218,.25)"; ctx.fillRect(0, H / 2, W, 1);
  const n = Math.min(Math.floor(W / 3), 700), step = Math.max(1, Math.floor(samples.length / n));
  for (let i = 0; i < n; i++) { let m = 0; for (let j = 0; j < step; j++) m = Math.max(m, Math.abs(samples[i * step + j] || 0)); const h = Math.max(2, m * H * .92); const played = cursor >= 0 && i / n <= cursor; ctx.fillStyle = played ? "#ffffff" : MINT; ctx.fillRect(i * (W / n), (H - h) / 2, Math.max(1.5, W / n - 1.5), h); }
  if (cursor >= 0) { ctx.fillStyle = "#fff"; ctx.fillRect(cursor * W - 1, 0, 2, H); }
}

class Preview {   // decoded hum + waveform + play/stop with a moving cursor
  constructor(canvas, meta) { this.canvas = canvas; this.meta = meta; this.buffer = null; this.ctx = null; this.src = null; this.raf = 0; this.t0 = 0; this.peaks = []; }
  async setBlob(blob) { const ab = await blob.arrayBuffer(); await this.setArrayBuffer(ab); }
  async setArrayBuffer(ab) { const ac = new AudioContext(); this.buffer = await ac.decodeAudioData(ab); ac.close(); this.peaks = Array.from(this.buffer.getChannelData(0)); drawWave(this.canvas, this.peaks); }
  async setFile(name) { if (!name) return; try { const r = await api.fetchApi(`/view?filename=${encodeURIComponent(name)}&type=input`); if (r.status !== 200) throw new Error(r.status); await this.setArrayBuffer(await r.arrayBuffer()); this.meta.innerHTML = `Loaded <b>${esc(name)}</b> · ${this.buffer.duration.toFixed(1)} s — press ▶ to hear it.`; } catch (e) { this.meta.textContent = "Could not load " + name + ": " + e.message; } }
  playing() { return !!this.src; }
  stop() { try { this.src?.stop(); } catch {} this.src = null; cancelAnimationFrame(this.raf); this.ctx?.close(); this.ctx = null; drawWave(this.canvas, this.peaks); }
  play(onEnd) { if (!this.buffer) return; this.stop(); this.ctx = new AudioContext(); this.src = this.ctx.createBufferSource(); this.src.buffer = this.buffer; this.src.connect(this.ctx.destination); this.t0 = this.ctx.currentTime; this.src.onended = () => { this.stop(); onEnd?.(); }; this.src.start();
    const tick = () => { if (!this.src) return; drawWave(this.canvas, this.peaks, (this.ctx.currentTime - this.t0) / this.buffer.duration); this.raf = requestAnimationFrame(tick); }; tick(); }
  seek(frac, onEnd) { if (!this.buffer) return; this.stop(); this.ctx = new AudioContext(); this.src = this.ctx.createBufferSource(); this.src.buffer = this.buffer; this.src.connect(this.ctx.destination); this.t0 = this.ctx.currentTime - frac * this.buffer.duration; this.src.onended = () => { this.stop(); onEnd?.(); }; this.src.start(0, frac * this.buffer.duration);
    const tick = () => { if (!this.src) return; drawWave(this.canvas, this.peaks, (this.ctx.currentTime - this.t0) / this.buffer.duration); this.raf = requestAnimationFrame(tick); }; tick(); }
}

function playerBlock(url, tags) {
  const box = document.createElement("div"); if (tags) { const t = document.createElement("div"); t.innerHTML = tags; box.appendChild(t); }
  const player = document.createElement("audio"); player.className = "hs-player"; player.controls = true; player.src = url; box.appendChild(player); return box;
}

app.registerExtension({
  name: "FS_Audio Suite",
  async setup() {
    api.addEventListener("humsong.stage", ({ detail }) => { const node = app.graph.getNodeById(Number(detail.node)); node?.hsStage?.(detail.stage, detail.pct, detail.detail); });
    api.addEventListener("fsaudio.train", ({ detail }) => { const node = app.graph.getNodeById(Number(detail.node)); node?.fsTrain?.(detail); });
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const FS = new Set(["FSAudioModelLoader", "FSAudioLoraLoader", "FSAudioAdapterLoader", "HumInput", "FSAudioSampler", "FSAudioOutput", "FSAudioAdapterDownloader", "FSAudioTrainAssets", "FSAudioDatasetBuilder", "FSAudioRegularizer", "FSAudioLoraTrainer", "FSAudioDecoderTrainer"]); if (!FS.has(nodeData.name)) return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () { onCreated?.apply(this, arguments); paint(this); try { this.hsSetup?.(); } catch (e) { console.error("[HumSong] UI setup failed for", nodeData.name, e); } };

    if (nodeData.name === "HumInput") {
      nodeType.prototype.hsSetup = function () {
        const wrap = document.createElement("div"); wrap.className = "hs-wrap"; wrap.appendChild(brand());
        const row = document.createElement("div"); row.className = "hs-row";
        const play = document.createElement("button"); play.className = "hs-btn icon"; play.textContent = "▶"; play.title = "Play the hum";
        const rec = document.createElement("button"); rec.className = "hs-btn"; rec.textContent = "🎙 Record";
        const up = document.createElement("button"); up.className = "hs-btn ghost"; up.textContent = "⬆ Upload";
        const dl = document.createElement("a"); dl.className = "hs-btn ghost icon"; dl.textContent = "⬇"; dl.title = "Download this hum"; dl.style.textDecoration = "none";
        const picker = document.createElement("input"); picker.type = "file"; picker.accept = "audio/*"; picker.style.display = "none";
        const wave = document.createElement("canvas"); wave.className = "hs-wave"; const meta = document.createElement("div"); meta.className = "hs-meta"; meta.textContent = "Record here, upload a file, or pick one above. 10–30 seconds of melody is plenty.";
        row.append(play, rec, up, dl, picker); wrap.append(row, wave, meta); addPanel(this, "humsong_rec", wrap, 236, 440);
        const setDownload = (name) => { if (!name || name.startsWith("(")) { dl.removeAttribute("href"); dl.style.opacity = .4; return; } dl.style.opacity = 1; dl.href = api.apiURL(`/view?filename=${encodeURIComponent(name)}&type=input`); dl.download = name; };
        const pv = new Preview(wave, meta); this.hsPreview = pv; drawWave(wave, []);
        const setPlayIcon = () => { play.textContent = pv.playing() ? "■" : "▶"; };
        play.onclick = () => { if (pv.playing()) pv.stop(); else pv.play(setPlayIcon); setPlayIcon(); };
        wave.onclick = (e) => { const frac = (e.offsetX / wave.clientWidth); pv.seek(frac, setPlayIcon); setPlayIcon(); };
        const combo = this.widgets.find(w => w.name === "audio");
        const setFile = (name) => { if (combo) { if (!combo.options.values.includes(name)) combo.options.values.push(name); combo.value = name; } setDownload(name); this.setDirtyCanvas(true, true); };
        if (combo) { const cb = combo.callback; combo.callback = (v, ...rest) => { cb?.call(combo, v, ...rest); pv.setFile(v); setDownload(v); }; if (combo.value && combo.value !== "(record or upload a hum)") { pv.setFile(combo.value); setDownload(combo.value); } else setDownload(null); }
        const upload = async (blob, name) => { const fd = new FormData(); fd.append("image", blob, name); fd.append("type", "input"); const r = await api.fetchApi("/upload/image", { method: "POST", body: fd }); if (r.status !== 200) throw new Error("upload " + r.status); return (await r.json()).name; };
        picker.onchange = async () => { const f = picker.files?.[0]; if (!f) return; meta.textContent = "Uploading…"; try { const n = await upload(f, f.name); setFile(n); await pv.setBlob(f); meta.innerHTML = `Loaded <b>${esc(n)}</b> · ${pv.buffer.duration.toFixed(1)} s — ready.`; } catch (e) { meta.textContent = "Upload failed: " + e.message; } };
        up.onclick = () => picker.click();
        let mr = null, chunks = [], actx = null, raf = 0, t0 = 0;
        rec.onclick = async () => {
          if (mr && mr.state === "recording") { mr.stop(); return; }
          try {
            pv.stop(); setPlayIcon();
            const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: true } });
            chunks = []; mr = new MediaRecorder(stream, { mimeType: MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "audio/webm" });
            actx = new AudioContext(); const an = actx.createAnalyser(); an.fftSize = 2048; actx.createMediaStreamSource(stream).connect(an); const buf = new Float32Array(an.fftSize); const hist = []; t0 = performance.now();
            const meter = () => { an.getFloatTimeDomainData(buf); let m = 0; for (const v of buf) m = Math.max(m, Math.abs(v)); hist.push(m); if (hist.length > 500) hist.shift(); drawWave(wave, hist); meta.innerHTML = `<b>Recording</b> ${((performance.now() - t0) / 1000).toFixed(1)} s — press ■ to stop`; raf = requestAnimationFrame(meter); }; meter();
            mr.ondataavailable = (e) => e.data.size && chunks.push(e.data);
            mr.onstop = async () => {
              cancelAnimationFrame(raf); actx?.close(); actx = null; stream.getTracks().forEach(t => t.stop()); rec.classList.remove("rec"); rec.textContent = "🎙 Record";
              const blob = new Blob(chunks, { type: "audio/webm" }); const name = `hum_${new Date().toISOString().replace(/[:.]/g, "-")}.webm`; meta.textContent = "Uploading…";
              try { const n = await upload(blob, name); setFile(n); await pv.setBlob(blob); meta.innerHTML = `Saved <b>${esc(n)}</b> · ${pv.buffer.duration.toFixed(1)} s — press ▶ to check it, then queue.`; } catch (e) { meta.textContent = "Upload failed: " + e.message; }
            };
            mr.start(250); rec.classList.add("rec"); rec.textContent = "■ Stop";
          } catch (e) { meta.textContent = "Microphone unavailable: " + e.message; }
        };
      };
    }

    if (nodeData.name === "FSAudioSampler") {
      nodeType.prototype.hsSetup = function () {
        const wrap = document.createElement("div"); wrap.className = "hs-wrap"; wrap.appendChild(brand());
        const row = document.createElement("div"); const label = document.createElement("span"); label.className = "hs-label"; label.textContent = "Ready"; const det = document.createElement("span"); det.className = "hs-detail"; row.append(label, det);
        const bar = document.createElement("div"); bar.className = "hs-stages"; const segs = STAGES.map(() => { const s = document.createElement("div"); s.className = "hs-stage"; bar.appendChild(s); return s; });
        wrap.append(row, bar); addPanel(this, "humsong_progress", wrap, 132, 460);
        this.hsStage = (stage, pct, detail) => { label.textContent = stage; det.textContent = detail || ""; const idx = STAGES.indexOf(stage); segs.forEach((s, i) => { s.className = "hs-stage" + (stage === "Done" || i < idx ? " done" : i === idx ? " on" : ""); }); };
      };
    }


    if (nodeData.name === "FSAudioAdapterDownloader") {
      nodeType.prototype.hsSetup = function () {
        const wrap = document.createElement("div"); wrap.className = "hs-wrap"; wrap.appendChild(brand());
        const row = document.createElement("div"); const label = document.createElement("span"); label.className = "hs-label"; label.textContent = "Queue to download"; const det = document.createElement("span"); det.className = "hs-detail"; row.append(label, det);
        const out = document.createElement("div"); out.className = "hs-meta"; out.style.whiteSpace = "pre-line"; out.textContent = "Fetches the adapter straight from Hugging Face into models/loras so the Adapter Loader can see it. Tick download_base_models to also fetch the YuE2 checkpoint (7.8 GB) and melody transcriber (1.3 GB).";
        wrap.append(row, out); addPanel(this, "fs_download_panel", wrap, 150, 440); this.hsOut = out;
        this.hsStage = (stage, pct, detail) => { label.textContent = stage; det.textContent = detail || ""; };
      };
      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (msg) { onExecuted?.apply(this, arguments); if (this.hsOut && msg?.fs_download?.[0]) { this.hsOut.innerHTML = msg.fs_download[0].split("\n").map(l => `<b>✓</b> ${esc(l)}`).join("<br>"); this.hsOut.style.whiteSpace = "normal"; } };
    }

    if (nodeData.name === "FSAudioTrainAssets") {
      nodeType.prototype.hsSetup = function () {
        const wrap = document.createElement("div"); wrap.className = "hs-wrap"; wrap.appendChild(brand());
        const row = document.createElement("div"); const label = document.createElement("span"); label.className = "hs-label"; label.textContent = "Queue to download"; const det = document.createElement("span"); det.className = "hs-detail"; row.append(label, det);
        const out = document.createElement("div"); out.className = "hs-meta"; out.textContent = "Fetches the audio → token head into models/fs_audio. Needed once before building datasets."; wrap.append(row, out); addPanel(this, "fs_assets_panel", wrap, 140, 440); this.hsOut = out;
        this.hsStage = (stage, pct, detail) => { label.textContent = stage; det.textContent = detail || ""; };
      };
      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (msg) { onExecuted?.apply(this, arguments); if (this.hsOut && msg?.fs_download?.[0]) this.hsOut.innerHTML = `<b>✓</b> ${esc(msg.fs_download[0])}`; };
    }
    if (nodeData.name === "FSAudioDatasetBuilder") {
      nodeType.prototype.hsSetup = function () {
        const wrap = document.createElement("div"); wrap.className = "hs-wrap"; wrap.appendChild(brand());
        const row = document.createElement("div"); const label = document.createElement("span"); label.className = "hs-label"; label.textContent = "Ready"; const det = document.createElement("span"); det.className = "hs-detail"; row.append(label, det);
        const bar = document.createElement("div"); bar.className = "hs-stages"; const seg = document.createElement("div"); seg.className = "hs-stage"; bar.appendChild(seg);
        const out = document.createElement("div"); out.className = "hs-meta"; out.textContent = "Drop songs in a folder with <song>.txt (style) and <song>.lyrics.txt sidecars, then queue.";
        wrap.append(row, bar, out); addPanel(this, "fs_dataset_panel", wrap, 150, 460);
        this.fsTrain = (d) => { if (d.stage) label.textContent = d.stage; det.textContent = d.detail || ""; if (d.pct != null) { seg.className = "hs-stage on"; seg.style.background = `linear-gradient(90deg, ${MINT} ${d.pct}%, ${LINE} ${d.pct}%)`; seg.style.animation = "none"; } if (d.stage === "Done") { seg.className = "hs-stage done"; seg.style.background = ""; out.textContent = d.detail || ""; } };
      };
    }
    if (nodeData.name === "FSAudioLoraTrainer" || nodeData.name === "FSAudioDecoderTrainer") {
      nodeType.prototype.hsSetup = function () {
        const wrap = document.createElement("div"); wrap.className = "hs-wrap"; wrap.appendChild(brand());
        const row = document.createElement("div"); const label = document.createElement("span"); label.className = "hs-label"; label.textContent = "Ready"; const det = document.createElement("span"); det.className = "hs-detail"; row.append(label, det);
        const chart = document.createElement("canvas"); chart.className = "hs-wave"; chart.style.height = "110px"; chart.style.cursor = "default";
        const out = document.createElement("div"); out.className = "hs-meta"; out.innerHTML = "Loss curve appears here. <b>mint</b> = training loss, <b>white</b> = held-out artist, <b>grey</b> = regularizer.";
        wrap.append(row, chart, out); addPanel(this, "fs_train_panel", wrap, 250, 500);
        const losses = [], evals = []; let total = 1;
        const draw = () => { const ctx = chart.getContext("2d"), W = chart.width = Math.max(200, chart.clientWidth) * 2, H = chart.height = 220; ctx.fillStyle = INK; ctx.fillRect(0, 0, W, H);
          ctx.fillStyle = "rgba(98,240,218,.12)"; for (let x = 8; x < W; x += 28) for (let y = 8; y < H; y += 28) ctx.fillRect(x, y, 2, 2);
          const all = losses.map(p => p[1]).concat(evals.flatMap(e => [e.artist, e.regularizer].filter(v => v != null))); if (!all.length) return;
          const lo = Math.min(...all) * 0.97, hi = Math.max(...all) * 1.03; const X = s => 12 + (W - 24) * s / total, Y = v => H - 10 - (H - 20) * (v - lo) / (hi - lo);
          ctx.strokeStyle = MINT; ctx.lineWidth = 2; ctx.beginPath(); losses.forEach(([s, v], i) => i ? ctx.lineTo(X(s), Y(v)) : ctx.moveTo(X(s), Y(v))); ctx.stroke();
          const line = (key, color) => { const pts = evals.filter(e => e[key] != null); if (pts.length < 1) return; ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.beginPath(); pts.forEach((e, i) => i ? ctx.lineTo(X(e.step), Y(e[key])) : ctx.moveTo(X(e.step), Y(e[key]))); ctx.stroke(); pts.forEach(e => { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(X(e.step), Y(e[key]), 4, 0, 7); ctx.fill(); }); };
          line("artist", "#ffffff"); line("regularizer", "#7b8a96");
          ctx.fillStyle = "#9fb3ae"; ctx.font = "18px ui-monospace, Menlo, monospace"; ctx.fillText(hi.toFixed(2), 14, 26); ctx.fillText(lo.toFixed(2), 14, H - 14); };
        this.fsTrain = (d) => { if (d.total) total = d.total; if (d.stage) label.textContent = d.stage; if (d.loss != null) { losses.push([d.step, d.loss]); if (losses.length > 2000) losses.shift(); label.textContent = `Step ${d.step} / ${total}`; det.textContent = `loss ${d.loss.toFixed(3)} · seq ${d.seq} · ETA ${Math.round((d.eta || 0) / 60)} min`; }
          if (d.evals) { evals.push({ step: d.step, ...d.evals }); const e = d.evals; out.innerHTML = `step <b>${d.step}</b>: held-out artist <b>${e.artist?.toFixed(3) ?? "–"}</b>` + (e.regularizer != null ? ` · regularizer <b>${e.regularizer.toFixed(3)}</b>` : ""); }
          if (d.detail && !d.loss) det.textContent = d.detail; draw(); };
      };
    }
    if (nodeData.name === "FSAudioOutput") {
      const isCompare = false;
      nodeType.prototype.hsSetup = function () {
        const wrap = document.createElement("div"); wrap.className = "hs-wrap"; wrap.appendChild(brand());
        const body = document.createElement("div"); body.className = "hs-meta"; body.textContent = isCompare ? "Queue to hear your hum laid over the song." : "Your song will appear here."; wrap.appendChild(body);
        this.hsPanel = addPanel(this, "humsong_player", wrap, isCompare ? 150 : 120, 460); this.hsBody = body;
      };
      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (msg) {
        onExecuted?.apply(this, arguments); const w = this.hsBody; if (!w) return; w.innerHTML = ""; w.className = "";
        const a = msg?.audio?.[0];
        if (a) {
          const url = api.apiURL(`/view?filename=${encodeURIComponent(a.filename)}&type=${a.type}&subfolder=${encodeURIComponent(a.subfolder || "")}&t=${Date.now()}`);
          let tags = "";
          if (isCompare) { const c = (() => { try { return JSON.parse(msg.humsong_compare?.[0] || "{}"); } catch { return {}; } })(); tags = `<span class="hs-tag">hum over song</span><span class="hs-tag">${c.hum_seconds ?? "?"} s of hum</span><span class="hs-tag">${c.hum_db ?? ""} dB</span><span class="hs-tag">at ${c.offset ?? 0} s</span>`; }
          else { const info = (() => { try { return JSON.parse(msg.humsong_info?.[0] || "{}"); } catch { return {}; } })();
            tags = info.seconds != null ? `<span class="hs-tag">${info.seconds} s</span><span class="hs-tag">${info.hum ? esc(String(info.melody)) : "score " + esc(String(info.score_mode))}</span>${info.hum ? `<span class="hs-tag">hum influence ${info.hum_influence}</span>` : ""}<span class="hs-tag">seed ${info.seed}</span><span class="hs-tag">${esc(String(info.sampler))} · ${esc(String(info.scheduler))} · ${info.steps}</span>` : `<span class="hs-tag">song</span>`; }
          w.appendChild(playerBlock(url, tags));
          const dl = document.createElement("a"); dl.href = url; dl.download = a.filename; dl.className = "hs-btn"; dl.style.marginTop = "8px"; dl.textContent = isCompare ? "⬇ Download comparison" : "⬇ Download FLAC"; w.appendChild(dl);
        }
        const score = msg?.humsong_score?.[0];
        if (score) { const pre = document.createElement("div"); pre.className = "hs-score"; let inHum = score.includes("% ---- your hum ends here");
          pre.innerHTML = score.split("\n").map(l => { if (l.startsWith("% ---- your hum ends")) { inHum = false; return `<span class="mark">${esc(l)}</span>`; } return inHum ? `<span class="hum">${esc(l)}</span>` : esc(l); }).join("\n");
          const cap = document.createElement("div"); cap.className = "hs-meta"; cap.innerHTML = "Score — <b>mint</b> is your hum, the rest is what the model wrote."; w.append(cap, pre); }
        const need = (a ? 150 : 40) + (score ? 300 : 0) + 70; this.hsPanel?.hsResize(need);
      };
    }
  },
});
