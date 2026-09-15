"""Hum -> prosody carrier. Pitch (pYIN, 65-1000 Hz) drives a sine; loudness follows the low-passed |hum|. Words and timbre are gone; melody and timing stay."""
import os, socket, numpy as np
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), f".numba_cache_{socket.gethostname()[:8]}"))
HOP = 512
def prosody_sine(y, sr=48000):
    """y: mono float32 -> (carrier float32 [n], voiced_fraction, onset_s, singing_seconds_after_onset)"""
    import librosa
    from scipy.signal import butter, sosfiltfilt
    f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=1000, sr=sr, hop_length=HOP, frame_length=4 * HOP)
    f0 = f0.copy(); idx = np.where(~np.isnan(f0))[0]
    if len(idx) == 0: return None, 0.0, -1.0, 0.0
    f0[:idx[0]] = f0[idx[0]]
    for i in range(1, len(f0)):
        if np.isnan(f0[i]): f0[i] = f0[i - 1]
    t = np.arange(len(y)) / sr; f = np.interp(t, np.arange(len(f0)) * HOP / sr, f0)
    env = np.abs(y); env = sosfiltfilt(butter(4, 30, btype="low", fs=sr, output="sos"), env); env = sosfiltfilt(butter(2, 80, btype="low", fs=sr, output="sos"), env); env = np.clip(env, 0, None)
    r = env * np.sin(2 * np.pi * np.cumsum(f) / sr); r = (r / (np.abs(r).max() + 1e-9) * 0.9).astype(np.float32)
    fr = y[:len(y) // HOP * HOP].reshape(-1, HOP); db = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-9); act = (db > db.max() - 30) & voiced[:len(db)].astype(bool)
    k = int(0.4 * sr / HOP); run = np.convolve(act.astype(int), np.ones(k, int), "valid") >= k; idx = np.where(run)[0]
    onset = float(idx[0] * HOP / sr) if len(idx) else -1.0; sing = float(act[idx[0]:].sum() * HOP / sr) if len(idx) else 0.0
    return r, float(np.nanmean(voiced)), onset, sing
