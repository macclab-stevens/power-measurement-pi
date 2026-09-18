import csv, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# pick richest run (most perop rows), fallback latest
def _rows(p):
    try:
        return list(csv.DictReader(open(p)))
    except Exception:
        return []

best_dir = None; best_score = -1
for d in sorted(glob.glob("runs/*/")):
    s = len([r for r in _rows(d + "perop_power.csv") if r.get("SKIPPED") != "True"])
    if s > best_score:
        best_score, best_dir = s, d
outdir = best_dir if best_dir else sorted(glob.glob("runs/*/"))[-1]
print("RUN:", outdir)

perop = [r for r in _rows(outdir + "perop_power.csv") if r["P_delta_W"] not in (None, "")]
samples = _rows(outdir + "samples.csv")
ctx = _rows(outdir + "context.csv")

t = np.array([float(r["t_ms"]) for r in samples])
I = np.array([float(r["I_A"]) for r in samples])
V = np.array([float(r["V_V"]) for r in samples])
P = I * V
ph = np.array([r["phase"] for r in samples])
wid = np.array([int(r["window_id"]) for r in samples])

fig, axs = plt.subplots(2, 2, figsize=(13, 9))

# 1) P over time colored by phase
ax = axs[0][0]
order = sorted(set(ph), key=lambda x: ["idle", "warmup", "forward", "prefill", "decode", "bench", "inference"].index(x) if x in ("idle","warmup","forward","prefill","decode","bench","inference") else 99)
for pcol in order:
    m = ph == pcol
    ax.scatter(t[m], P[m], s=4, alpha=0.35, label=pcol.split(":")[0])
# moving average over all
if len(P) > 200:
    k = 200
    ma = np.convolve(P, np.ones(k) / k, mode="same")
    ax.plot(t, ma, color="black", lw=1.6, label="moving avg")
ax.set_xlabel("t_ms"); ax.set_ylabel("VDD_CORE P (W)"); ax.set_title("Input power over run"); ax.legend(fontsize=7)

# 2) temp + freq
ax = axs[0][1]
ct = np.array([float(r["t_ms"]) for r in ctx])
tc = np.array([float(r["temp_C"]) for r in ctx if r.get("temp_C")])
freq = np.array([float(r["arm_MHz"])/1000 for r in ctx if r.get("arm_MHz")])
ax.plot(ct, tc, "-o", ms=3, color="tab:red", label="temp C")
ax.set_ylabel("temp (C)", color="tab:red"); ax.set_xlabel("t_ms")
ax2 = ax.twinx()
ax2.plot(ct, freq, "-s", ms=3, color="tab:blue", label="arm GHz")
ax2.set_ylabel("arm freq (GHz)", color="tab:blue")
ax.set_title("thermal / DVFS over run")

# 3) per-op power vs per-call time (colored by E)
ax = axs[1][0]
tp = np.array([float(r["t_per_call_ms"]) for r in perop])
pd = np.array([float(r["P_delta_W"]) for r in perop])
E = np.array([float(r["E_per_call_mJ"]) for r in perop])
sc = ax.scatter(tp, pd, c=E, cmap="viridis", s=40)
ax.set_xscale("log"); ax.set_xlabel("t_per_call (ms, log)"); ax.set_ylabel("P_delta (W)")
ax.set_title("per-op power vs cost"); plt.colorbar(sc, ax=ax, label="E mJ")

# 4) per-op power by op class
ax = axs[1][1]
classes = ["Conv2d", "BatchNorm2d", "SiLU", "Concat", "Identity", "Upsample", "ConvTranspose2d", "MaxPool2d", "Linear", "LayerNorm", "Embedding", "MultiheadAttention"]
pp = {c: [float(r["P_delta_W"]) for r in perop if r["class"] == c] for c in classes}
pp = {k: v for k, v in pp.items() if v}
pos = np.arange(len(pp))
ax.bar(pos, [np.median(v) for v in pp.values()], yerr=[np.std(v) for v in pp.values()], capsize=4, alpha=0.7)
ax.set_xticks(pos); ax.set_xticklabels(pp.keys(), rotation=20)
ax.set_ylabel("median P_delta (W)"); ax.set_title("per-op power by class")

plt.tight_layout()
png = "/tmp/trend_%s.png" % os.path.basename(outdir.rstrip("/"))
plt.savefig(png, dpi=100)
print("saved", png)

# numeric digest
print("\n== power phases (VDD_CORE) ==")
for pcol in ("idle", "warmup", "forward", "prefill", "decode", "bench", "inference"):
    m = ph == pcol
    if m.any():
        print(f"  {pcol:10s} n={m.sum():6d} mean={P[m].mean():.3f}W  max={P[m].max():.3f}W")

print("\n== thermal/DVFS ==")
if tc.size: print(f"  temp  start={tc[0]:.1f}C end={tc[-1]:.1f}C peak={tc.max():.1f}C")
if freq.size: print(f"  arm   start={freq[0]:.2f}GHz end={freq[-1]:.2f}GHz")

print("\n== per-op scale (top by E) ==")
for r in sorted(perop, key=lambda x: -float(x["E_per_call_mJ"]))[:6]:
    print(f"  {r['class']:8s} t={float(r['t_per_call_ms']):7.2f}ms P_delta={float(r['P_delta_W']):.3f}W E={float(r['E_per_call_mJ']):7.2f}mJ")

# MACs correlation for convs (from config_json)
def macs_of(r):
    import json
    try:
        c = json.loads(r["config_json"])
    except Exception:
        return None
    if r["class"] != "Conv2d": return None
    try:
        oc, k, h, w = c["out_channels"], c["kernel_size"], c["stride"], c["padding"]
    except Exception:
        return None
    return int((h or 0) * (w or 0))  # rough proxy: use H*W from shapes instead
conv = [r for r in perop if r["class"] == "Conv2d"]
if conv:
    pdc = np.array([float(r["P_delta_W"]) for r in conv])
    tc2 = np.array([float(r["t_per_call_ms"]) for r in conv])
    print(f"  {len(conv)} convs | P_delta range {pdc.min():.2f}-{pdc.max():.2f}W | "
          f"t range {tc2.min():.2f}-{tc2.max():.2f}ms | corr(P,t)= {np.corrcoef(pdc, tc2)[0,1]:.2f}")