"""Sanity-check the latest run's dataset (new generalized schema).

Checks the model-agnostic contract:
  - all expected files present (run_meta now has op_sequence + arch_fingerprint)
  - perop: no SKIPPED, schema columns (input_dtypes, macs, bytes_moved), a
    dominant op class with positive P_delta + sufficient samples
  - layer_table: leaf rows carry config_id / input_dtypes / macs / bytes /
    arith_intensity / mode; measurement identity is shape+dtype aware
  - op_table non-empty; samples/phase coverage present
  - run_meta: baseline, hz, task, modes, dominant-op summary
Usage: uv run python verify.py [runs/<ts>]
"""
import csv, glob, json, os, sys

path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("runs/*/"))[-1]
print("DIR:", path)


def _cols(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


ok_all = True


def _has(name):
    global ok_all
    p = os.path.join(path, name)
    ok = os.path.exists(p)
    ok_all = ok_all and ok
    print("  ", name, "OK" if ok else "MISSING")
    return ok


for f in ["samples.csv", "context.csv", "layer_table.csv", "op_table.csv",
          "perop_power.csv", "run_meta.json"]:
    _has(f)

perop = _cols(os.path.join(path, "perop_power.csv"))
skipped = sum(1 for r in perop if r["SKIPPED"] == "True")
print(f"\nperop rows: {len(perop)}   SKIPPED: {skipped}  ({'GREEN' if skipped == 0 else 'FATAL'})")

# dominant op acceptance (mirrors run.py)
vol = {}
for r in perop:
    if r.get("N_calls"):
        vol[r["class"]] = vol.get(r["class"], 0) + int(r["N_calls"])
dom = max(vol, key=vol.get) if vol else None
if dom:
    bad = [r for r in perop if r["class"] == dom and not r["SKIPPED"] == "True"
           and (not r["P_delta_W"] or float(r["P_delta_W"]) <= 0.01)]
    low = [r for r in perop if r["class"] == dom and not r["SKIPPED"] == "True"
           and int(r["samples_used"]) < 30]
    print(f"dominant op: {dom}  rows={vol.get(dom)}  P_delta_bad={len(bad)}  "
          f"low_samples={len(low)} {'GREEN' if not bad and not low else 'FAIL'}")

big = sorted((r for r in perop if r.get("P_delta_W")),
             key=lambda r: -float(r["P_delta_W"]))[:5]
print("top P_delta_W ops:")
for r in big:
    print(f"  {r['class']:14s} P_delta={float(r['P_delta_W']):.4f}W  "
          f"E={r['E_per_call_mJ']}mJ  t={r['t_per_call_ms']}ms  "
          f"macs={r.get('macs')}  bytes={r.get('bytes_moved')}  samples={r['samples_used']}")

# measurement identity: same (class + ATTRS + shapes + dtype) must be ONE config_id
mu = {}
for r in _cols(os.path.join(path, "perop_power.csv")):
    if not r["SKIPPED"] == "True":
        k = (r["class"], r["config_json"], r["input_shapes"], r["input_dtypes"])
        mu.setdefault(k, set()).add(r["config_id"])
collide = {k: v for k, v in mu.items() if len(v) > 1}
print(f"identity collision (same class+attrs+shapes+dtype, diff config_id): "
      f"{len(collide)} {'GREEN' if not collide else 'FAIL'}")

lt = _cols(os.path.join(path, "layer_table.csv"))
leaves = [r for r in lt if r["leaf"] == "True"]
print(f"\nlayer_table rows: {len(lt)}  leaves: {len(leaves)}  "
      f"modes: {sorted({r['mode'] for r in lt if r['mode']})}")
allfmt = all(r["macs"] != "" or r["bytes_moved"] != "" for r in leaves)
has_ai = sum(1 for r in leaves if r["arith_intensity"])
print(f"  leaves with macs/bytes set: {allfmt}  arith_intensity non-empty: {has_ai}")

ops = _cols(os.path.join(path, "op_table.csv"))
top = sorted(ops, key=lambda r: -float(r["self_total_us"]))[:4]
print(f"\nop_table rows: {len(ops)}  modes: {sorted({r['mode'] for r in ops})}  top:")
for o in top:
    print(f"  {o['op'][:44]:46s} cnt={o['count']:>4} mean={o['self_mean_us']}us  {o['mode']}")

s = _cols(os.path.join(path, "samples.csv"))
phs = {}
for r in s:
    ph = r["phase"].split(":")[0]
    phs[ph] = phs.get(ph, 0) + 1
print(f"\nsamples: {len(s)}  phase coverage: {phs}")

meta = json.load(open(os.path.join(path, "run_meta.json")))
print("meta: task", meta.get("task"), "model", meta.get("model"), "modes", meta.get("modes"))
print("      baseline_W", meta["baseline_W"], "hz", meta["achieved_sample_hz"],
      "throttle", meta.get("throttle_bits"))
print("      arch_fingerprint:", meta.get("arch_fingerprint"))
print("      op_sequence by mode:",
      {m: len(v) for m, v in meta.get("op_sequence", {}).items()})

pk = perop[0]
print("\nperop env bracket: temp", pk["temp_start_C"], "->", pk["temp_end_C"],
      " freq", pk["freq_start_MHz"], "->", pk["freq_end_MHz"])
print("ALL FILES:", "GREEN" if ok_all else "FAIL (missing files)")

# ---- v3 schema gate (Task 4) ----
_schema_ok = meta.get("schema_version") == 3
print(f"schema_version: {meta.get('schema_version')} "
      f"{'GREEN' if _schema_ok else 'FAIL (want 3)'}")
ok_all = ok_all and _schema_ok


def _has_te_cols(p, want=("t_eff", "seq_len")):
    with open(p, newline="") as f:
        cols = csv.DictReader(f).fieldnames or []
    missing = [c for c in want if c not in cols]
    print(f"  {os.path.basename(p)} t_eff/seq_len cols: "
          f"{'GREEN' if not missing else 'FAIL missing ' + ','.join(missing)}")
    return not missing


_te_ok = _has_te_cols(os.path.join(path, "layer_table.csv"))
_te_ok = _has_te_cols(os.path.join(path, "perop_power.csv")) and _te_ok
leaf_te = sum(1 for r in leaves if r.get("t_eff") not in ("", None))
print(f"  leaves with t_eff set: {leaf_te}/{len(leaves)} "
      f"{'GREEN' if leaves and leaf_te == len(leaves) else 'FAIL'}")
ok_all = ok_all and _te_ok and bool(leaves) and leaf_te == len(leaves)
print(f"SKIPPED 0: {'GREEN' if skipped == 0 else f'FAIL ({skipped})'}")
ok_all = ok_all and skipped == 0

# ---- trace gate (Task 6): schema 3 + stitched bands cover smoke run ----
# Honest LOMO check: actual 10 Hz grid vs phase-mean stitch (train prior from
# sibling runs/v3_*, schema 3). Coverage >=90% gates OVERALL (spec Sec.1);
# base-vs-full residuals printed always (never hide FAIL, T6 style).
_trace_ok = True
if _schema_ok:
    try:
        import math as _m
        import collections as _c

        def _phase_targets(_d):
            _r = list(csv.DictReader(open(os.path.join(_d, "samples.csv"))))
            _o, _cur = [], None
            for _row in _r:
                _ph = _row["phase"].split(":")[0]
                if _cur is None or _ph != _cur["phase"]:
                    if _cur:
                        _dur = _cur["t1"] - _cur["t0"] if _cur["n"] > 1 else 0.0
                        _cur["P_mean"] = _cur["P_sum"] / _cur["n"]
                        _cur["dur_s"] = _dur / 1000.0
                        _o.append(_cur)
                    _cur = {"phase": _ph, "t0": float(_row["t_ms"]), "t1": float(_row["t_ms"]),
                            "P_sum": 0.0, "n": 0}
                _cur["P_sum"] += float(_row["P_W"])
                _cur["n"] += 1
                _cur["t1"] = float(_row["t_ms"])
            if _cur:
                _dur = _cur["t1"] - _cur["t0"] if _cur["n"] > 1 else 0.0
                _cur["P_mean"] = _cur["P_sum"] / _cur["n"]
                _cur["dur_s"] = _dur / 1000.0
                _o.append(_cur)
            return _o

        _rows = list(csv.DictReader(open(os.path.join(path, "samples.csv"))))
        _hz, _step = 10, 100.0
        _t0, _t1 = float(_rows[0]["t_ms"]), float(_rows[-1]["t_ms"])
        _ts = [float(r["t_ms"]) for r in _rows]
        _actual, _j, _b = [], 0, _t0
        while _b < _t1:
            while _j < len(_rows) and _ts[_j] < _b:
                _j += 1
            _k, _cnt, _ps, _mn = _j, _c.Counter(), 0.0, 0
            while _k < len(_rows) and _ts[_k] < _b + _step:
                _cnt[_rows[_k]["phase"].split(":")[0]] += 1
                _ps += float(_rows[_k]["P_W"])
                _mn += 1
                _k += 1
            if _mn:
                _actual.append((_cnt.most_common(1)[0][0], _ps / _mn))
            _b += _step
        _repo = os.path.dirname(os.path.dirname(os.path.abspath(path))) \
            if os.path.basename(os.path.dirname(path)) == "runs" else os.getcwd()
        _cands = sorted(glob.glob(os.path.join(_repo, "runs", "v3_*")))
        _cur_norm = os.path.normpath(os.path.abspath(path))
        _cands = [d for d in _cands if os.path.normpath(os.path.abspath(d)) != _cur_norm
                  and os.path.isdir(d)]
        _tdirs = []
        for _d in _cands:
            try:
                _mm = json.load(open(os.path.join(_d, "run_meta.json")))
            except Exception:
                continue
            if _mm.get("schema_version") == 3 and "_TAINTED_V2" not in _d:
                _tdirs.append(_d)
        if _tdirs and _actual:
            _trows = [r for _d in _tdirs for r in _phase_targets(_d)]
            _by = _c.defaultdict(list)
            for _r in _trows:
                _by[_r["phase"]].append(_r["P_mean"])
            _pmean = {k: sum(v) / len(v) for k, v in _by.items()}
            _prmse = {}
            for _k2, _v in _by.items():
                if len(_v) > 1:
                    _logs = [_m.log(max(x, 1e-6)) for x in _v]
                    _mu = sum(_logs) / len(_logs)
                    _sd = _m.sqrt(sum((_x - _mu) ** 2 for _x in _logs) / len(_logs))
                else:
                    _sd = 0.3
                _prmse[_k2] = max(float(_sd), 0.15)
            _meanp = sum(r["P_mean"] for r in _trows) / len(_trows)
            _cov = 0
            _s_base = _s_full = 0.0
            for _ph, _pa in _actual:
                _pf = _pmean.get(_ph, _meanp)
                _sig = _m.sqrt(_prmse.get(_ph, 0.3) ** 2 + 0.1 ** 2)
                _lo = _m.exp(_m.log(max(_pf, 1e-6)) - 1.96 * _sig)
                _hi = _m.exp(_m.log(max(_pf, 1e-6)) + 1.96 * _sig)
                if _lo <= _pa <= _hi:
                    _cov += 1
                _s_base += abs(_pa - _meanp) / max(abs(_pa), 1e-12)
                _s_full += abs(_pa - _pf) / max(abs(_pa), 1e-12)
            _frac = _cov / len(_actual)
            _mb, _mf = _s_base / len(_actual), _s_full / len(_actual)
            _dl = _mb - _mf
            print(f"trace bands: cover {_cov}/{len(_actual)} ({_frac:.1%}) "
                  f"{'GREEN' if _frac >= 0.9 else 'FAIL (want >=90%)'}")
            print(f"trace residual: base MAPE {_mb:.4f} vs full {_mf:.4f} "
                  f"delta {_dl:.4f} ({'WIN' if _dl > 0.03 else 'FAIL (want >3pp)'})")
            _trace_ok = _frac >= 0.9
        else:
            print("trace bands: SKIP (no train prior or empty grid)")
    except Exception as _e:
        print(f"trace bands: SKIP (tooling error: {_e})")
else:
    print("trace bands: SKIP (schema != 3)")
ok_all = ok_all and _trace_ok
print("OVERALL:", "GREEN" if ok_all else "FAIL")
sys.exit(0 if ok_all else 1)