# revisao/09e_fixedb_corrigido.py   (da raiz: python revisao\09e_fixedb_corrigido.py gate | obs | shift | param)
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
import sys, time, argparse, importlib
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "revisao" / "saidas"
LOGS = ROOT / "revisao" / "logs"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "revisao"))
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from monte_carlo_lp import caminho_choque
d9 = importlib.import_module("09d_mc_ext")
NIV = (1, 2.5, 5, 10); Z5 = 1.959963984540
_c = {}
def crit_fb(T, L, N, nsim=200000):
    if (T, L, N) in _c: return _c[(T, L, N)]
    rng = np.random.default_rng(2026); corr = N * T / (N * T - 3 - N); ts = []
    for _ in range(nsim // 50000):
        x = rng.standard_normal((50000, T)); xd = x - x.mean(1, keepdims=True)
        V = (xd * xd).sum(1)
        for l in range(1, L + 1): V = V + 2 * (1 - l / (L + 1)) * (xd[:, l:] * xd[:, :-l]).sum(1)
        ts.append(np.abs(x.sum(1)) / np.sqrt(np.maximum(V * corr, 1e-30)))
    t = np.concatenate(ts)
    _c[(T, L, N)] = {nv: float(np.quantile(t, 1 - nv / 100)) for nv in NIV}
    return _c[(T, L, N)]

def t0(shock, h, N, rho, phi, gamma, seed):
    rng = np.random.default_rng(seed)
    y, zt = d9.simula(shock, h, N, .02, .10, rho, phi, gamma, rng)
    b, hh = d9.fit_h(d9.Pre(y, zt), shock[None, :])
    return abs(b[0] / d9.se_from_h(hh, h + 1, N, y.shape[1])[0])

def build_cells(modo, horizontes):
    if modo == "obs":
        sh = d9.shock_observado()
        return [(dict(cenario="observada"), sh, h, 300, .9, phi, g) for h in horizontes for g in (0.0, 0.04) for phi in (0.0, 0.5)]
    if modo == "shift":
        base = caminho_choque("calibrada")
        return [(dict(cenario="calibrada_deslocada", d=d), np.roll(base, d), h, 300, .9, 0.0, 0.0) for h in horizontes for d in range(h + 1)]
    obs, cal = d9.shock_observado(), caminho_choque("calibrada"); cells = []
    var = [("rho0.5", dict(rho=.5)), ("rho0.95", dict(rho=.95)), ("N100", dict(N=100)), ("N1000", dict(N=1000)), ("T108", dict(tile=2))]
    for cn, sh0 in (("observada", obs), ("calibrada", cal)):
        for vn, kw in var:
            cells.append((dict(cenario=cn, variante=vn), np.tile(sh0, kw.get("tile", 1)), 5, kw.get("N", 300), kw.get("rho", .9), 0.0, 0.0))
    return cells

def gate(a):
    mx = 0.0
    for cen in ("iid", "calibrada"):
        sh = caminho_choque(cen)
        for h in (3, 7):
            for g in (0.0, 0.04):
                for i in range(5):
                    o = d9.replica(sh, h, 300, .02, .10, .9, 0.0, g, 19, a.seed + i)[0]
                    mx = max(mx, abs(o - t0(sh, h, 300, .9, 0.0, g, a.seed + i)))
    print(f"[gate 1] 40 replicas: max |dif| em |t| vs 09d = {mx:.2e}")
    c5 = {}
    for L in (0, 4, 6, 8):
        c = crit_fb(54, L, 300); c5[L] = c[5]
        print(f"[fixed-b corrigido] T=54 L={L}: crit |t| 1%/2,5%/5%/10% = " + " / ".join(f"{c[nv]:.3f}" for nv in NIV))
    ok = mx < 1e-8 and 1.98 < c5[0] < 2.03 and c5[0] < c5[4] < c5[6] < c5[8] and c5[4] > 2.05
    print("[gate] " + ("OK" if ok else "<<< FALHOU (esperado: ~2,0 em L=0, crescente em L, >2,05 em L=4)"))
    if not ok: sys.exit(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["gate", "obs", "shift", "param"])
    ap.add_argument("--reps", type=int, default=None); ap.add_argument("--horizontes", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--seed", type=int, default=2026); ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    if a.modo == "gate": return gate(a)
    reps = a.reps or (5000 if a.modo == "param" else 10000)
    off = {"obs": 0, "shift": 1000, "param": 2000}[a.modo]; rows = []
    for k, (lab, sh, h, N, rho, phi, g) in enumerate(build_cells(a.modo, a.horizontes)):
        t_ = time.time(); seed0 = a.seed + 100000 * (off + k)
        t = np.array(Parallel(n_jobs=a.jobs, batch_size=64)(delayed(t0)(sh, h, N, rho, phi, g, seed0 + i) for i in range(reps)))
        crit = crit_fb(len(sh), h + 1, N)
        row = dict(modo=a.modo, **lab, h=h, N=N, rho=rho, phi=phi, gamma=g, T=len(sh), reps=reps, dk_5=float((t > Z5).mean()))
        for nv in NIV:
            r = (t > crit[nv]).astype(float); row[f"fixedb_{nv:g}"] = r.mean(); row[f"fixedb_{nv:g}_epmc"] = r.std(ddof=1) / np.sqrt(len(r)); row[f"crit_{nv:g}"] = crit[nv]
        rows.append(row); pd.DataFrame(rows).to_csv(OUT / f"09e_fixedb_{a.modo}_reps{reps}.csv", index=False)
        print(f"{lab} h={h} N={N} rho={rho} phi={phi} gamma={g}: DK {row['dk_5']:.3f} | fixed-b 5% {row['fixedb_5']:.3f} (crit {crit[5]:.3f}) {time.time() - t_:.0f}s", flush=True)
    prev = OUT / f"09d_mc_{a.modo}_reps{reps}.csv"
    if prev.exists():
        p = pd.read_csv(prev)
        if len(p) == len(rows): print(f"[confere] max |dk_5 (09e) - dk_5 (09d)| = {np.abs(p['dk_5'].to_numpy() - np.array([r['dk_5'] for r in rows])).max():.2e}  (esperado 0)")

if __name__ == "__main__":
    main()