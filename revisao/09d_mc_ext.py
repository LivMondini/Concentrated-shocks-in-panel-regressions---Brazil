# revisao/09d_mc_ext.py
# uso (da raiz): python revisao\09d_mc_ext.py gate
#                python revisao\09d_mc_ext.py obs|shift|param [--reps R] [--horizontes 5]
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
import monte_carlo_perm as mp
from monte_carlo_lp import caminho_choque, n_efetivo

NIV = (1, 2.5, 5, 10)
ZN = {1: 2.575829303549, 2.5: 2.241402727604, 5: 1.959963984540, 10: 1.644853626951}

def gera_z(N, n_per, rho, rng):                       # copia literal de monte_carlo_lp.py
    z = np.empty((N, n_per)); z[:, 0] = rng.standard_normal(N)
    for s in range(1, n_per):
        z[:, s] = rho * z[:, s - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal(N)
    return (z - z.mean(0)) / z.std(0)

def simula(shock, h, N, sig_lam, sig_e, rho, phi, gamma, rng):   # = simula_painel com T = len(shock)
    T = len(shock); n_per = T + h
    zl = gera_z(N, n_per, rho, rng)
    lam = np.empty(n_per); lam[0] = sig_lam * rng.standard_normal()
    for s in range(1, n_per):
        lam[s] = phi * lam[s - 1] + np.sqrt(1 - phi ** 2) * sig_lam * rng.standard_normal()
    A = lam[None, :] * zl + sig_e * rng.standard_normal((N, n_per))
    C = np.concatenate([np.zeros((N, 1)), np.cumsum(A, axis=1)], axis=1)
    t = np.arange(T); y = C[:, t + h + 1] - C[:, t]
    zt = zl[:, :T]
    return y + gamma * shock[None, :] * zt, zt

class Pre:                                            # copia de 09b_mc_rapido.py
    def __init__(self, y, z):
        self.N = z.shape[0]; self.z = z
        self.yd = y - y.mean(1, keepdims=True); self.zd = z - z.mean(1, keepdims=True)
        zd, yd = self.zd, self.yd
        self.cz, self.cy = zd.sum(0), yd.sum(0)
        self.Zs, self.zz = z.sum(0), (z * z).sum(0)
        self.q, self.zdzd = (zd * z).sum(0), (zd * zd).sum(0)
        self.zdy, self.zy = (zd * yd).sum(0), (z * yd).sum(0)
        self.c22, self.c2y = self.zdzd.sum(), self.zdy.sum()

def fit_h(pre, S):                                    # fit_lote com T = S.shape[1]; devolve beta e scores h_t
    N, z, zd, yd = pre.N, pre.z, pre.zd, pre.yd; P, T = S.shape
    A = S - S.mean(1, keepdims=True)
    M = (z @ S.T) / T
    Msum, MM = M.sum(0), (M * M).sum(0)
    ZM, DM, YM = (z.T @ M).T, (zd.T @ M).T, (yd.T @ M).T
    SZ = S * pre.Zs[None, :]
    XtX = np.empty((P, 3, 3))
    XtX[:, 0, 0] = N * (A * A).sum(1); XtX[:, 0, 1] = A @ pre.cz
    XtX[:, 0, 2] = (A * SZ).sum(1) - Msum * A.sum(1)
    XtX[:, 1, 1] = pre.c22; XtX[:, 1, 2] = S @ pre.q
    XtX[:, 2, 2] = (S * S) @ pre.zz - T * MM
    XtX[:, 1, 0], XtX[:, 2, 0], XtX[:, 2, 1] = XtX[:, 0, 1], XtX[:, 0, 2], XtX[:, 1, 2]
    Xty = np.stack([A @ pre.cy, np.full(P, pre.c2y), S @ pre.zy], axis=1)
    e3 = np.broadcast_to(np.array([0.0, 0.0, 1.0])[None, :, None], (P, 3, 1))
    sol = np.linalg.solve(XtX, np.concatenate([Xty[:, :, None], e3], axis=2))
    b, r = sol[:, :, 0], sol[:, :, 1]
    b1, b2, b3 = b[:, 0:1], b[:, 1:2], b[:, 2:3]
    Ssum = SZ - Msum[:, None]
    g1 = A * (pre.cy[None, :] - b1 * N * A - b2 * pre.cz[None, :] - b3 * Ssum)
    g2 = pre.zdy[None, :] - b1 * A * pre.cz[None, :] - b2 * pre.zdzd[None, :] - b3 * (S * pre.q[None, :] - DM)
    zu = pre.zy[None, :] - b1 * A * pre.Zs[None, :] - b2 * pre.q[None, :] - b3 * (S * pre.zz[None, :] - ZM)
    mu = YM - b1 * A * Msum[:, None] - b2 * DM - b3 * (S * ZM - MM[:, None])
    hh = r[:, 0:1] * g1 + r[:, 1:2] * g2 + r[:, 2:3] * (S * zu - mu)
    return b[:, 2], hh

def se_from_h(hh, L, N, T):                           # Bartlett, pesos 1 - l/(L+1); L = 0 e cluster no tempo
    V = (hh * hh).sum(1)
    for l in range(1, L + 1):
        V = V + 2 * (1 - l / (L + 1)) * (hh[:, l:] * hh[:, :-l]).sum(1)
    return np.sqrt(np.maximum(V * (N * T) / (N * T - 3 - N), 1e-30))

_cache = {}
def crit_fixedb(T, L, N, nsim=200000):
    """Valores criticos de |t| com scores gaussianos iid de comprimento T, mesmo Bartlett e mesma correcao."""
    key = (T, L, N)
    if key in _cache: return _cache[key]
    rng = np.random.default_rng(2026); corr = N * T / (N * T - 3 - N); ts = []
    for _ in range(nsim // 50000):
        x = rng.standard_normal((50000, T)); V = (x * x).sum(1)
        for l in range(1, L + 1): V = V + 2 * (1 - l / (L + 1)) * (x[:, l:] * x[:, :-l]).sum(1)
        ts.append(np.abs(x.sum(1)) / np.sqrt(np.maximum(V * corr, 1e-30)))
    t = np.concatenate(ts)
    _cache[key] = {nv: float(np.quantile(t, 1 - nv / 100)) for nv in NIV}
    return _cache[key]

def replica(shock, h, N, sig_lam, sig_e, rho, phi, gamma, P, seed):
    rng = np.random.default_rng(seed)
    y, zt = simula(shock, h, N, sig_lam, sig_e, rho, phi, gamma, rng)
    L = h + 1
    perms = np.array([mp.block_perm(shock, L, rng) for _ in range(P)])
    b, hh = fit_h(Pre(y, zt), np.vstack([shock[None, :], perms]))
    T = y.shape[1]; se1 = se_from_h(hh, L, N, T)
    t0, bs, ts = b[0] / se1[0], b[1:], b[1:] / se1[1:]
    pc = (1 + np.sum(np.abs(bs) >= abs(b[0]))) / (P + 1)
    ps = (1 + np.sum(np.abs(ts) >= abs(t0))) / (P + 1)
    se2 = se_from_h(hh[:1], 2 * h + 2, N, T)[0]; se0 = se_from_h(hh[:1], 0, N, T)[0]
    return np.array([abs(t0), abs(b[0] / se2), abs(b[0] / se0), pc, ps])

def agrega(r, crit):
    row = {}
    for nv in NIV:
        rej = {"dk": r[:, 0] > ZN[nv], "dk2": r[:, 1] > ZN[nv], "cluster": r[:, 2] > ZN[nv],
               "fixedb": r[:, 0] > crit[nv], "perm_coef": r[:, 3] < nv / 100, "perm_stud": r[:, 4] < nv / 100}
        for k, v in rej.items():
            v = v.astype(float); row[f"{k}_{nv:g}"] = v.mean(); row[f"{k}_{nv:g}_epmc"] = v.std(ddof=1) / np.sqrt(len(v))
    return row

def rodar_celula(shock, h, N, rho, phi, gamma, P, reps, seed0, jobs):
    r = np.array(Parallel(n_jobs=jobs, batch_size=64)(
        delayed(replica)(shock, h, N, 0.02, 0.10, rho, phi, gamma, P, seed0 + i) for i in range(reps)))
    T = len(shock); crit = crit_fixedb(T, h + 1, N)
    row = dict(n_efetivo=n_efetivo(shock - shock.mean()), T=T, N=N, rho=rho, phi=phi, gamma=gamma, h=h, reps=reps, P=P)
    row.update(agrega(r, crit)); row.update({f"crit_fixedb_{nv:g}": crit[nv] for nv in NIV})
    return row

def shock_observado():
    import placebos as pb
    idx = pb.V[1:]
    assert len(idx) == 54 and (np.diff(idx) == 1).all(), "serie observada inesperada"
    return pb.S0[idx].astype(float)

def gate(a):
    m09b = importlib.import_module("09b_mc_rapido")
    from monte_carlo_lp import simula_painel as sp0, estima
    bad, mx, dy_max, mse, mb = 0, 0.0, 0.0, 0.0, 0.0
    for cen in ("iid", "calibrada"):
        sh = caminho_choque(cen)
        for h in (3, 7):
            for g in (0.0, 0.04):
                for i in range(5):
                    o = m09b.replica(sh, h, 300, .02, .10, .9, 0.0, g, 199, a.seed + i)
                    n = replica(sh, h, 300, .02, .10, .9, 0.0, g, 199, a.seed + i)
                    mx = max(mx, abs(o[0] - n[0])); bad += (o[1] != n[3]) or (o[2] != n[4])
            y0, X0 = sp0(sh, h, 300, .02, .10, .9, 0.0, 0.04, np.random.default_rng(a.seed))
            y1, zt1 = simula(sh, h, 300, .02, .10, .9, 0.0, 0.04, np.random.default_rng(a.seed))
            dy_max = max(dy_max, np.abs(y0 - y1).max())
            est = estima(y0, X0, [("a", h + 1), ("b", 2 * h + 2), ("c", 0)], h, 2, np.random.default_rng(1))
            b, hh = fit_h(Pre(y1, zt1), sh[None, :]); N_, T_ = y1.shape; corr = N_ * T_ / (N_ * T_ - 3 - N_)
            mb = max(mb, abs(b[0] - est["beta"]))
            for nome, L in (("a", h + 1), ("b", 2 * h + 2), ("c", 0)):
                mse = max(mse, abs(se_from_h(hh, L, N_, T_)[0] / np.sqrt(corr) / est[nome] - 1))
    print(f"[gate 1] 40 replicas vs 09b.replica: p diferentes = {bad} | max |dif| |t| = {mx:.2e}")
    print(f"[gate 2] DGP: max |dif y| = {dy_max:.2e} | beta vs estima: {mb:.2e} | EP DK(h+1), DK(2h+2), cluster vs estima: max rel = {mse:.2e}")
    for L in (0, 4, 8):
        c = crit_fixedb(54, L, 300); print(f"[fixed-b] T=54 L={L}: crit |t| 1%/2,5%/5%/10% = " + " / ".join(f"{c[nv]:.3f}" for nv in NIV))
    sm = replica(np.tile(caminho_choque('iid'), 2), 5, 100, .02, .10, .9, 0.0, 0.0, 19, a.seed)
    print(f"[T=108] teste de fumaca: {np.round(sm, 3)} (finito: {bool(np.isfinite(sm).all())})")
    if bad or mx > 1e-8 or dy_max > 1e-12 or mse > 1e-8 or mb > 1e-10 or not np.isfinite(sm).all():
        sys.exit("Gate falhou: cole estas linhas.")
    print("[gate] OK")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["gate", "obs", "shift", "param"])
    ap.add_argument("--reps", type=int, default=None); ap.add_argument("--P", type=int, default=999)
    ap.add_argument("--horizontes", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--seed", type=int, default=2026); ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    if a.modo == "gate": return gate(a)
    reps = a.reps or (5000 if a.modo == "param" else 10000)
    cells = []                                         # (rotulos, choque, h, N, rho, phi, gamma)
    if a.modo == "obs":
        sh = shock_observado()
        cells = [(dict(cenario="observada"), sh, h, 300, .9, phi, g) for h in a.horizontes for g in (0.0, 0.04) for phi in (0.0, 0.5)]
    elif a.modo == "shift":
        base = caminho_choque("calibrada")
        cells = [(dict(cenario="calibrada_deslocada", d=d, alinh=(39 + d) % (h + 1)), np.roll(base, d), h, 300, .9, 0.0, 0.0)
                 for h in a.horizontes for d in range(h + 1)]
    else:
        obs, cal = shock_observado(), caminho_choque("calibrada")
        var = [("rho0.5", dict(rho=.5)), ("rho0.95", dict(rho=.95)), ("N100", dict(N=100)), ("N1000", dict(N=1000)), ("T108", dict(tile=2))]
        for cn, sh0 in (("observada", obs), ("calibrada", cal)):
            for vn, kw in var:
                sh = np.tile(sh0, kw.get("tile", 1))
                cells.append((dict(cenario=cn, variante=vn), sh, 5, kw.get("N", 300), kw.get("rho", .9), 0.0, 0.0))
    off = {"obs": 0, "shift": 1000, "param": 2000}[a.modo]
    out = OUT / f"09d_mc_{a.modo}_reps{reps}.csv"; rows = []
    for k, (lab, sh, h, N, rho, phi, g) in enumerate(cells):
        t0 = time.time()
        row = dict(modo=a.modo, **lab, **rodar_celula(sh, h, N, rho, phi, g, a.P, reps, a.seed + 100000 * (off + k), a.jobs))
        rows.append(row); pd.DataFrame(rows).to_csv(out, index=False)
        print(f"{lab} h={h} N={N} rho={rho} phi={phi} gamma={g}: DK {row['dk_5']:.3f} | DK2 {row['dk2_5']:.3f} | cluster {row['cluster_5']:.3f} | "
              f"fixed-b {row['fixedb_5']:.3f} | perm {row['perm_coef_5']:.3f} | stud {row['perm_stud_5']:.3f} (EPMC {row['perm_stud_5_epmc']:.4f}) {time.time() - t0:.0f}s", flush=True)
    if a.modo == "shift":
        df = pd.DataFrame(rows)
        for h, g in df.groupby("h"):
            s = g.set_index("d")["perm_stud_5"]; dif = s.loc[0] - s.median()
            print(f"[criterio 9b] h={h}: stud 5% em d=0 = {s.loc[0]:.4f} | mediana(d) = {s.median():.4f} | min {s.min():.4f} max {s.max():.4f} | "
                  f"dif = {100 * dif:+.2f} pp -> {'compativel com alinhamento' if dif >= 0.01 else 'alinhamento nao explica'}")

if __name__ == "__main__":
    main()