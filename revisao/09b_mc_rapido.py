# revisao/09b_mc_rapido.py -- Monte Carlo vetorizado (DK, perm. nao studentizada, perm. studentizada)
# uso (da raiz): python revisao\09b_mc_rapido.py --reps 20 --cenarios iid --horizontes 3        (teste + gate)
#                python revisao\09b_mc_rapido.py --reps 10000 --gammas 0 --cenarios calibrada   (final)
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"     # 1 thread de BLAS por processo
import sys, time, argparse, importlib
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "revisao" / "saidas"
LOGS = ROOT / "revisao" / "logs"
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import monte_carlo_perm as mp
from monte_carlo_lp import T, caminho_choque, simula_painel, n_efetivo

class Pre:
    """Estatisticas que nao dependem do choque: calculadas 1x por replica."""
    def __init__(self, y, z):
        self.N = z.shape[0]; self.z = z
        self.yd = y - y.mean(1, keepdims=True); self.zd = z - z.mean(1, keepdims=True)
        zd, yd = self.zd, self.yd
        self.cz, self.cy = zd.sum(0), yd.sum(0)
        self.Zs, self.zz = z.sum(0), (z * z).sum(0)
        self.q, self.zdzd = (zd * z).sum(0), (zd * zd).sum(0)
        self.zdy, self.zy = (zd * yd).sum(0), (z * yd).sum(0)
        self.c22, self.c2y = self.zdzd.sum(), self.zdy.sum()

def fit_lote(pre, S, L):
    """S: (P,T) series de choque. Devolve beta de Shock*z e EP DK Bartlett(bw=L) com N/(N-k-G)."""
    N, z, zd, yd = pre.N, pre.z, pre.zd, pre.yd; P = S.shape[0]
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
    sol = np.linalg.solve(XtX, np.concatenate([Xty[:, :, None], e3], axis=2))   # [beta | linha 3 de (X'X)^-1]
    b, r = sol[:, :, 0], sol[:, :, 1]
    b1, b2, b3 = b[:, 0:1], b[:, 1:2], b[:, 2:3]
    Ssum = SZ - Msum[:, None]
    g1 = A * (pre.cy[None, :] - b1 * N * A - b2 * pre.cz[None, :] - b3 * Ssum)
    g2 = pre.zdy[None, :] - b1 * A * pre.cz[None, :] - b2 * pre.zdzd[None, :] - b3 * (S * pre.q[None, :] - DM)
    zu = pre.zy[None, :] - b1 * A * pre.Zs[None, :] - b2 * pre.q[None, :] - b3 * (S * pre.zz[None, :] - ZM)
    mu = YM - b1 * A * Msum[:, None] - b2 * DM - b3 * (S * ZM - MM[:, None])
    h = r[:, 0:1] * g1 + r[:, 1:2] * g2 + r[:, 2:3] * (S * zu - mu)   # r'g_t
    V33 = (h * h).sum(1)
    for l in range(1, L + 1):
        V33 += 2 * (1 - l / (L + 1)) * (h[:, l:] * h[:, :-l]).sum(1)
    return b[:, 2], np.sqrt(np.maximum(V33 * (N * T) / (N * T - 3 - N), 1e-30))

def replica(shock, h, N, sig_lam, sig_e, rho, phi, gamma, P, seed):
    rng = np.random.default_rng(seed)
    y, X = simula_painel(shock, h, N, sig_lam, sig_e, rho, phi, gamma, rng)   # mesmo consumo de RNG
    L = h + 1
    perms = np.array([mp.block_perm(shock, L, rng) for _ in range(P)])        # mesma ordem de sorteio
    b, se = fit_lote(Pre(y, X[:, :, 1]), np.vstack([shock[None, :], perms]), L)
    b0, t0, bs, ts = b[0], b[0] / se[0], b[1:], b[1:] / se[1:]
    pc = (1 + np.sum(np.abs(bs) >= abs(b0))) / (P + 1)
    ps = (1 + np.sum(np.abs(ts) >= abs(t0))) / (P + 1)
    return (abs(t0), pc, ps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10000); ap.add_argument("--P", type=int, default=999)
    ap.add_argument("--N", type=int, default=300); ap.add_argument("--sig_lam", type=float, default=0.02)
    ap.add_argument("--sig_e", type=float, default=0.10); ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--phi", type=float, default=0.0)
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.04])
    ap.add_argument("--horizontes", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--cenarios", nargs="+", default=["iid", "calibrada", "extrema"])
    ap.add_argument("--seed", type=int, default=2026);ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--sem_gate", action="store_true")
    a = ap.parse_args()

    if not a.sem_gate:    # gate: igual a versao nao vetorizada (09_mc_unificado), replica por replica
        old = importlib.import_module("09_mc_unificado"); bad, mx = 0, 0.0
        for cen in ("iid", "calibrada"):
            sh = caminho_choque(cen)
            for h in (3, 7):
                for g in (0.0, 0.04):
                    for i in range(5):
                        sd = a.seed + i
                        o = old.replica(sh, h, a.N, a.sig_lam, a.sig_e, a.rho, a.phi, g, 199, sd)
                        n = replica(sh, h, a.N, a.sig_lam, a.sig_e, a.rho, a.phi, g, 199, sd)
                        mx = max(mx, abs(o[0] - n[0])); bad += (o[1] != n[1]) or (o[2] != n[2])
        print(f"[gate] 40 replicas: p-valores diferentes = {bad} | max |dif| em |t| observado = {mx:.2e}")
        if bad or mx > 1e-8: sys.exit("Gate falhou: cole esta linha.")

    tag = "_".join(a.cenarios) + "_g" + "-".join(map(str, a.gammas)) + f"_phi{a.phi}"
    out = OUT / f"09b_mc_{tag}.csv"; rows, k = [], 0
    for cen in a.cenarios:
        shock = caminho_choque(cen)
        for h in a.horizontes:
            for g in a.gammas:
                t0 = time.time(); seeds = [a.seed + 100000 * k + i for i in range(a.reps)]
                r = np.array(Parallel(n_jobs=a.jobs, batch_size=64)(
                    delayed(replica)(shock, h, a.N, a.sig_lam, a.sig_e, a.rho, a.phi, g, a.P, s) for s in seeds))
                row = dict(cenario=cen, h=h, gamma=g, phi=a.phi, n_efetivo=n_efetivo(shock - shock.mean()),
                           reps=a.reps, P=a.P)
                regras = {"dk": {5: r[:, 0] > 1.959964, 10: r[:, 0] > 1.644854},
                          "perm_coef": {5: r[:, 1] < .05, 10: r[:, 1] < .10},
                          "perm_stud": {5: r[:, 2] < .05, 10: r[:, 2] < .10}}
                for nome, d in regras.items():
                    for nivel, rej in d.items():
                        v = rej.astype(float)
                        row[f"{nome}_{nivel}"] = v.mean(); row[f"{nome}_{nivel}_epmc"] = v.std(ddof=1) / np.sqrt(len(v))
                dd = regras["dk"][5].astype(float) - regras["perm_stud"][5].astype(float)
                row["dif_dk_menos_stud_5"], row["dif_epmc"] = dd.mean(), dd.std(ddof=1) / np.sqrt(len(dd))
                rows.append(row); pd.DataFrame(rows).to_csv(out, index=False); k += 1
                print(f"{cen} h={h} gamma={g}: DK {row['dk_5']:.3f} | perm {row['perm_coef_5']:.3f} | "
                      f"perm stud {row['perm_stud_5']:.3f} (EP MC {row['perm_stud_5_epmc']:.4f}) {time.time() - t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()