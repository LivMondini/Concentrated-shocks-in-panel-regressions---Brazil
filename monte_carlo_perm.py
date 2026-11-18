"""monte_carlo_perm.py -- tamanho e poder do teste de permutacao em blocos (pooled).
Versao paralela (joblib). Uso:
    python monte_carlo_perm.py --reps 20 --P 49 --cenarios iid --horizontes 3   (teste)
    python monte_carlo_perm.py --reps 300 --P 199                               (final)
Saida: monte_carlo_perm.csv
"""
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from monte_carlo_lp import T, caminho_choque, simula_painel, n_efetivo

ROOT = Path(__file__).resolve().parent

def beta_gamma(y, shock_vec, zt):
    """Coeficiente de Shock*z com efeito fixo de firma. shock_vec (T,), zt (N,T)."""
    N = y.shape[0]
    X = np.stack([np.broadcast_to(shock_vec[None, :], (N, T)), zt,
                  shock_vec[None, :] * zt], axis=2)
    yd = y - y.mean(1, keepdims=True)
    Xd = X - X.mean(1, keepdims=True)
    XtX = np.einsum("itk,itl->kl", Xd, Xd)
    Xty = np.einsum("itk,it->k", Xd, yd)
    return np.linalg.solve(XtX, Xty)[2]


def block_perm(v, L, rng):
    """Embaralha todos os blocos de tamanho L (o ultimo pode ser mais curto)."""
    blocks = [v[i:i + L] for i in range(0, len(v), L)]
    return np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])


def uma_replica(shock, h, N, sig_lam, sig_e, rho, phi, gamma, P, seed):
    """Uma replica do Monte Carlo; independente, roda em paralelo."""
    rng = np.random.default_rng(seed)
    y, X = simula_painel(shock, h, N, sig_lam, sig_e, rho, phi, gamma, rng)
    zt = X[:, :, 1]
    b0 = abs(beta_gamma(y, shock, zt))
    sims = np.array([abs(beta_gamma(y, block_perm(shock, h + 1, rng), zt))
                     for _ in range(P)])
    p = (1 + np.sum(sims >= b0)) / (P + 1)
    return p < 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--P", type=int, default=199)
    ap.add_argument("--N", type=int, default=300)
    ap.add_argument("--sig_lam", type=float, default=0.02)
    ap.add_argument("--sig_e", type=float, default=0.10)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--phi", type=float, default=0.0)
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.04])
    ap.add_argument("--horizontes", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--cenarios", nargs="+", default=["iid", "calibrada", "extrema"])
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    rows, k = [], 0
    for cen in a.cenarios:
        shock = caminho_choque(cen)
        for h in a.horizontes:
            for g in a.gammas:
                t0 = time.time()
                # semente fixa por combinacao (k) e por replica (i): reprodutivel
                seeds = [a.seed + 100000 * k + i for i in range(a.reps)]
                rej = Parallel(n_jobs=-1)(
                    delayed(uma_replica)(shock, h, a.N, a.sig_lam, a.sig_e,
                                         a.rho, a.phi, g, a.P, s) for s in seeds)
                rows.append(dict(cenario=cen, h=h, gamma=g, rejeicao=float(np.mean(rej)),
                                 n_efetivo=n_efetivo(shock - shock.mean()),
                                 reps=a.reps, P=a.P))
                print(f"{cen} h={h} gamma={g}: rej={np.mean(rej):.3f} ({time.time()-t0:.0f}s)",
                      flush=True)
                pd.DataFrame(rows).to_csv(ROOT / "monte_carlo_perm.csv", index=False)
                k += 1


if __name__ == "__main__":
    main()