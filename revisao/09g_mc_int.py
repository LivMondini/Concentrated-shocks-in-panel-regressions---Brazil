# revisao/09g_mc_int.py -- tamanho/poder com interacoes S_t x X_k correlacionadas com Cash
# Compara: DK(h+1), DK(2h+2), cluster, fixed-b, permutacao forte (spec completa),
#          Freedman-Lane em blocos, wild cluster bootstrap por trimestre e em blocos.
# uso (da raiz): python revisao\09g_mc_int.py gate
#                python revisao\09g_mc_int.py run --calib obs --K 4 --reps 2000 --P 399 --horizontes 3 5 7
#                python revisao\09g_mc_int.py run --calib stress --K 4 --reps 2000 --P 399
#                python revisao\09g_mc_int.py run --calib obs --K 0 --reps 500 --P 199   (K=0: confere com 09d)
# Nao altera 09d. Usa de 09d_mc_ext: gera_z, simula, shock_observado, NIV, ZN.
# RNG: simula e permutacao forte consomem o rng principal exatamente como o 09d (K=0 reproduz o 09d);
# X_k usa default_rng([seed,7]); FL e WCB usam default_rng([seed,8]).
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
d9 = importlib.import_module("09d_mc_ext")
NIV, ZN = d9.NIV, d9.ZN
METODOS = ["dk", "dk2", "cluster", "fixedb", "perm_forte", "FL", "wcb_trim", "wcb_bloco"]

# ---- calibracoes (ordem: Alavancagem, Tamanho, FCO, Tobin Q); fonte: mapa do repositorio ----
RHO_OBS = np.array([-0.151, -0.055, 0.047, 0.105])
BETA_OBS = {3: [-0.0157, -0.0208, 0.0022, -0.0053],      # amostra Restrita, ctrl_int
            5: [-0.0189, -0.0151, -0.0086, -0.0164],
            7: [0.0023, -0.0070, -0.0312, -0.0177]}

def calibracao(nome, h, K):
    if nome == "obs": return RHO_OBS[:K], np.array(BETA_OBS[h][:K])
    return np.full(K, 0.5), np.full(K, 0.04)             # estresse

# ---- DGP ----
def gera_x(zt, rho, rho_x, rng_x):
    """x_k = rho_x z + sqrt(1-rho_x^2) W_k (W_k AR(1) como z), padronizado dentro do trimestre."""
    K = len(rho_x); N, T = zt.shape; XT = np.empty((K, N, T))
    for k in range(K):
        W = d9.gera_z(N, T, rho, rng_x)
        x = rho_x[k] * zt + np.sqrt(1 - rho_x[k] ** 2) * W
        XT[k] = (x - x.mean(0)) / x.std(0)
    return XT

def simula_int(shock, h, N, rho, phi, gamma, rng, beta1, K, rho_x, beta_x, rng_x):
    y, zt = d9.simula(shock, h, N, .02, .10, rho, phi, gamma, rng)
    y = y + beta1 * shock[None, :]
    XT = gera_x(zt, rho, rho_x, rng_x) if K else np.empty((0,) + zt.shape)
    for k in range(K): y = y + beta_x[k] * shock[None, :] * XT[k]
    return y, zt, XT

# ---- estimacao geral (FE de firma); colunas: S, z, x_1..x_K, S*z, S*x_1..S*x_K ----
def build(S, zt, XT):
    K, N, T = XT.shape; X = np.empty((N, T, 2 * K + 3))
    X[:, :, 0] = S[None, :]; X[:, :, 1] = zt
    X[:, :, 2:2 + K] = XT.transpose(1, 2, 0)
    X[:, :, K + 2] = S[None, :] * zt
    X[:, :, K + 3:] = (S[None, None, :] * XT).transpose(1, 2, 0)
    return X - X.mean(1, keepdims=True)

def bart(s, L):
    V = s @ s
    for l in range(1, L + 1): V += 2 * (1 - l / (L + 1)) * (s[l:] @ s[:-l])
    return V

def fit_geral(Xd, yd, idxC, Ls):
    """beta_C e EP (Bartlett L, correcao NT/(NT-p-N)) para cada L em Ls."""
    N, T, p = Xd.shape; X2 = Xd.reshape(N * T, p)
    inv = np.linalg.inv(X2.T @ X2); b = inv @ (X2.T @ yd.ravel())
    u = yd - (X2 @ b).reshape(N, T)
    s = np.einsum("itk,it->tk", Xd, u) @ inv[:, idxC]
    corr = N * T / (N * T - p - N)
    return b[idxC], [np.sqrt(max(bart(s, L) * corr, 1e-30)) for L in Ls]

_cache = {}
def crit_fixedb(T, L, N, p, nsim=200000):
    """Critico |t| de scores iid centrados (como 09e), mesma correcao com p regressores."""
    key = (T, L, N, p)
    if key in _cache: return _cache[key]
    rng = np.random.default_rng(2026); corr = N * T / (N * T - p - N); ts = []
    for _ in range(nsim // 50000):
        x = rng.standard_normal((50000, T)); xd = x - x.mean(1, keepdims=True); V = (xd * xd).sum(1)
        for l in range(1, L + 1): V = V + 2 * (1 - l / (L + 1)) * (xd[:, l:] * xd[:, :-l]).sum(1)
        ts.append(np.abs(x.sum(1)) / np.sqrt(np.maximum(V * corr, 1e-30)))
    t = np.concatenate(ts)
    _cache[key] = {nv: float(np.quantile(t, 1 - nv / 100)) for nv in NIV}
    return _cache[key]

# ---- uma replica ----
def build_fast(S, V, Vd):
    """Mesmo X de build(), reaproveitando V (N,T,q) e Vd = V demeaned. Colunas: S, V, S*V."""
    N, T, q = V.shape; X = np.empty((N, T, 2 * q + 1))
    X[:, :, 0] = S - S.mean()
    X[:, :, 1:1 + q] = Vd
    W = S[None, :, None] * V
    X[:, :, 1 + q:] = W - W.mean(1, keepdims=True)
    return X

def replica_int(shock, h, N, rho, phi, gamma, P, seed, beta1, K, rho_x, beta_x, P2=None):
    P2 = P if P2 is None else P2
    rng = np.random.default_rng(seed)
    y, zt, XT = simula_int(shock, h, N, rho, phi, gamma, rng, beta1, K, rho_x, beta_x,
                           np.random.default_rng([seed, 7]))
    T = y.shape[1]; L = h + 1; p = 2 * K + 3; iC = K + 2
    perms = [mp.block_perm(shock, L, rng) for _ in range(P)]     # mesmo consumo do 09d
    rng2 = np.random.default_rng([seed, 8])
    yd = y - y.mean(1, keepdims=True)
    V = np.concatenate([zt[:, :, None], XT.transpose(1, 2, 0)], axis=2)   # (N,T,K+1)
    Vd = V - V.mean(1, keepdims=True)
    Xd = build_fast(shock, V, Vd)
    corr = N * T / (N * T - p - N)
    b0, (se1, se2, se0) = fit_geral(Xd, yd, iC, [L, 2 * h + 2, 0])
    t0 = b0 / se1
    # permutacao forte
    ts = np.array([(lambda r: r[0] / r[1][0])(fit_geral(build_fast(sp, V, Vd), yd, iC, [L])) for sp in perms])
    p_perm = (1 + np.sum(np.abs(ts) >= abs(t0))) / (P + 1)
    # modelo restrito (sem S*z)
    cols = [c for c in range(p) if c != iC]
    Xr = Xd[:, :, cols].reshape(N * T, -1)
    br = np.linalg.solve(Xr.T @ Xr, Xr.T @ yd.ravel())
    fit_r = (Xr @ br).reshape(N, T); e_r = yd - fit_r
    # ---- reducao a dimensao T (exata): X, inv fixos para FL e WCB ----
    X2 = Xd.reshape(N * T, p); inv = np.linalg.inv(X2.T @ X2)
    g = inv[:, iC]; Xg = Xd @ g                                   # (N,T)
    Fs = np.einsum("itk,it->k", Xd, fit_r)                        # X'fit_r
    Fg = np.einsum("it,it->t", Xg, fit_r)
    C = np.tensordot(Xd, e_r, axes=(0, 0)).transpose(0, 2, 1)     # C[t,u,:] = sum_i Xd[i,t,:] e_r[i,u]
    Cg = C @ g                                                    # (T,T)
    Mg = np.einsum("itk,it->tk", Xd, Xg)                          # (T,p)
    ar = np.arange(T); Cd = C[ar, ar]; Cdg = Cg[ar, ar]

    def stat_fl(idx, Lx):
        b = inv @ (Fs + C[ar, idx].sum(0))
        s = Fg + Cg[ar, idx] - Mg @ b
        return b[iC] / np.sqrt(max(bart(s, Lx) * corr, 1e-30))

    def stat_w(w, Lx):
        b = inv @ (Fs + w @ Cd)
        s = Fg + w * Cdg - (Cg @ w) / T - Mg @ b
        return b[iC] / np.sqrt(max(bart(s, Lx) * corr, 1e-30))

    ordem = np.arange(T, dtype=float)
    tfl = np.empty(P2)
    for k in range(P2):
        idx = mp.block_perm(ordem, L, rng2).astype(int)
        tfl[k] = stat_fl(idx, L)
    p_fl = (1 + np.sum(np.abs(tfl) >= abs(t0))) / (P2 + 1)
    t0c = b0 / se0; tw = np.empty(P2)
    for k in range(P2):
        w = rng2.choice([-1.0, 1.0], T)
        tw[k] = stat_w(w, 0)
    p_w = (1 + np.sum(np.abs(tw) >= abs(t0c))) / (P2 + 1)
    nb = -(-T // L); twb = np.empty(P2)
    for k in range(P2):
        w = np.repeat(rng2.choice([-1.0, 1.0], nb), L)[:T]
        twb[k] = stat_w(w, L)
    p_wb = (1 + np.sum(np.abs(twb) >= abs(t0))) / (P2 + 1)
    return np.array([abs(t0), abs(b0 / se2), abs(t0c), p_perm, p_fl, p_w, p_wb])

def agrega(r, crit):
    row = {}
    for nv in NIV:
        rej = {"dk": r[:, 0] > ZN[nv], "dk2": r[:, 1] > ZN[nv], "cluster": r[:, 2] > ZN[nv],
               "fixedb": r[:, 0] > crit[nv], "perm_forte": r[:, 3] < nv / 100, "FL": r[:, 4] < nv / 100,
               "wcb_trim": r[:, 5] < nv / 100, "wcb_bloco": r[:, 6] < nv / 100}
        for k, v in rej.items():
            v = v.astype(float); row[f"{k}_{nv:g}"] = v.mean(); row[f"{k}_{nv:g}_epmc"] = v.std(ddof=1) / np.sqrt(len(v))
    return row

# ---- gates ----
def gate(a):
    sh = caminho_choque("calibrada"); ok = True
    # gate 4: K=0 reproduz 09d (t de DK, DK2, cluster; p da perm. studentizada)
    mt, mp_ = 0.0, 0.0
    for h in (3, 5):
        for g in (0.0, 0.04):
            for i in range(3):
                o = d9.replica(sh, h, 300, .02, .10, .9, 0.0, g, 199, a.seed + i)
                n = replica_int(sh, h, 300, .9, 0.0, g, 199, a.seed + i, 0.0, 0, np.empty(0), np.empty(0), P2=9)
                mt = max(mt, abs(o[0] - n[0]), abs(o[1] - n[1]), abs(o[2] - n[2])); mp_ = max(mp_, abs(o[4] - n[3]))
    print(f"[gate 4] K=0 vs 09d: max |dif| t = {mt:.2e} | max |dif| p perm stud = {mp_:.2e}"); ok &= mt < 1e-8 and mp_ < 1e-12
    # gate 5: fit_geral (K=2) vs MQO bruto com dummies de firma
    rng = np.random.default_rng(5); N, T, K = 15, 20, 2
    zt = rng.standard_normal((N, T)); XT = rng.standard_normal((K, N, T)); S = rng.standard_normal(T)
    y = rng.standard_normal((N, T)) + 0.3 * S * zt + 0.2 * S * XT[0]
    Xd = build(S, zt, XT); yd = y - y.mean(1, keepdims=True)
    b, _ = fit_geral(Xd, yd, K + 2, [0])
    Xf = np.concatenate([Xd.reshape(N * T, -1)[:, :0], np.column_stack([
        np.broadcast_to(S, (N, T)).ravel(), zt.ravel(), XT[0].ravel(), XT[1].ravel(), (S * zt).ravel(),
        (S * XT[0]).ravel(), (S * XT[1]).ravel()]), np.kron(np.eye(N), np.ones((T, 1)))], axis=1)
    bf = np.linalg.lstsq(Xf, y.ravel(), rcond=None)[0][4]
    print(f"[gate 5] fit_geral K=2 vs MQO com dummies: |dif| beta_C = {abs(b - bf):.2e}"); ok &= abs(b - bf) < 1e-9
    # gate 6: x_k com media 0, DP 1 por trimestre e correlacao com z ~ rho_x
    rng = np.random.default_rng(6); z = d9.gera_z(3000, 54, .9, rng); rx = np.array([-0.151, 0.5])
    X = gera_x(z, .9, rx, np.random.default_rng(7))
    cr = [np.mean([np.corrcoef(z[:, t], X[k][:, t])[0, 1] for t in range(54)]) for k in range(2)]
    print(f"[gate 6] max|media|={np.abs(X.mean(1)).max():.1e} | DP-1 max={np.abs(X.std(1) - 1).max():.1e} | corr(z,x)={np.round(cr, 3)} vs {rx}")
    ok &= np.abs(X.mean(1)).max() < 1e-10 and abs(cr[0] - rx[0]) < .03 and abs(cr[1] - rx[1]) < .03
    # gate 7 (fumaca, nao falha): tamanho 5% sob calibracao obs, K=4, R=200, P=99
    K = 4; rx, bx = calibracao("obs", 5, K); t0 = time.time()
    r = np.array(Parallel(n_jobs=a.jobs)(delayed(replica_int)(sh, 5, 300, .9, 0.0, 0.0, 99, a.seed + i, 0.1, K, rx, bx) for i in range(200)))
    row = agrega(r, crit_fixedb(len(sh), 6, 300, 2 * K + 3))
    print("[gate 7] fumaca R=200 (erro MC ~1,5pp): " + " | ".join(f"{m} {row[f'{m}_5']:.3f}" for m in METODOS) + f" ({time.time() - t0:.0f}s)")
    print("[gate] " + ("OK" if ok else "FALHOU: cole estas linhas.")); sys.exit(0 if ok else 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["gate", "run"])
    ap.add_argument("--calib", choices=["obs", "stress"], default="obs")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--reps", type=int, default=2000); ap.add_argument("--P", type=int, default=399)
    ap.add_argument("--horizontes", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--beta1", type=float, nargs="+", default=[0.0, 0.10])
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.0])
    ap.add_argument("--seed", type=int, default=2026); ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    if a.modo == "gate": return gate(a)
    sh = d9.shock_observado(); T = len(sh); N = 300; rows = []; k = 0
    out = OUT / f"09g_mc_int_{a.calib}_K{a.K}_reps{a.reps}.csv"
    for h in a.horizontes:
        rx, bx = calibracao(a.calib, h, a.K)
        for b1 in a.beta1:
            for g in a.gammas:
                t0 = time.time(); seed0 = a.seed + 100000 * (3000 + k); k += 1
                r = np.array(Parallel(n_jobs=a.jobs, batch_size=16)(delayed(replica_int)(
                    sh, h, N, .9, 0.0, g, a.P, seed0 + i, b1, a.K, rx, bx) for i in range(a.reps)))
                crit = crit_fixedb(T, h + 1, N, 2 * a.K + 3)
                row = dict(calib=a.calib, K=a.K, h=h, beta1=b1, gamma=g, N=N, T=T, reps=a.reps, P=a.P,
                           rho_x=";".join(map(str, rx)), beta_x=";".join(map(str, bx)))
                row.update(agrega(r, crit)); rows.append(row); pd.DataFrame(rows).to_csv(out, index=False)
                print(f"{a.calib} K={a.K} h={h} beta1={b1} gamma={g}: " + " | ".join(f"{m} {row[f'{m}_5']:.3f}" for m in METODOS)
                      + f" (EPMC perm {row['perm_forte_5_epmc']:.4f}) {time.time() - t0:.0f}s", flush=True)
    print(f"\nSalvo em {out}")

if __name__ == "__main__":
    main()