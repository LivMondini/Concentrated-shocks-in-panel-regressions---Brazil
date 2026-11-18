# revisao/11_causa_distorcao.py -- isola a causa do excesso de tamanho da permutacao em blocos
# Arquivo NOVO. Nao altera nada existente. Especificacao fixada ANTES de rodar.
#
# Modelo reduzido sob H0 (phi = 0, gamma = 0): a estatistica depende do quarter-level
#   c_t = Cov_i(z_it, y_it) ~ sum_{k=0}^{h} rho^k * lam_{t+k},  lam iid  ->  MA(h) com pesos rho^k
# e beta_hat ~ sum_t S~_t c_t / sum_t S~_t^2 (S~ = choque demeado).
#
# uso (da raiz):
#   python revisao\11_causa_distorcao.py pred      # (b) previsao analitica vs tamanhos ja medidos
#   python revisao\11_causa_distorcao.py interv    # (c) intervencao: troca a dependencia entre blocos
#   opcoes interv: --hs 1 2 5 --reps 10000 --P 399 --rho 0.9 --seed 2026 --jobs -1
#
# LEITURA (fixada antes de rodar):
#   pred  : correlacao (Pearson e Spearman) entre tamanho PREVISTO e tamanho MEDIDO (perm. do coeficiente).
#           Causa "isolada" se rho_Spearman >= 0.8 nos pontos (h, caminho) e nas 18 celulas de deslocamento.
#   interv: estrutura 'ma' deve reproduzir o excesso; 'iid' e 'bloco' devem dar ~5% (dentro de 2 EPMC).
#           Se 'bloco' continuar > 5% + 2 EPMC, a dependencia ENTRE blocos NAO e a causa unica.
import sys, time, argparse, math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "revisao" / "saidas"
LOGS = ROOT / "revisao" / "logs"
Z975 = 1.959963984540

# ---------- tamanhos JA MEDIDOS (10.000 replicas; saidas de 10_diag_tamanho.py h e de 09d shift) ----------
MED = {  # (coef, stud, razao q95), h = 0..7
    "obs": ([.048, .061, .072, .064, .054, .076, .082, .070], [.049, .060, .071, .059, .048, .077, .099, .061],
            [.995, .946, .915, .944, .989, .884, .814, .936]),
    "iid": ([.051, .048, .030, .049, .067, .047, .064, .053], [.049, .051, .035, .051, .066, .055, .071, .068],
            [.983, .975, 1.056, .972, .902, .960, .893, .905]),
    "calibrada": ([.050, .056, .058, .066, .056, .081, .095, .058], [.047, .055, .058, .062, .059, .079, .092, .065],
                  [.994, .959, .949, .923, .934, .864, .816, .910]),
}
SHIFT = {  # (h, d): (coef, stud)  -- calibrada deslocada, modo shift do 09d
    (3, 0): (.0627, .0578), (3, 1): (.0645, .0609), (3, 2): (.0721, .0748), (3, 3): (.0563, .0521),
    (5, 0): (.0813, .0794), (5, 1): (.0771, .0699), (5, 2): (.0610, .0599), (5, 3): (.0476, .0522),
    (5, 4): (.0890, .0805), (5, 5): (.0787, .0639),
    (7, 0): (.0585, .0591), (7, 1): (.0580, .0574), (7, 2): (.0626, .0714), (7, 3): (.0473, .0514),
    (7, 4): (.0804, .0642), (7, 5): (.0906, .0811), (7, 6): (.0932, .0976), (7, 7): (.0899, .0833),
}


def get_shocks():
    import os
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
    import importlib
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "revisao"))
    from monte_carlo_lp import caminho_choque
    d9 = importlib.import_module("09d_mc_ext")
    return {"obs": np.asarray(d9.shock_observado(), float), "iid": np.asarray(caminho_choque("iid"), float),
            "calibrada": np.asarray(caminho_choque("calibrada"), float)}, ROOT


# ---------- utilitarios ----------
def gamma_ma(T, h, rho):
    """Autocovariancia (T x T, Toeplitz) de c_t = sum_{k=0}^h rho^k lam_{t+k}, lam ~ N(0,1) iid."""
    w = rho ** np.arange(h + 1)
    g = np.zeros(T)
    for l in range(min(h, T - 1) + 1):
        g[l] = np.sum(w[:h + 1 - l] * w[l:])
    idx = np.abs(np.subtract.outer(np.arange(T), np.arange(T)))
    return g[idx]


def block_perm_idx(T, L, n, rng):
    """n permutacoes em blocos de L (ultimo bloco pode ser curto), sem circularidade. Devolve (n, T) de indices."""
    nb = -(-T // L)
    blocks = [np.arange(i * L, min((i + 1) * L, T)) for i in range(nb)]
    return np.array([np.concatenate([blocks[j] for j in rng.permutation(nb)]) for _ in range(n)])


def bart(g, L):
    V = (g * g).sum(1)
    for l in range(1, L + 1):
        V = V + 2 * (1 - l / (L + 1)) * (g[:, l:] * g[:, :-l]).sum(1)
    return np.maximum(V, 1e-30)


def rank(x): return np.argsort(np.argsort(x)).astype(float)


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.corrcoef(a, b)[0, 1], np.corrcoef(rank(a), rank(b))[0, 1]


# ---------- (b) previsao analitica ----------
def razao_var(S, h, rho, L, rng, n=4000):
    """R = E_pi[S~pi' G S~pi] / (S~' G S~): variancia da permutacao relativa a verdadeira (condicional ao caminho S)."""
    Sd = S - S.mean(); T = len(S); G = gamma_ma(T, h, rho)
    Sp = Sd[block_perm_idx(T, L, n, rng)]
    num = np.einsum("ij,jk,ik->i", Sp, G, Sp).mean()
    return num / (Sd @ G @ Sd)


def prev_tamanho(R):                                   # normal: rejeita se |stat| > 1.96*sd_perm
    return math.erfc(Z975 * math.sqrt(R) / math.sqrt(2))


def modo_pred(a):
    sh, _ = get_shocks(); rng = np.random.default_rng(a.seed)
    pts = []                                            # (rotulo, previsto, med_coef, med_stud, sqrtR, razao_q95)
    for nome in ("obs", "iid", "calibrada"):
        for h in range(8):
            R = razao_var(sh[nome], h, a.rho, h + 1, rng)
            c, s, q = MED[nome][0][h], MED[nome][1][h], MED[nome][2][h]
            pts.append((f"{nome} h={h}", prev_tamanho(R), c, s, math.sqrt(R), q))
    base = sh["calibrada"]
    for (h, d), (c, s) in SHIFT.items():
        R = razao_var(np.roll(base, d), h, a.rho, h + 1, rng)
        pts.append((f"shift h={h} d={d}", prev_tamanho(R), c, s, math.sqrt(R), np.nan))
    df = pd.DataFrame(pts, columns=["celula", "previsto", "med_coef", "med_stud", "sqrtR", "razao_q95_med"])
    print(df.round(4).to_string(index=False))
    for nome, sel in (("choque x h (24 pontos)", ~df.celula.str.startswith("shift")), ("deslocamentos (18)", df.celula.str.startswith("shift")),
                      ("todos (42)", np.ones(len(df), bool))):
        d = df[sel]
        pc, sc = corr(d.previsto, d.med_coef); ps_, ss = corr(d.previsto, d.med_stud)
        print(f"{nome}: previsto vs coef  Pearson {pc:.2f} Spearman {sc:.2f} | previsto vs stud  Pearson {ps_:.2f} Spearman {ss:.2f}")
    d = df[~df.celula.str.startswith("shift")]
    pc, sc = corr(d.sqrtR, d.razao_q95_med)
    print(f"sqrt(R) previsto vs razao q95 medida (24 pontos): Pearson {pc:.2f} Spearman {sc:.2f}")
    out = ROOT / a.out if a.out else OUT
    df.to_csv(out / "11_pred.csv", index=False) if a.out else None


# ---------- (c) intervencao ----------
def gera_c(struct, lam, w, T, h, L):
    if struct == "iid":
        return lam[:T].copy()
    c = np.zeros(T); t = np.arange(T)
    for k in range(h + 1):
        term = w[k] * lam[k:k + T]
        if struct == "bloco":                           # corta a soma na fronteira do bloco: Gamma bloco-diagonal
            term = term * ((t + k) // L == t // L)
        c += term
    return c


def rep(seed, Sd, h, rho, struct, L, Ldk, P):
    rng = np.random.default_rng(seed); T = len(Sd); w = rho ** np.arange(h + 1)
    lam = rng.standard_normal(T + h)
    c = gera_c(struct, lam, w, T, h, L); cc = c - c.mean()
    idx = block_perm_idx(T, L, P, rng)
    Sm = np.vstack([Sd[None, :], Sd[idx]])
    ss = (Sm * Sm).sum(1); num = Sm @ cc; b = num / ss
    e = cc[None, :] - b[:, None] * Sm
    t = num / np.sqrt(bart(Sm * e, Ldk))
    pc = (1 + np.sum(np.abs(b[1:]) >= abs(b[0]))) / (P + 1)
    ps = (1 + np.sum(np.abs(t[1:]) >= abs(t[0]))) / (P + 1)
    return np.array([pc <= .05 + 1e-9, ps <= .05 + 1e-9, abs(t[0]) > Z975], float)


def modo_interv(a):
    from joblib import Parallel, delayed
    sh, root = get_shocks(); rows = []
    for nome in ("obs", "calibrada"):
        Sd = sh[nome] - sh[nome].mean(); T = len(Sd)
        for h in a.hs:
            L = h + 1
            if T % L: print(f"[aviso] T={T} nao divisivel por L={L}: h={h} ignorado (blocos nao identicos)"); continue
            for k, struct in enumerate(("ma", "bloco", "iid")):
                t0 = time.time()
                r = np.array(Parallel(n_jobs=a.jobs, batch_size=64)(
                    delayed(rep)(a.seed + 100000 * (11000 + 10 * h + k) + i, Sd, h, a.rho, struct, L, L, a.P) for i in range(a.reps)))
                m, se = r.mean(0), r.std(0, ddof=1) / np.sqrt(len(r))
                rows.append(dict(choque=nome, h=h, L=L, blocos=T // L, estrutura=struct, reps=a.reps, P=a.P,
                                 coef=m[0], stud=m[1], dk_assint=m[2], epmc=se[0]))
                print(f"{nome} h={h} L={L} (blocos={T // L}) Gamma={struct:5s}: coef {m[0]:.3f} | stud {m[1]:.3f} | "
                      f"DK {m[2]:.3f} (EPMC {se[0]:.4f}) {time.time() - t0:.0f}s", flush=True)
    out = OUT / f"11_interv_reps{a.reps}.csv"
    pd.DataFrame(rows).to_csv(out, index=False); print(f"\nSalvo em {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["pred", "interv"])
    ap.add_argument("--hs", type=int, nargs="+", default=[1, 2, 5])
    ap.add_argument("--reps", type=int, default=10000); ap.add_argument("--P", type=int, default=399)
    ap.add_argument("--rho", type=float, default=0.9); ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--jobs", type=int, default=-1); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    modo_pred(a) if a.modo == "pred" else modo_interv(a)


if __name__ == "__main__":
    main()