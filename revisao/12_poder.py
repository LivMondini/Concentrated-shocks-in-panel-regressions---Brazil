# revisao/12_poder.py -- poder ajustado por tamanho, limite "oracle" e efeito minimo detectavel (MDE)
# Arquivo NOVO. Nao altera nada existente. Especificacao fixada ANTES de rodar.
#
# uso (da raiz):  python revisao\12_poder.py --reps 5000 --P 399
#   opcoes: --hs 3 5 7 --phis 0 0.5 --gammas 0 0.02 0.04 0.06 0.08 0.10 --seed 2026 --jobs -1
#
# DGP: o mesmo do 09d (observado, N=300, rho=.9, sig_lam=.02, sig_e=.10), com gamma variando.
# As sementes sao PAREADAS entre gammas (simula soma gamma*S*z sem sortear nada novo): a curva de poder
# usa os mesmos choques aleatorios em todos os gammas.
#
# Testes (todos bilaterais, 5%):
#   dk        : |t| DK(h+1) contra 1,96 (bruto)  e contra o valor critico exato sob H0 (ajustado)
#   perm_coef : permutacao em blocos do coeficiente   (bruto: p <= 5%; ajustado: p com desempate aleatorio)
#   perm_stud : permutacao studentizada em blocos      (idem)
#   oracle    : |beta_hat| contra o quantil 95% EXATO de |beta_hat| sob H0 (distribuicao verdadeira, MC)
# Ajustado por tamanho = valor critico tirado da propria distribuicao sob H0 (gamma = 0) da MESMA celula.
#
# LEITURA (fixada antes de rodar):
#   (i)  se o poder ajustado de perm_stud fica a <= 5 p.p. do oracle em gamma = 0,04 na maioria das celulas,
#        a perda de poder e INTRINSECA a T=54 (informacao do desenho), nao defeito do teste;
#   (ii) se o proprio oracle tem poder < 80% em gamma = 0,04, o MDE(80%) excede 0,04 por limitacao de informacao.
import sys, time, argparse, importlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "revisao" / "saidas"
LOGS = ROOT / "revisao" / "logs"
Z975 = 1.959963984540


def get_d9():
    import os
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "revisao"))
    return importlib.import_module("09d_mc_ext"), ROOT


def replica_pod(shock, h, N, rho, phi, gamma, P, seed):
    """Mesmo consumo de rng que 09d_mc_ext.replica. Devolve [|t DK|, |beta|, p_coef, p_stud]."""
    d9 = importlib.import_module("09d_mc_ext")
    import monte_carlo_perm as mp
    rng = np.random.default_rng(seed)
    y, zt = d9.simula(shock, h, N, .02, .10, rho, phi, gamma, rng)
    L = h + 1
    perms = np.array([mp.block_perm(shock, L, rng) for _ in range(P)])
    b, hh = d9.fit_h(d9.Pre(y, zt), np.vstack([shock[None, :], perms]))
    T = y.shape[1]
    t = b / d9.se_from_h(hh, L, N, T)
    pc = (1 + np.sum(np.abs(b[1:]) >= abs(b[0]))) / (P + 1)
    ps = (1 + np.sum(np.abs(t[1:]) >= abs(t[0]))) / (P + 1)
    return np.array([abs(t[0]), abs(b[0]), pc, ps])


def estatisticas(r, P, rng_j):
    """Estatisticas de evidencia (maior = mais evidencia contra H0). Desempate aleatorio nos p-valores discretos."""
    n = len(r)
    jc, js = rng_j.random(n) / (P + 1), rng_j.random(n) / (P + 1)
    return {"dk": r[:, 0], "perm_coef": -(r[:, 2] + jc), "perm_stud": -(r[:, 3] + js), "oracle": r[:, 1]}


def analisa(r_gamma, P, seed=7):
    """r_gamma: {gamma: array (reps,4)}; exige gamma = 0. Devolve DataFrame com poder bruto e ajustado."""
    s0 = estatisticas(r_gamma[0.0], P, np.random.default_rng(seed))
    crit = {k: np.quantile(v, 0.95) for k, v in s0.items()}
    rows = []
    for g, r in sorted(r_gamma.items()):
        s = estatisticas(r, P, np.random.default_rng(seed + 1))
        row = dict(gamma=g,
                   dk_bruto=np.mean(r[:, 0] > Z975), dk_adj=np.mean(s["dk"] > crit["dk"]),
                   permC_bruto=np.mean(r[:, 2] <= .05 + 1e-9), permC_adj=np.mean(s["perm_coef"] > crit["perm_coef"]),
                   permS_bruto=np.mean(r[:, 3] <= .05 + 1e-9), permS_adj=np.mean(s["perm_stud"] > crit["perm_stud"]),
                   oracle=np.mean(s["oracle"] > crit["oracle"]))
        rows.append(row)
    return pd.DataFrame(rows)


def mde(df, col, alvo=0.8):
    g, p = df["gamma"].to_numpy(), df[col].to_numpy()
    if p.max() < alvo: return np.nan                      # acima da grade
    ok = np.argsort(p)
    return float(np.interp(alvo, p[ok], g[ok]))


def main():
    from joblib import Parallel, delayed
    ap = argparse.ArgumentParser()
    ap.add_argument("--hs", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--phis", type=float, nargs="+", default=[0.0, 0.5])
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
    ap.add_argument("--reps", type=int, default=5000); ap.add_argument("--P", type=int, default=399)
    ap.add_argument("--seed", type=int, default=2026); ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    assert 0.0 in a.gammas, "a grade precisa incluir gamma = 0 (valor critico sob H0)"
    d9, root = get_d9(); shock = np.asarray(d9.shock_observado(), float)
    out_rows, resumo = [], []
    for h in a.hs:
        for phi in a.phis:
            t0 = time.time(); rg = {}
            for g in a.gammas:
                seed0 = a.seed + 100000 * (13000 + 10 * h + int(phi * 10))      # mesmo seed0 para todos os gammas
                rg[g] = np.array(Parallel(n_jobs=a.jobs, batch_size=64)(
                    delayed(replica_pod)(shock, h, 300, .9, phi, g, a.P, seed0 + i) for i in range(a.reps)))
            df = analisa(rg, a.P); df.insert(0, "phi", phi); df.insert(0, "h", h); out_rows.append(df)
            print(f"\n=== h={h} phi={phi:g} ({time.time() - t0:.0f}s) | poder (%), bruto e ajustado por tamanho ===")
            print((df.set_index(["h", "phi", "gamma"]) * 100).round(1).to_string())
            r04 = df.loc[np.isclose(df.gamma, 0.04)]
            resumo.append(dict(h=h, phi=phi,
                               permS_adj_004=float(r04.permS_adj.iloc[0]) if len(r04) else np.nan,
                               oracle_004=float(r04.oracle.iloc[0]) if len(r04) else np.nan,
                               MDE80_permS_adj=mde(df, "permS_adj"), MDE80_oracle=mde(df, "oracle"),
                               MDE80_permS_bruto=mde(df, "permS_bruto")))
    res = pd.DataFrame(resumo); res["gap_pp_004"] = 100 * (res.oracle_004 - res.permS_adj_004)
    print("\n=== RESUMO (MDE = menor gamma com 80% de poder; NaN = acima da grade) ===")
    print(res.round(3).to_string(index=False))
    pd.concat(out_rows).to_csv(OUT / f"12_poder_reps{a.reps}.csv", index=False); res.to_csv(OUT / f"12_poder_resumo_reps{a.reps}.csv", index=False)
    print(f"\nSalvo em {OUT}")


if __name__ == "__main__":
    main()