"""permutacao_lp.py -- teste de permutacao em blocos do choque trimestral.
Uso:  python permutacao_lp.py 99      (teste)
      python permutacao_lp.py 999     (final)
Saida: permutacao_resultados.csv
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from fechamento_letter import prepare_panel, winsor, HS, AMOSTRAS
from fechamento_b1 import E21, E1315, REF_POOL, REF_21, REF_1315

ROOT = Path(__file__).resolve().parent

B = int(sys.argv[1]) if len(sys.argv) > 1 else 99
SEED = 12345
INT = {"Ch_Caixa": "Cash_Ratio_z", "Ch_Alav": "Alavancagem_L1_z", "Ch_Tam": "Tamanho_z",
       "Ch_FCO": "Fluxo_Caixa_Op_Ativo_L1_z", "Ch_Q": "Tobin_Q_L1_z"}


def within_ols(d, cols):
    """OLS com efeito fixo de firma (demeaning). Retorna Series de coeficientes."""
    y = winsor(d["y"])
    X = d[cols].copy()
    X["_y"] = y.values
    g = d["CNPJ_CIA"].values
    Xd = X - X.groupby(g).transform("mean")
    b = np.linalg.lstsq(Xd[cols].to_numpy(float), Xd["_y"].to_numpy(float), rcond=None)[0]
    return pd.Series(b, index=cols)


def apply_shock(d, smap, am):
    d = d.copy()
    d["Shock"] = d["t"].map(smap)
    d["Shock_L1"] = (d["t"] - 1).map(smap)
    for ch, z in INT.items():
        if ch == "Ch_Q" and am != "Restrita":
            continue
        d[ch] = d["Shock"] * d[z]
    return d


def stats(d, cols, am):
    t = d["t"]
    in21, in1315 = t.isin(E21).to_numpy(), t.isin(E1315).to_numpy()
    out = {"Pooled": within_ols(d, cols)["Ch_Caixa"],
           "Excluindo 2021": within_ols(d[~in21], cols)["Ch_Caixa"]}
    j = d.copy()
    w1, w2 = in1315.astype(float), in21.astype(float)
    j["P1"], j["P2"] = w1, w2
    extra = ["P1", "P2"]
    vars_ = ["Shock", "Ch_Caixa", "Ch_Alav", "Ch_Tam", "Ch_FCO"] + (["Ch_Q"] if am == "Restrita" else [])
    for v in vars_:
        j[v + "_P1"], j[v + "_P2"] = j[v] * w1, j[v] * w2
        extra += [v + "_P1", v + "_P2"]
    b = within_ols(j, cols + extra)
    out["2013-15"] = b["Ch_Caixa"] + b["Ch_Caixa_P1"]
    out["2021"] = b["Ch_Caixa"] + b["Ch_Caixa_P2"]
    out["Calmo"] = b["Ch_Caixa"]
    return out


def block_perm(vals, L, rng):
    n = len(vals)
    blocks = [vals[i:i + L] for i in range(0, n, L)]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[i] for i in order])


def main():
    p, shock, serie_y, samples, specs = prepare_panel()
    q = shock.dropna()
    tq, sv = q.index.to_numpy(), q.to_numpy(float)
    rng = np.random.default_rng(SEED)
    refs = {"Pooled": REF_POOL, "2021": REF_21, "2013-15": REF_1315}
    rows = []
    for am in AMOSTRAS:
        for h in HS:
            d, cols = samples[(am, h)], specs[am]
            obs = stats(d, cols, am)
            key = (am, h)
            print(f"[{am} h={h}] validacao:", {k: round(obs[k], 4) for k in ("Pooled", "2021", "2013-15")},
                  "ref:", {k: round(refs[k][key], 4) for k in refs}, flush=True)
            L = h + 1
            sims = {k: [] for k in obs}
            t0 = time.time()
            for b in range(B):
                smap = dict(zip(tq, block_perm(sv, L, rng)))
                s = stats(apply_shock(d, smap, am), cols, am)
                for k in sims:
                    sims[k].append(s[k])
                if b == 0:
                    print(f"   1a permutacao: {time.time() - t0:.1f}s (estimado total {B * (time.time() - t0) / 60:.1f} min)", flush=True)
            for k, v in obs.items():
                a = np.abs(np.array(sims[k]))
                pv = (1 + np.sum(a >= abs(v))) / (B + 1)
                rows.append(dict(amostra=am, h=h, estatistica=k, coef=v, p_perm=pv, B=B,
                                 p_min=1 / (B + 1), sd_perm=float(np.std(sims[k]))))
            pd.DataFrame(rows).to_csv(ROOT / "permutacao_resultados.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()