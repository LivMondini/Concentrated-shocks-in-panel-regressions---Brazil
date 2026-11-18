# revisao/31_numeros_choque.py   (da raiz: python revisao\31_numeros_choque.py)
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "revisao" / "saidas"
LOGS = ROOT / "revisao" / "logs"
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from placebos import S0, G, V
ordq = lambda a, b: set(pd.period_range(a, b, freq="Q").asi8)
Q1315, Q21, Q214 = ordq("2013Q2", "2015Q2"), ordq("2021Q1", "2021Q4"), ordq("2021Q4", "2021Q4")
first = V[0]                                           # 2011Q1 (so entra como lag)
JAN = {"54 base (2011Q2-2024Q3)": V[V != first], "55 choques (2011Q1-2024Q3)": V}
try:
    from monte_carlo_lp import n_efetivo as n_ef_mc
except Exception as e:
    n_ef_mc = None; print("n_efetivo do monte_carlo_lp indisponivel:", e)

rows = []
for nome, idx in JAN.items():
    s, q = S0[idx], G[idx]
    d2 = (s - s.mean()) ** 2; tot = d2.sum(); w = d2 / tot
    r = dict(janela=nome, n=len(s), media=s.mean(), dp_ddof1=s.std(ddof=1),
             share_1315=w[np.isin(q, list(Q1315))].sum(), share_2021=w[np.isin(q, list(Q21))].sum(),
             share_2021Q4=w[np.isin(q, list(Q214))].sum(), n_ef_hhi=1.0 / (w ** 2).sum())
    if n_ef_mc: r["n_ef_monte_carlo_lp"] = float(n_ef_mc(s - s.mean()))
    for ep, Q in (("1315", Q1315), ("2021", Q21)):
        m = np.isin(q, list(Q)); r[f"pos_{ep}"] = int((s[m] > 0).sum()); r[f"neg_{ep}"] = int((s[m] < 0).sum()); r[f"zero_{ep}"] = int((s[m] == 0).sum())
    m = ~np.isin(q, list(Q1315 | Q21))
    r["pos_demais"], r["neg_demais"], r["zero_demais"] = int((s[m] > 0).sum()), int((s[m] < 0).sum()), int((s[m] == 0).sum())
    r["pos_todos"], r["neg_todos"], r["zero_todos"] = int((s > 0).sum()), int((s < 0).sum()), int((s == 0).sum())
    rows.append(r)
res = pd.DataFrame(rows)
res.to_csv(OUT / "31_numeros_choque.csv", index=False)
print(res.T.to_string())
for nome, tag in (("share_2021", 0.3446), ("n_ef_hhi", 9.80)):
    for _, r in res.iterrows():
        print(f"[gate] {r['janela']}: {nome} = {r[nome]:.4f} | alvo {tag} -> {'OK' if abs(r[nome] - tag) < (5e-4 if nome == 'share_2021' else 5e-3) else '<<< DIFERENTE'}")

print("\nChoque (unidade 100 pb) por trimestre, 2013Q2-2015Q2 e 2021:")
for o in sorted(Q1315 | Q21):
    print(f"  {pd.Period(ordinal=int(o), freq='Q')}: {S0[int(o) - G[0]]:+.4f}")

t1 = pd.read_csv(OUT / "07_tabela1_stud.csv"); t1 = t1[t1["linha"] == "Pooled"]
sd54, sd55 = res.iloc[0]["dp_ddof1"], res.iloc[1]["dp_ddof1"]
t1 = t1.assign(beta_x_dp54=t1["coef"] * sd54, beta_x_dp55=t1["coef"] * sd55)
print("\nbeta_C (Pooled, Tabela 1 stud) x 1 DP do choque, em log points (x100 = %):")
print(t1[["amostra", "h", "coef", "beta_x_dp54", "beta_x_dp55"]].to_string(index=False, float_format=lambda v: f"{v:.5f}"))