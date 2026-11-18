# revisao/05_tabelas_artigo.py  -- rodar: python revisao\05_tabelas_artigo.py
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "revisao" / "saidas"
LOGS = ROOT / "revisao" / "logs"
S = OUT

def fmt(v):
    return f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
def md(df):
    c = list(df.columns)
    L = ["| " + " | ".join(c) + " |", "|" + "---|" * len(c)]
    L += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(L)

out = []
# ---- T1: pesos FWL por trimestre (substitui o proxy de 34,5% da secao 3.2)
pw = pd.read_csv(S / "01_pesos_resumo.csv"); pw = pw[pw.grupo == "todos"]
wb = pd.read_csv(S / "01b_within_between.csv"); wb = wb[wb.grupo == "todos"]
t1 = pw.merge(wb[["amostra", "h", "between_total"]], on=["amostra", "h"])
q4 = []
for a, h in zip(t1.amostra, t1.h):
    q = pd.read_csv(S / f"01_pesos_{a}_h{h}_trimestre.csv")
    q4.append(float(q.loc[q.trim == "2021Q4", "omega"].iloc[0]))
t1["omega_2021Q4"] = q4
t1["between/beta_C"] = t1.between_total / t1.beta_C
t1 = t1[["amostra", "h", "beta_C", "omega_2021", "omega_2021Q4", "share_c_2021", "between/beta_C"]]
out += ["### T1. Pesos FWL exatos de beta_C (omega_t = peso do trimestre; share_c = contribuicao de 2021 a beta_C)",
        md(t1), "*between/beta_C = parte de beta_C vinda de comparar trimestres (sem efeitos de tempo).*\n"]

# ---- T2: 2021, incumbentes x entrantes (modelo B), tres definicoes de entrante
t2 = pd.read_csv(S / "03_inter_def.csv")
t2 = t2[["def", "amostra", "h", "g_inc", "g_ent", "obs_ent", "inc/pooled", "inc/ent", "regra"]]
out += ["### T2. Gradiente de 2021 entre incumbentes (g_inc) e entrantes (g_ent), estimativas pontuais",
        md(t2), "*Entrante primario = listagem FCA em 2019Q1-2021Q4. DK de 2021 nao e inferencia valida.*\n"]

# ---- T3: permutacao studentizada + IC de g_inc por inversao
rows = []
for f in sorted(S.glob("04_perm_stud_*.csv")):
    r = pd.read_csv(f).set_index("estatistica")
    am, h, d_ = r["amostra"].iloc[0], int(r["h"].iloc[0]), r["def"].iloc[0]
    tag = f"{am}_h{h}_" + d_.replace(", ", "_").replace(" ", "_")
    ci = pd.read_csv(S / f"04_ci_ginc_{tag}.csv")
    o95 = ci.loc[ci.p_stud > 0.05, "g0"]
    rows.append(dict(definicao=d_, amostra=am, h=h, beta_C=r.loc["beta_C", "coef"],
                     p_perm_beta=r.loc["beta_C", "p_coef"], p_stud_beta=r.loc["beta_C", "p_stud"],
                     t95_nula=r.loc["beta_C", "t_null_q95"], g_inc=r.loc["g_inc", "coef"],
                     p_stud_ginc=r.loc["g_inc", "p_stud"],
                     ic95_lo=o95.min() if len(o95) else np.nan, ic95_hi=o95.max() if len(o95) else np.nan,
                     p_stud_gent=r.loc["g_ent", "p_stud"], p_stud_dif=r.loc["dif_ent_inc", "p_stud"]))
t3 = pd.DataFrame(rows).sort_values(["definicao", "amostra", "h"])
out += ["### T3. Permutacao studentizada (blocos h+1 nao circulares, B=999, seed 12345)",
        md(t3), "*t95_nula = percentil 95 de |t| na nula (DK nominal: 1,96). IC por inversao da permutacao studentizada.*\n"]

(S / "05_tabelas_artigo.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out)); print("\n[ok] ->", S / "05_tabelas_artigo.md")