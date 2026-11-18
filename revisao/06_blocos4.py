# revisao/06_blocos4.py -- deleta cada janela de 4 trimestres consecutivos e reestima beta_C agrupado
# uso (da raiz): python revisao\06_blocos4.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "revisao" / "saidas"
LOGS = ROOT / "revisao" / "logs"
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import placebos as pb
import pandas as pd; df = pd.read_csv(OUT / "06_blocos4.csv")

print(df.groupby(["amostra","h"])["n_removido"].agg(lambda s: (s==0).sum()))
print(df.groupby(["amostra","h"])["n_removido"].min())

OUT.mkdir(parents=True, exist_ok=True)
G, W = pb.G, 4
J21 = tuple(range(pd.Period("2021Q1", freq="Q").ordinal, pd.Period("2021Q4", freq="Q").ordinal + 1))

def coef_sem(am, h, excl):
    d, x = pb.build(am, h, excl_q=tuple(excl))
    return (None, 0) if d is None else (pb.Fast(d, x).coefs(pb.S0, pb.SL10)["Ch_Caixa"], len(d))

# ---- gate: janela de 1 trimestre deve reproduzir placebo_loqo_detalhe.csv ----
f = ROOT / "placebo_loqo_detalhe.csv"
if f.exists():
    ref = pd.read_csv(f); ref = ref[ref["alvo"] == "Ch_Caixa"]
    bad = 0
    for am in pb.AMOSTRAS_TESTE:
        for h in pb.HS:
            for i in list(pb.V)[:3] + list(pb.V)[-3:]:
                r = ref[(ref.amostra == am) & (ref.h == h) & (ref.trimestre_excluido == pb.rot(G[i]))]
                c, _ = coef_sem(am, h, (G[i],))
                if len(r) and abs(c - float(r["coef"].iloc[0])) > 1e-8: bad += 1
    print(f"[gate] janela=1 vs placebo_loqo_detalhe.csv: {'OK' if bad == 0 else f'<<< {bad} DIFERENCAS'}")
    if bad: sys.exit("Gate falhou.")
else:
    print("[gate] placebo_loqo_detalhe.csv nao encontrado: gate pulado (rode `python placebos.py loqo` antes)")

# ---- janelas ----
print(f"trimestres na grade: {len(G)} -> {len(G) - W + 1} janelas de {W}")
rows = []
for am in pb.AMOSTRAS_TESTE:
    for h in pb.HS:
        d, x = pb.build(am, h)
        b0, n0 = pb.Fast(d, x).coefs(pb.S0, pb.SL10)["Ch_Caixa"], len(d)
        for i in range(len(G) - W + 1):
            win = tuple(int(q) for q in G[i:i + W])
            c, n = coef_sem(am, h, win)
            if c is None: continue
            rows.append(dict(amostra=am, h=h, ini=pb.rot(win[0]), fim=pb.rot(win[-1]), eh_2021=win == J21,
                             coef_pool=b0, coef_sem=c, delta=b0 - c, n_removido=n0 - n))
df = pd.DataFrame(rows)
df["rank_abs_delta"] = df.groupby(["amostra", "h"])["delta"].transform(lambda s: s.abs().rank(ascending=False).astype(int))
df.to_csv(OUT / "06_blocos4.csv", index=False)

for (am, h), g in df.groupby(["amostra", "h"]):
    print(f"\n== {am} h={h} | coef agrupado {g['coef_pool'].iloc[0]:.4f} | janelas: {len(g)} ==")
    print(g.sort_values("rank_abs_delta").head(5)[["ini", "fim", "coef_sem", "delta", "n_removido", "rank_abs_delta"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    r = g[g["eh_2021"]].iloc[0]
    print(f"   2021Q1-Q4: delta={r['delta']:.4f} (coef sem = {r['coef_sem']:.4f}), posicao {int(r['rank_abs_delta'])}/{len(g)}")

try:                                                        # grafico opcional
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    for a, ((am, h), g) in zip(ax.ravel(), df.groupby(["amostra", "h"])):
        a.bar(range(len(g)), g["delta"], color=np.where(g["eh_2021"], "C3", "C0"))
        a.set_title(f"{am} h={h}"); a.set_xticks(range(0, len(g), 8)); a.set_xticklabels(g["ini"].iloc[::8], rotation=45, fontsize=7)
    fig.supylabel("delta = beta_pool - beta_sem janela"); fig.tight_layout()
    fig.savefig(OUT / "06_blocos4.png", dpi=150); print("\ngrafico:", OUT / "06_blocos4.png")
except Exception as e:
    print("sem grafico:", e)