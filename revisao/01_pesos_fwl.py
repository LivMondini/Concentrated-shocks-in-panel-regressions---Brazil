# revisao/01_pesos_fwl.py  -- rodar da raiz: python revisao/01_pesos_fwl.py
import sys, time, datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "revisao" / "saidas"
LOGS = ROOT / "revisao" / "logs"
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd

import placebos as pb                      # faz chdir para a raiz
from placebos import build, fit_full, S0, G0, HS, ano_de_t
from ipo import firmas_onda, ONDAS         # WAVE = entrantes 2019Q1-2021Q4 (proxy T0)

TOL = 1e-8
ANO = 2021
OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logf = open(LOGS / f"01_pesos_fwl_{stamp}.txt", "w", encoding="utf-8")
def log(*a):
    s = " ".join(str(x) for x in a); print(s); logf.write(s + "\n"); logf.flush()

WAVE = firmas_onda(*ONDAS["entrada 2019Q1-2021Q4"])
ALVO = "Ch_Caixa"

def within(a, codes, cnt):
    return a - (np.bincount(codes, weights=a) / cnt)[codes]

def pesos(am, h):
    d, x = build(am, h)
    codes = pd.factorize(d["CNPJ_CIA"])[0]
    cnt = np.bincount(codes).astype(float)
    X = np.column_stack([within(d[v].to_numpy(float), codes, cnt) for v in x])
    y = within(d["y"].to_numpy(float), codes, cnt)
    j = x.index(ALVO)
    Xo = np.delete(X, j, axis=1)
    g = np.linalg.lstsq(Xo, X[:, j], rcond=None)[0]
    r = X[:, j] - Xo @ g                      # residuo FWL de Ch_Caixa
    den = float(r @ r)
    beta = float(r @ y) / den
    # gate: bate com PanelOLS (fit_full) e com a soma dos c_t
    b_pkg = float(fit_full(d, x, h).params[ALVO])
    df = pd.DataFrame({"t": d["t"].to_numpy(),
                       "grupo": np.where(d["CNPJ_CIA"].isin(WAVE), "entrante", "incumbente"),
                       "r2": r * r / den, "ry": r * y / den})
    q = df.groupby(["t", "grupo"]).agg(omega=("r2", "sum"), c=("ry", "sum"),
                                       n=("r2", "size")).reset_index()
    return d, q, beta, b_pkg

rows = []
for am in ("Restrita", "Total"):
    for h in HS:
        t0 = time.time()
        d, q, beta, b_pkg = pesos(am, h)
        soma_c, soma_w = q["c"].sum(), q["omega"].sum()
        ok = abs(soma_c - beta) < TOL and abs(soma_c - b_pkg) < TOL and abs(soma_w - 1) < TOL
        log(f"[gate] {am:8s} h={h}: sum c={soma_c:.10f} beta_FWL={beta:.10f} "
            f"beta_PanelOLS={b_pkg:.10f} sum omega={soma_w:.10f} -> {'OK' if ok else '<<< FALHA'}")
        if not ok:
            sys.exit("Gate FWL falhou: nao use os pesos.")
        qt = q.groupby("t")[["omega", "c", "n"]].sum()
        qt["b_t"] = qt["c"] / qt["omega"]
        qt["S_t"] = S0[qt.index.to_numpy() - G0]
        qt["ano"] = np.asarray(ano_de_t(qt.index.to_numpy()))
        qt["trim"] = [str(pd.Period(ordinal=int(o), freq="Q")) for o in qt.index]
        qt.reset_index().to_csv(OUT / f"01_pesos_{am}_h{h}_trimestre.csv", index=False)
        q["ano"] = np.asarray(ano_de_t(q["t"].to_numpy()))
        q.to_csv(OUT / f"01_pesos_{am}_h{h}_trimestre_grupo.csv", index=False)
        for gr, gg in [("todos", q)] + list(q.groupby("grupo")):
            s21 = gg[gg["ano"] == ANO]
            rows.append(dict(amostra=am, h=h, grupo=gr, beta_C=beta,
                             omega_2021=s21["omega"].sum(), c_2021=s21["c"].sum(),
                             share_c_2021=s21["c"].sum() / beta,
                             omega_total_grupo=gg["omega"].sum(),
                             c_total_grupo=gg["c"].sum()))
        top = qt.sort_values("omega", ascending=False).head(5)
        log(f"   top-5 omega: " + ", ".join(f"{r.trim}={r.omega:.3f}" for r in top.itertuples()))
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(qt["S_t"], qt["b_t"], s=2000 * qt["omega"] + 8,
                       c=(qt["ano"] == ANO), cmap="coolwarm", alpha=.7)
            ax.axhline(beta, ls="--", c="k", lw=.8)
            ax.set(xlabel="choque S_t", ylabel="inclinacao b_t",
                   title=f"{am} h={h} (area = omega_t; vermelho = {ANO}; linha = beta_C)")
            fig.tight_layout(); fig.savefig(OUT / f"01_bt_{am}_h{h}.png", dpi=150); plt.close(fig)
        except Exception as e:
            log("[aviso] grafico nao gerado:", e)
        log(f"   {time.time()-t0:.0f}s")

res = pd.DataFrame(rows)
res.to_csv(OUT / "01_pesos_resumo.csv", index=False)
log(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
log("Log em:", logf.name)