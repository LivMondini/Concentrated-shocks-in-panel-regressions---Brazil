# revisao/01b_pesos_within.py
import sys
from pathlib import Path
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "placebos.py").exists())
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import placebos as pb
from placebos import build, HS
from ipo import firmas_onda, ONDAS

OUT = ROOT / "revisao" / "saidas"
f = pd.read_csv(OUT / "02_entrantes.csv", dtype={"CNPJ_CIA": str})
ENT = set(f.loc[f["ent_fca"], "CNPJ_CIA"])
ALVO = "Ch_Caixa"
rows = []
for am in ("Restrita", "Total"):
    for h in HS:
        d, x = build(am, h)
        codes = pd.factorize(d["CNPJ_CIA"])[0]; cnt = np.bincount(codes).astype(float)
        dm = lambda a: a - (np.bincount(codes, weights=a) / cnt)[codes]
        X = np.column_stack([dm(d[v].to_numpy(float)) for v in x]); y = dm(d["y"].to_numpy(float))
        j = x.index(ALVO); Xo = np.delete(X, j, axis=1)
        r = X[:, j] - Xo @ np.linalg.lstsq(Xo, X[:, j], rcond=None)[0]
        den = float(r @ r); beta = float(r @ y) / den
        t = d["t"].to_numpy(); ano = pd.PeriodIndex.from_ordinals(t, freq="Q").year.to_numpy()
        gr = np.where(d["CNPJ_CIA"].isin(ENT), "entrante", "incumbente")
        df = pd.DataFrame({"t": t, "ano": ano, "gr": gr, "r": r, "y": y})
        df["m"] = df.groupby("t")["r"].transform("mean")          # media trimestral do residuo
        df["w"] = (df["r"] - df["m"]) * df["y"] / den              # parte dentro do trimestre
        df["b"] = df["m"] * df["y"] / den                          # parte entre trimestres
        assert abs(df["w"].sum() + df["b"].sum() - beta) < 1e-8, "gate falhou"
        for g, gg in [("todos", df)] + list(df.groupby("gr")):
            s = gg[gg["ano"] == 2021]
            rows.append(dict(amostra=am, h=h, grupo=g, beta_C=beta, within_2021=s["w"].sum(),
                             between_2021=s["b"].sum(), within_total=gg["w"].sum(), between_total=gg["b"].sum()))
res = pd.DataFrame(rows); res.to_csv(OUT / "01b_within_between.csv", index=False)
print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))