# revisao/03_inter_def.py  -- rodar da raiz: python revisao/03_inter_def.py
import sys, io, contextlib, datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import placebos as pb
import ipo2
from placebos import ESPERADO

OUT = ROOT / "revisao" / "saidas"
LOGD = ROOT / "revisao" / "logs"; LOGD.mkdir(parents=True, exist_ok=True)
logf = open(LOGD / f"03_inter_def_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt", "w", encoding="utf-8")
def log(*a):
    s = " ".join(str(x) for x in a); print(s); logf.write(s + "\n"); logf.flush()

f = pd.read_csv(OUT / "02_entrantes.csv", dtype={"CNPJ_CIA": str})
defs = {"T0": set(f.loc[f["ent_T0"], "CNPJ_CIA"]),
        "FCA": set(f.loc[f["ent_fca"], "CNPJ_CIA"]),
        "FCA, fallback T0": set(f.loc[f["ent_fca"] | (~f["match_fca"] & f["ent_T0"]), "CNPJ_CIA"])}
if defs["T0"] != set(ipo2.WAVE):
    sys.exit("Gate falhou: T0 do 02_entrantes.csv difere de ipo2.WAVE.")
for k, v in defs.items():
    log(f"entrantes [{k}]: {len(v)} firmas")

frames = []
for nome, S in defs.items():
    ipo2.WAVE = S
    with contextlib.redirect_stdout(io.StringIO()):
        df = ipo2.teste_inter()
    df["def"] = nome
    frames.append(df)
res = pd.concat(frames, ignore_index=True)

# gate: T0 reproduz o ipo2_inter.csv ja visto
REF = {("Restrita", 5, "B: 2021, incumbentes"): 0.0519, ("Restrita", 5, "B: 2021, entrantes"): 0.0542,
       ("Total", 3, "B: 2021, incumbentes"): 0.0671}
for (am, h, q), v in REF.items():
    x = res[(res["def"] == "T0") & (res["amostra"] == am) & (res["h"] == h) & (res["quantidade"] == q)]["estimativa"].iloc[0]
    ok = abs(x - v) < 5e-4
    log(f"[gate] {am} h={h} {q}: {x:.4f} (ref {v}) -> {'OK' if ok else '<<< FALHA'}")
    if not ok:
        sys.exit("Gate falhou.")

p = res.pivot_table(index=["def", "amostra", "h"], columns="quantidade", values="estimativa")
out = pd.DataFrame({"g_inc": p["B: 2021, incumbentes"], "g_ent": p["B: 2021, entrantes"]})
out["obs_ent"] = res.groupby(["def", "amostra", "h"])["obs_entrantes"].first()
out["beta_pooled"] = [ESPERADO[(a, h)] for (_, a, h) in out.index]
out["inc/pooled"] = out["g_inc"] / out["beta_pooled"]
out["inc/ent"] = out["g_inc"] / out["g_ent"]
out["regra"] = np.where(out["inc/pooled"] >= .75, "nao se sustenta",
                np.where(out["g_inc"] < .5 * out["g_ent"], "se sustenta*", "intermediario"))
out = out.reset_index()
out.to_csv(OUT / "03_inter_def.csv", index=False)
res.to_csv(OUT / "03_inter_def_completo.csv", index=False)
log("\n(* 'se sustenta' exige ainda p_perm studentizado > 0,10; as estimativas sao pontuais, o DK de 2021 nao vale)")
log(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
log("Log em:", logf.name)