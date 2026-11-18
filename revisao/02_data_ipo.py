# revisao/02_data_ipo.py  -- rodar da raiz: python revisao/02_data_ipo.py
import sys, datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import placebos as pb
from placebos import build
from ipo import T0, CENS, ONDAS, ord_q, firmas_onda

OUT = ROOT / "revisao" / "saidas"; OUT.mkdir(parents=True, exist_ok=True)
LOGD = ROOT / "revisao" / "logs"; LOGD.mkdir(parents=True, exist_ok=True)
logf = open(LOGD / f"02_data_ipo_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt", "w", encoding="utf-8")
def log(*a):
    s = " ".join(str(x) for x in a); print(s); logf.write(s + "\n"); logf.flush()

def c14(s):
    return s.astype(str).str.replace(r"\D", "", regex=True).str.zfill(14)
def le(f):
    return pd.read_csv(f, sep=";", encoding="latin-1", dtype=str)
def qord(dt):                       # ordinal trimestral, igual ao t do painel
    return (dt.dt.year - 1970) * 4 + dt.dt.quarter - 1

arqs = sorted((ROOT / "dados_fca").glob("fca_cia_aberta_valor_mobiliario_*.csv"))
fca = pd.concat([le(f)[["CNPJ_Companhia", "Data_Inicio_Negociacao", "Data_Inicio_Listagem"]] for f in arqs])
fca["cnpj14"] = c14(fca["CNPJ_Companhia"])
for c in ("Data_Inicio_Negociacao", "Data_Inicio_Listagem"):
    fca[c] = pd.to_datetime(fca[c], errors="coerce")
g = fca.groupby("cnpj14").agg(L_list=("Data_Inicio_Listagem", "min"), L_neg=("Data_Inicio_Negociacao", "min"))
cadf = ROOT / "cad_cia_aberta.csv"
cad = le(cadf)[["CNPJ_CIA", "DT_REG"]]
cad["cnpj14"] = c14(cad["CNPJ_CIA"]); cad["DT_REG"] = pd.to_datetime(cad["DT_REG"], errors="coerce")
cad = cad.groupby("cnpj14")["DT_REG"].min()
log(f"FCA: {len(arqs)} arquivos, {fca['cnpj14'].nunique()} CNPJs | cad: {len(cad)} CNPJs")

f = pd.DataFrame({"CNPJ_CIA": T0.index, "T0": T0.values})
f["cnpj14"] = c14(f["CNPJ_CIA"])
f = f.merge(g, left_on="cnpj14", right_index=True, how="left").merge(cad, left_on="cnpj14", right_index=True, how="left")
f["L_min"] = f[["L_list", "L_neg"]].min(axis=1)
f["Lq"] = qord(f["L_min"]); f["REGq"] = qord(f["DT_REG"])
a, b = ord_q(*[ONDAS["entrada 2019Q1-2021Q4"][0]]), ord_q(ONDAS["entrada 2019Q1-2021Q4"][1])
f["match_fca"] = f["L_min"].notna()
f["ent_T0"] = f["CNPJ_CIA"].isin(firmas_onda(*ONDAS["entrada 2019Q1-2021Q4"]))
f["ent_fca"] = f["match_fca"] & f["Lq"].between(a, b)
f["ent_reg"] = f["REGq"].between(a, b)
f.to_csv(OUT / "02_entrantes.csv", index=False)

log(f"\nFirmas no painel: {len(f)} | match FCA: {f['match_fca'].sum()} ({f['match_fca'].mean():.1%}) | com DT_REG: {f['DT_REG'].notna().mean():.1%}")
for am in ("Restrita", "Total"):
    d, _ = build(am, 5)
    s = f[f["CNPJ_CIA"].isin(d["CNPJ_CIA"].unique())]
    log(f"  {am} h=5: {len(s)} firmas, match FCA {s['match_fca'].mean():.1%}")
m = f[f["match_fca"]]
log("\nEntrante T0 x FCA (so firmas com match):")
log(pd.crosstab(m["ent_T0"], m["ent_fca"], margins=True).to_string())
log(f"Entrantes FCA sem match de T0 na onda: {(m['ent_fca'] & ~m['ent_T0']).sum()} | T0 sem FCA: {(~f['match_fca'] & f['ent_T0']).sum()}")
w = m[m["ent_T0"] | m["ent_fca"]]
log("\nT0 - trimestre de listagem (trimestres), entrantes por qualquer definicao:")
log((w["T0"] - w["Lq"]).describe().round(1).to_string())
log("\nFirmas por ano da menor data de listagem (2015-2022):")
log(m.assign(ano=m["L_min"].dt.year).query("2015 <= ano <= 2022")["ano"].value_counts().sort_index().to_string())
dif = (m["L_neg"] - m["L_list"]).dt.days.dropna()
log(f"\nL_neg - L_list (dias): mediana {dif.median():.0f}, share |dif|>365: {(dif.abs() > 365).mean():.1%}")
log("Log em:", logf.name)