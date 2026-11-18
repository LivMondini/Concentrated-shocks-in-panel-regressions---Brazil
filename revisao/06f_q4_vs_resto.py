# revisao/06f_q4_vs_resto.py -- exclui (a) 2021Q4, (b) 2021Q1-Q3 (mantem Q4), (c) bloco 2021, e compara com beta agrupado
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "revisao" / "saidas"
LOGS = ROOT / "revisao" / "logs"
sys.path.insert(0, str(ROOT))
import pandas as pd
import placebos as pb

Q = lambda s: pd.Period(s, freq="Q").ordinal
cenarios = {"so_2021Q4": (Q("2021Q4"),),
            "2021Q1-Q3 (mantem Q4)": (Q("2021Q1"), Q("2021Q2"), Q("2021Q3")),
            "bloco_2021": tuple(Q(f"2021Q{k}") for k in (1, 2, 3, 4))}
rows = []
for am in pb.AMOSTRAS_TESTE:
    for h in pb.HS:
        d, x = pb.build(am, h)
        b0 = pb.Fast(d, x).coefs(pb.S0, pb.SL10)["Ch_Caixa"]
        linha = dict(amostra=am, h=h, beta_pool=b0)
        for nome, ex in cenarios.items():
            dq, xq = pb.build(am, h, excl_q=ex)
            c = pb.Fast(dq, xq).coefs(pb.S0, pb.SL10)["Ch_Caixa"]
            linha[nome] = c
            linha["queda_" + nome] = (b0 - c) / b0
        rows.append(linha)
df = pd.DataFrame(rows)
df.to_csv(OUT / "06f_q4_vs_resto.csv", index=False)
print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))