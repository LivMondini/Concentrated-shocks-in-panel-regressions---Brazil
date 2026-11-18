# -*- coding: utf-8 -*-
"""
revisao/00_gate_reproducao.py

GATE DE REPRODUCAO. Rode ANTES de qualquer resultado novo.
Nao altera nenhum arquivo do pipeline original; so le e grava logs em revisao/logs/.

  A) Reestima, com as funcoes que o repositorio ja usa (placebos.build / placebos.fit_full),
     as celulas "pooled" e "sem 2021" da Tabela 1 / A1 / A15 e compara coeficiente,
     erro-padrao Driscoll-Kraay e N.
  B) Confere a serie do choque (placebos.S0) com a Tabela A2 (concentracao).
  C) Le permutacao_resultados.csv (saida JA gravada, nao reexecuta a permutacao) e compara
     com os p-valores de permutacao da Tabela 1 / A15. Informa o B usado.

Criterio de aprovacao (celulas de A):  N exato; |coef - ref| <= 0.00006; |EP - ref| <= 0.0006.

Uso, na raiz do repositorio (PowerShell):   python revisao\\00_gate_reproducao.py
"""
import os
import sys
import time
import platform
import traceback
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "revisao" / "saidas"
LOGS = ROOT / "revisao" / "logs"
sys.path.insert(0, str(ROOT))
LOGS.mkdir(parents=True, exist_ok=True)
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


class Tee:
    """Escreve na tela e no arquivo de log ao mesmo tempo."""
    def __init__(self, caminho):
        self.arq = open(caminho, "w", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.arq.write(s)

    def flush(self):
        self.out.flush()
        self.arq.flush()


tee = Tee(LOGS / f"00_gate_reproducao_{STAMP}.txt")
sys.stdout = tee


def versao(pacote):
    try:
        return metadata.version(pacote)
    except Exception:
        return "?"


print("=" * 78)
print("GATE DE REPRODUCAO", STAMP)
print("raiz:", ROOT)
print("python", platform.python_version(),
      "| numpy", versao("numpy"), "| pandas", versao("pandas"),
      "| statsmodels", versao("statsmodels"), "| linearmodels", versao("linearmodels"))
print("=" * 78)

# ---------------------------------------------------------------------------
# Alvos (Tabela 1, A1 e A15 do manuscrito/apendice v4)
# (amostra, h, excl_years, coef_4dec, ep_3dec, N)
# ---------------------------------------------------------------------------
ALVOS = [
    ("Restrita", 3, (), 0.0476, 0.013, 8920),
    ("Restrita", 5, (), 0.0549, 0.014, 8817),
    ("Restrita", 7, (), 0.0601, 0.015, 8692),
    ("Total", 3, (), 0.0430, 0.014, 14029),
    ("Total", 5, (), 0.0404, 0.018, 13720),
    ("Total", 7, (), 0.0323, 0.018, 13406),
    ("Restrita", 3, (2021,), 0.0223, 0.018, 8140),
    ("Restrita", 5, (2021,), 0.0250, 0.022, 8039),
    ("Restrita", 7, (2021,), 0.0344, 0.022, 7921),
    ("Total", 3, (2021,), 0.0025, 0.022, 12821),
    ("Total", 5, (2021,), -0.0064, 0.033, 12526),
    ("Total", 7, (2021,), -0.0156, 0.034, 12233),
]
TOL_COEF, TOL_SE = 0.00006, 0.0006

resumo = []   # linhas para o CSV final

# ---------------------------------------------------------------------------
# A) Coeficientes e EP
# ---------------------------------------------------------------------------
print("\n[A] Reestimacao com placebos.build / placebos.fit_full")
t0 = time.time()
try:
    import placebos as pb
except Exception:
    print("FALHA ao importar placebos.py (o import pode rodar codigo de preparo do painel):")
    traceback.print_exc()
    print("\nGATE NAO APROVADO: cole este log inteiro para eu diagnosticar.")
    sys.exit(1)
print(f"import de placebos ok em {time.time() - t0:.1f}s")
print("chaves de AMOSTRAS:", list(pb.AMOSTRAS))

for am, h, excl, ref_coef, ref_se, ref_n in ALVOS:
    rot = f"{am:8s} h={h} {'sem 2021' if excl else 'pooled  '}"
    try:
        d, x = pb.build(am, h, excl_years=excl)
        if d is None:
            print(f"{rot}: build devolveu None (poucas observacoes)")
            resumo.append(dict(bloco="A", item=rot, status="ERRO"))
            continue
        r = pb.fit_full(d, x, h)
        coef = float(r.params["Ch_Caixa"])
        se = float(r.std_errors["Ch_Caixa"])
        n = int(r.nobs)
        ok = (n == ref_n) and abs(coef - ref_coef) <= TOL_COEF and abs(se - ref_se) <= TOL_SE
        print(f"{rot}: coef={coef:.5f} (ref {ref_coef:.4f}) | EP={se:.5f} (ref {ref_se:.3f}) | "
              f"N={n} (ref {ref_n}) | firmas={d['CNPJ_CIA'].nunique()} | T={d['t'].nunique()} | "
              f"regressores={len(x)} -> {'OK' if ok else 'DIFERE'}")
        resumo.append(dict(bloco="A", item=rot, coef=coef, ref_coef=ref_coef, se=se, ref_se=ref_se,
                           n=n, ref_n=ref_n, status="OK" if ok else "DIFERE"))
    except Exception:
        print(f"{rot}: EXCECAO")
        traceback.print_exc()
        resumo.append(dict(bloco="A", item=rot, status="ERRO"))

# ---------------------------------------------------------------------------
# B) Serie do choque x Tabela A2
# ---------------------------------------------------------------------------
print("\n[B] Serie do choque (placebos.S0) x Tabela A2")
try:
    S0 = np.asarray(pb.S0, dtype=float)
    SL = np.asarray(pb.SL10, dtype=float)
    ordinais = int(pb.G0) + np.arange(len(S0))
    usa = np.isfinite(S0) & np.isfinite(SL)          # trimestres com choque e choque defasado
    s, o = S0[usa], ordinais[usa]
    anos = np.array([pd.Period(ordinal=int(k), freq="Q").year for k in o])
    qtrs = np.array([pd.Period(ordinal=int(k), freq="Q").quarter for k in o])
    dev2 = (s - s.mean()) ** 2
    share = dev2 / dev2.sum()
    checks = {
        "trimestres usados": (len(s), 54, 0),
        "share 2021": (share[anos == 2021].sum(), 0.3446, 0.0005),
        "share 2021Q4": (share[(anos == 2021) & (qtrs == 4)].sum(), 0.2768, 0.0005),
        "n efetivo (centrado)": (1.0 / np.sum(share ** 2), 9.797, 0.01),
        "media": (s.mean(), 0.0, 0.0005),
        "DP (ddof=0)": (s.std(ddof=0), 0.264, 0.0005),
        "minimo": (s.min(), -0.515, 0.0005),
        "maximo": (s.max(), 1.019, 0.0005),
        "n positivos": (int((s > 0).sum()), 29, 0),
        "n negativos": (int((s < 0).sum()), 25, 0),
    }
    for nome, (val, ref, tol) in checks.items():
        ok = abs(val - ref) <= tol
        print(f"  {nome:22s} {val:10.4f}  (ref {ref})  -> {'OK' if ok else 'DIFERE'}")
        resumo.append(dict(bloco="B", item=nome, coef=float(val), ref_coef=ref,
                           status="OK" if ok else "DIFERE"))
    print(f"  (S0 tem {len(S0)} posicoes, {int(np.isfinite(S0).sum())} nao-NaN; "
          f"primeiro ordinal G0={int(pb.G0)} = {pd.Period(ordinal=int(pb.G0), freq='Q')})")
except Exception:
    print("EXCECAO em B")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# C) permutacao_resultados.csv (saida ja gravada) x Tabela 1 / A15
# ---------------------------------------------------------------------------
print("\n[C] permutacao_resultados.csv (leitura; a permutacao NAO foi reexecutada)")
REF_PERM = {
    ("Restrita", "Pooled"): [0.019, 0.017, 0.001],
    ("Restrita", "Excluindo 2021"): [0.256, 0.266, 0.074],
    ("Restrita", "2013-15"): [0.001, 0.024, 0.001],
    ("Restrita", "2021"): [0.470, 0.568, 0.349],
    ("Restrita", "Calmo"): [0.110, 0.121, 0.010],
    ("Total", "Pooled"): [0.115, 0.154, 0.252],
    ("Total", "Excluindo 2021"): [0.925, 0.814, 0.550],
    ("Total", "2013-15"): [0.001, 0.060, 0.010],
    ("Total", "2021"): [0.542, 0.674, 0.463],
    ("Total", "Calmo"): [0.118, 0.200, 0.295],
}
try:
    arq = ROOT / "permutacao_resultados.csv"
    mod = datetime.fromtimestamp(os.path.getmtime(arq)).strftime("%Y-%m-%d %H:%M")
    pr = pd.read_csv(arq)
    pr["amostra"] = pr["amostra"].astype(str).str.strip()
    pr["estatistica"] = pr["estatistica"].astype(str).str.strip()
    print(f"  arquivo modificado em {mod}; B usado: {sorted(pr['B'].unique())}; "
          f"amostras: {sorted(pr['amostra'].unique())}")
    if set(pr["B"].unique()) != {999}:
        print("  ATENCAO: B != 999 -> este CSV nao e o da Tabela 1/A15 (o default do script e 99).")
    for (am, est), refs in REF_PERM.items():
        for h, ref in zip((3, 5, 7), refs):
            lin = pr[(pr["amostra"] == am) & (pr["estatistica"] == est) & (pr["h"] == h)]
            if lin.empty:
                print(f"  {am:8s} {est:15s} h={h}: NAO ENCONTRADO no CSV")
                resumo.append(dict(bloco="C", item=f"{am} {est} h={h}", status="AUSENTE"))
                continue
            p = float(lin["p_perm"].iloc[0])
            ok = abs(p - ref) <= 0.0005
            print(f"  {am:8s} {est:15s} h={h}: p_perm={p:.3f} (ref {ref:.3f}) "
                  f"coef={float(lin['coef'].iloc[0]):.4f} sd={float(lin['sd_perm'].iloc[0]):.4f} "
                  f"-> {'OK' if ok else 'DIFERE'}")
            resumo.append(dict(bloco="C", item=f"{am} {est} h={h}", coef=p, ref_coef=ref,
                               status="OK" if ok else "DIFERE"))
except Exception:
    print("EXCECAO em C")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# Resumo
# ---------------------------------------------------------------------------
df = pd.DataFrame(resumo)
df.to_csv(LOGS / f"00_gate_resumo_{STAMP}.csv", index=False)
print("\n" + "=" * 78)
if not df.empty:
    print(df.groupby(["bloco", "status"]).size().to_string())
    a = df[df["bloco"] == "A"]
    aprovado = (not a.empty) and bool((a["status"] == "OK").all())
    print("\nGATE A (coeficientes, EP, N):", "APROVADO" if aprovado else "NAO APROVADO")
print("Log em:", tee.arq.name)
print("Cole este log inteiro na conversa.")