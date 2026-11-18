# revisao/09i_grade_concentracao.py -- tamanho dos testes em funcao da concentracao do choque (Neff/T)
# Reusa 09d_mc_ext.rodar_celula (DGP do artigo, gamma = 0 => tamanho). Constroi uma familia de choques a partir do choque
# OBSERVADO por transformacao monotona s -> sign(s)|s|^a (a > 1 concentra, a < 1 espalha), com a escolhido por bisseccao
# para atingir Neff/T alvo (Neff = n_efetivo do monte_carlo_lp). Preserva a ordem temporal (e quase todos os sinais) do choque observado.
# uso (da raiz): python revisao\09i_grade_concentracao.py gate
#                python revisao\09i_grade_concentracao.py run --horizontes 5 --reps 200 --P 99          (teste de tempo)
#                python revisao\09i_grade_concentracao.py run --horizontes 3 5 7 --reps 2000 --P 399 --jobs 8
#                python revisao\09i_grade_concentracao.py run --horizontes 5 --tile 2 --reps 2000 --P 399 --jobs 8   (T = 108)
# Common random numbers: com gamma = 0, y nao depende do choque; todas as celulas de um mesmo h usam a mesma semente,
# entao as diferencas entre pontos da grade sao pareadas (menos ruido que o erro de MC de cada ponto isolado).
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
import sys, time, argparse, importlib
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "revisao" / "saidas"
LOGS = ROOT / "revisao" / "logs"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "revisao"))
import numpy as np, pandas as pd
from monte_carlo_lp import caminho_choque, n_efetivo
d9 = importlib.import_module("09d_mc_ext")
METODOS = ["dk", "dk2", "cluster", "fixedb", "perm_coef", "perm_stud"]

def razao(s): s = np.asarray(s, float); return n_efetivo(s - s.mean()) / len(s)

def transforma(base, a):
    c = np.asarray(base, float) - np.mean(base)
    s = np.sign(c) * np.abs(c) ** a
    s = s - s.mean()
    return s / s.std()

def constroi(base, alvo, tol=0.004):
    """Acha a tal que Neff/T da transformacao = alvo (bisseccao em log a). Falha alto se nao atingir."""
    lo, hi = -3.0, 3.0                                   # log a; razao decresce com a
    if not (razao(transforma(base, np.exp(hi))) < alvo < razao(transforma(base, np.exp(lo)))):
        raise ValueError(f"alvo {alvo:.3f} fora do alcance da familia (razao entre {razao(transforma(base, np.exp(hi))):.3f} e {razao(transforma(base, np.exp(lo))):.3f})")
    for _ in range(60):
        mid = (lo + hi) / 2
        if razao(transforma(base, np.exp(mid))) > alvo: lo = mid
        else: hi = mid
    a = np.exp((lo + hi) / 2); s = transforma(base, a)
    if abs(razao(s) - alvo) > tol: raise ValueError(f"bisseccao nao convergiu para {alvo}")
    return a, s

def limiar(df, col="dk_5", nivel=0.10):
    """Neff/T em que a taxa de rejeicao cruza `nivel` (interpolacao linear; so descritivo)."""
    g = df.sort_values("razao_alvo"); x, y = g["razao_alvo"].to_numpy(), g[col].to_numpy()
    if (y > nivel).all(): return "acima de %.0f%% em toda a grade" % (100 * nivel)
    if (y <= nivel).all(): return "abaixo de %.0f%% em toda a grade" % (100 * nivel)
    for i in range(len(x) - 1):
        if (y[i] - nivel) * (y[i + 1] - nivel) <= 0 and y[i] != y[i + 1]:
            return f"cruza {100 * nivel:.0f}% perto de Neff/T = {x[i] + (nivel - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]):.2f}"
    return "sem cruzamento unico (curva nao monotona)"

def gate(a):
    base = np.tile(d9.shock_observado(), a.tile); T = len(base)
    print(f"[info] T = {T} | Neff/T do choque observado = {razao(base):.3f} (Neff = {razao(base) * T:.2f}) | "
          f"iid do monte_carlo_lp: Neff/T = {razao(caminho_choque('iid')):.3f} | calibrado: {razao(caminho_choque('calibrada')):.3f}")
    ok = True
    for alvo in a.razoes:
        s = constroi(base, alvo)[1]; r = razao(s); ok &= abs(r - alvo) < 0.004
        print(f"[gate] alvo {alvo:.2f}: obtido {r:.3f} | Neff = {r * T:.2f} | media {s.mean():+.1e} dp {s.std():.3f} | fracao de sinais iguais ao observado: "
              f"{np.mean(np.sign(s) == np.sign(base - base.mean())):.2f}"); ok &= bool(abs(s.mean()) < 1e-10)
    print("[gate] " + ("OK" if ok else "FALHOU: cole estas linhas.")); sys.exit(0 if ok else 1)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("modo", choices=["gate", "run"])
    ap.add_argument("--razoes", type=float, nargs="+", default=[0.10, 0.14, 0.25, 0.33, 0.42, 0.50])
    ap.add_argument("--horizontes", type=int, nargs="+", default=[5]); ap.add_argument("--tile", type=int, default=1)
    ap.add_argument("--rolls", type=int, nargs="+", default=[0]); ap.add_argument("--N", type=int, default=300)
    ap.add_argument("--reps", type=int, default=2000); ap.add_argument("--P", type=int, default=399)
    ap.add_argument("--seed", type=int, default=2026); ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    if a.modo == "gate": return gate(a)
    obs = np.tile(d9.shock_observado(), a.tile); T = len(obs); rows = []
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"09i_grade_T{T}_reps{a.reps}_h{'-'.join(map(str, a.horizontes))}.csv"
    pontos = [("obs", 1.0, obs - obs.mean())] + [(f"{r:.2f}",) + constroi(obs, r) for r in a.razoes]
    for k, h in enumerate(a.horizontes):
        seed0 = a.seed + 100000 * (7000 + k)                                   # a mesma para todos os pontos do h (CRN)
        for rot, aexp, s in pontos:
            for d in a.rolls:
                t0 = time.time()
                row = d9.rodar_celula(np.roll(s, d), h, a.N, .9, 0.0, 0.0, a.P, a.reps, seed0, a.jobs)
                row.update(rotulo=rot, razao_alvo=(razao(obs) if rot == "obs" else float(rot)), razao_obtida=razao(s), expoente_a=aexp, roll=d)
                rows.append(row); pd.DataFrame(rows).to_csv(out, index=False)
                print(f"h={h} Neff/T={razao(s):.3f} (a={aexp:.2f}, roll={d}): " + " | ".join(f"{m} {row[f'{m}_5']:.3f}" for m in METODOS)
                      + f" (EPMC DK {row['dk_5_epmc']:.4f}) {time.time() - t0:.0f}s", flush=True)
        df = pd.DataFrame([r for r in rows if r["h"] == h and r["roll"] == a.rolls[0]])
        print(f"[h={h}] DK (h+1) {limiar(df, 'dk_5')} | DK (2h+2) {limiar(df, 'dk2_5')} | cluster {limiar(df, 'cluster_5')} | perm. stud. {limiar(df, 'perm_stud_5', 0.07)} (nivel 7%)")
    print(f"\nSalvo em {out}")

if __name__ == "__main__":
    main()