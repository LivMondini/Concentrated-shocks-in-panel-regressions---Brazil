# revisao/10_diag_tamanho.py -- diagnostico do excesso de tamanho (6-8%) da permutacao studentizada
# Arquivo NOVO: nao altera 09d/09g. Usa de 09d_mc_ext: simula, Pre, fit_h, se_from_h, shock_observado.
#
# uso (da raiz do repositorio):
#   python revisao\10_diag_tamanho.py gate
#   python revisao\10_diag_tamanho.py sanidade            # h=0 (c_t iid): deve dar ~5%
#   python revisao\10_diag_tamanho.py h                   # tamanho vs h = 0..7
#   python revisao\10_diag_tamanho.py rho --h 5           # rho = 0, .5, .9, .95
#   python revisao\10_diag_tamanho.py bloco --h 5         # comprimento do bloco de permutacao
#   python revisao\10_diag_tamanho.py dk --h 5            # largura de banda do EP DK
#   python revisao\10_diag_tamanho.py h --choque iid      # idem com choque iid (ou calibrada)
# opcoes: --reps 2000 --P 399 --phi 0.0 --seed 2026 --jobs -1
#
# Todos os experimentos sao SOB H0 (gamma = 0, beta1 = 0). Especificacao fixada antes de rodar:
# estatistica = t studentizado (DK Bartlett, L_dk), p = (1 + #{|t*|>=|t|})/(P+1), nivel 5%.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
import sys, time, argparse, importlib
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "revisao" / "saidas"
LOGS = ROOT / "revisao" / "logs"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "revisao"))
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import monte_carlo_perm as mp
from monte_carlo_lp import caminho_choque
d9 = importlib.import_module("09d_mc_ext")

BINS = [0.0, 0.05, 0.10, 0.20, 0.50, 1.0001]      # esperado sob uniforme: 5 / 5 / 10 / 30 / 50 %


def replica_diag(shock, h, N, rho, phi, P, seed, Lperm, Ldk):
    """Mesmo consumo de rng que 09d_mc_ext.replica (simula, depois P block_perm), mas com
    Lperm (bloco da permutacao) e Ldk (banda DK) separados."""
    rng = np.random.default_rng(seed)
    y, zt = d9.simula(shock, h, N, .02, .10, rho, phi, 0.0, rng)
    perms = np.array([mp.block_perm(shock, Lperm, rng) for _ in range(P)])
    b, hh = d9.fit_h(d9.Pre(y, zt), np.vstack([shock[None, :], perms]))
    T = y.shape[1]
    t = b / d9.se_from_h(hh, Ldk, N, T)
    t0, ts = t[0], t[1:]
    pc = (1 + np.sum(np.abs(b[1:]) >= abs(b[0]))) / (P + 1)
    ps = (1 + np.sum(np.abs(ts) >= abs(t0))) / (P + 1)
    # [|t0|, p_coef, p_stud, q95(|t*|)]
    return np.array([abs(t0), pc, ps, np.quantile(np.abs(ts), 0.95)])


def agrega(r):
    n = len(r)
    row = {}
    for nome, v in (("dk_assint", r[:, 0] > 1.959963984540), ("perm_coef", r[:, 1] < .05), ("perm_stud", r[:, 2] < .05)):
        v = v.astype(float)
        row[f"{nome}_5"] = v.mean(); row[f"{nome}_5_epmc"] = v.std(ddof=1) / np.sqrt(n)
    q_verd = float(np.quantile(r[:, 0], 0.95))        # quantil 95% VERDADEIRO de |t| (sobre as replicas, sob H0)
    q_perm = float(r[:, 3].mean())                    # quantil 95% medio da distribuicao de permutacao
    row.update(q95_verdadeiro=q_verd, q95_permutacao=q_perm, razao_perm_verd=q_perm / q_verd)
    h = np.histogram(r[:, 2], bins=BINS)[0] / n
    row.update({f"ps_bin_{BINS[i]:g}_{min(BINS[i + 1], 1):g}": h[i] for i in range(len(h))})
    return row


def rodar(shock, h, N, rho, phi, P, reps, seed0, jobs, Lperm, Ldk):
    r = np.array(Parallel(n_jobs=jobs, batch_size=64)(
        delayed(replica_diag)(shock, h, N, rho, phi, P, seed0 + i, Lperm, Ldk) for i in range(reps)))
    T = len(shock)
    row = dict(T=T, N=N, h=h, rho=rho, phi=phi, P=P, reps=reps, Lperm=Lperm, Ldk=Ldk,
               n_blocos=-(-T // Lperm))
    row.update(agrega(r))
    return row


def gate(a):
    sh = d9.shock_observado(); mx = 0.0
    for h in (0, 3, 5):
        for i in range(3):
            o = d9.replica(sh, h, 300, .02, .10, .9, 0.0, 0.0, 99, a.seed + i)
            n = replica_diag(sh, h, 300, .9, 0.0, 99, a.seed + i, h + 1, h + 1)
            mx = max(mx, abs(o[0] - n[0]), abs(o[3] - n[1]), abs(o[4] - n[2]))
    print(f"[gate] replica_diag(Lperm=Ldk=h+1) vs 09d.replica: max |dif| = {mx:.2e}")
    if mx > 1e-10:
        sys.exit("Gate falhou: cole esta linha.")
    print("[gate] OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["gate", "sanidade", "h", "rho", "bloco", "dk"])
    ap.add_argument("--h", type=int, default=5)
    ap.add_argument("--reps", type=int, default=2000); ap.add_argument("--P", type=int, default=399)
    ap.add_argument("--phi", type=float, default=0.0)
    ap.add_argument("--choque", choices=["obs", "iid", "calibrada"], default="obs",
                    help="serie do choque usada em h/rho/bloco/dk (sanidade usa obs e iid)")
    ap.add_argument("--Lperm", type=int, nargs="+", default=[1, 3, 6, 9])
    ap.add_argument("--seed", type=int, default=2026); ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    if a.exp == "gate": return gate(a)

    iid = caminho_choque("iid")
    obs = {"obs": d9.shock_observado, "iid": lambda: iid, "calibrada": lambda: caminho_choque("calibrada")}[a.choque]()
    cells = []                                         # (rotulo, choque, h, rho, Lperm, Ldk)
    if a.exp == "sanidade":
        for nome, sh in (("observado", obs), ("iid", iid)):
            cells.append((dict(choque=nome), sh, 0, .9, 1, 1))
    elif a.exp == "h":
        cells = [(dict(choque=a.choque), obs, h, .9, h + 1, h + 1) for h in range(0, 8)]
    elif a.exp == "rho":
        cells = [(dict(choque=a.choque), obs, a.h, rho, a.h + 1, a.h + 1) for rho in (0.0, .5, .9, .95)]
    elif a.exp == "bloco":
        cells = [(dict(choque=a.choque), obs, a.h, .9, L, a.h + 1) for L in a.Lperm]
    else:
        cells = [(dict(choque=a.choque), obs, a.h, .9, a.h + 1, Ld) for Ld in (0, a.h + 1, 2 * a.h + 2)]

    out = OUT / f"10_diag_{a.exp}_{a.choque}_h{a.h}_phi{a.phi:g}_reps{a.reps}.csv"
    rows = []
    for k, (lab, sh, h, rho, Lp, Ld) in enumerate(cells):
        t0 = time.time()
        row = dict(exp=a.exp, **lab, **rodar(sh, h, 300, rho, a.phi, a.P, a.reps, a.seed + 100000 * (9000 + k), a.jobs, Lp, Ld))
        rows.append(row); pd.DataFrame(rows).to_csv(out, index=False)
        print(f"{lab} h={h} rho={rho} Lperm={Lp} (blocos={row['n_blocos']}) Ldk={Ld}: "
              f"DK {row['dk_assint_5']:.3f} | coef {row['perm_coef_5']:.3f} | stud {row['perm_stud_5']:.3f} "
              f"(EPMC {row['perm_stud_5_epmc']:.4f}) | q95 perm/verd = {row['razao_perm_verd']:.3f} {time.time() - t0:.0f}s", flush=True)
    print(f"\nSalvo em {out}")


if __name__ == "__main__":
    main()