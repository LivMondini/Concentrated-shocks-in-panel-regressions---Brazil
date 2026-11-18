# revisao/09f_consolida_mc.py   (da raiz: python revisao\09f_consolida_mc.py)
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "revisao" / "saidas"
LOGS = ROOT / "revisao" / "logs"
frames = []
for modo, reps in (("obs", 10000), ("shift", 10000), ("param", 5000)):
    a = pd.read_csv(OUT / f"09d_mc_{modo}_reps{reps}.csv")
    b = pd.read_csv(OUT / f"09e_fixedb_{modo}_reps{reps}.csv")
    assert len(a) == len(b), f"{modo}: numero de linhas difere ({len(a)} vs {len(b)})"
    for c in ("h", "N", "rho", "phi", "gamma", "T"):
        assert (a[c].to_numpy() == b[c].to_numpy()).all(), f"{modo}: coluna {c} difere entre 09d e 09e"
    assert np.abs(a["dk_5"].to_numpy() - b["dk_5"].to_numpy()).max() < 1e-12, f"{modo}: dk_5 difere"
    a = a.drop(columns=[c for c in a.columns if c.startswith("fixedb_") or c.startswith("crit_fixedb_")])
    fb = b[[c for c in b.columns if c.startswith("fixedb_") or c.startswith("crit_")]].rename(
        columns=lambda c: c.replace("crit_", "crit_fixedb_") if c.startswith("crit_") else c)
    frames.append(pd.concat([a.reset_index(drop=True), fb.reset_index(drop=True)], axis=1))
    print(f"[ok] {modo}: {len(a)} linhas, reps={reps}")
df = pd.concat(frames, ignore_index=True)
df.to_csv(OUT / "09f_mc_consolidado.csv", index=False)
lab = [c for c in ("modo", "cenario", "variante", "d", "h", "N", "rho", "T", "phi", "gamma") if c in df.columns]
for nv in ("5", "1", "10"):
    cols = lab + [f"{k}_{nv}" for k in ("dk", "dk2", "cluster", "fixedb", "perm_coef", "perm_stud")]
    print(f"\n=== nivel nominal {nv}% ===")
    print(df[cols].round(4).to_string(index=False))