# revisao/07c_tabela1_esquemas.py
# Refaz os p-valores da permutacao studentizada (linhas "Pooled" e "Excluindo 2021" da Tabela 1) com ESQUEMAS de
# permutacao alternativos ao do 07: nc2 (blocos 2(h+1)), circ (deslocamento circular + blocos h+1), shift (so deslocamento
# circular) e iid (sem blocos). O esquema nc (blocos h+1 nao circulares) e o do 07: serve de gate (reproduz p_stud do CSV).
# uso (da raiz): python revisao\07c_tabela1_esquemas.py [amostra] [h]      padrao: as 6 celulas
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "revisao" / "saidas"
LOGS = ROOT / "revisao" / "logs"
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from placebos import build, cov_kwargs, S0, SL10, G0, ZMAP, V, lag
from fechamento_b1 import E21

NPERM, SEED = 999, 12345
E21T = tuple(int(q) for q in E21)
ESQ = ["nc", "nc2", "circ", "shift", "iid"]

def weights(bw): return 1 - np.arange(bw + 1) / (bw + 1)
def lagdot(x, y, j): return float(x @ y) if j == 0 else float(x[j:] @ y[:-j])
def dk_var(q, w, c): return c * sum((1.0 if j == 0 else 2.0 * wj) * lagdot(q, q, j) for j, wj in enumerate(w))
def perm_series(rng, vals, L):                       # copia literal do 07 (blocos nao circulares, nao sobrepostos)
    blocos = [vals[i:i + L] for i in range(0, len(vals), L)]
    return np.concatenate([blocos[k] for k in rng.permutation(len(blocos))])

def esquema(nome, vals, L, rng):
    n = len(vals)
    if nome == "nc": return perm_series(rng, vals, L)
    if nome == "nc2": return perm_series(rng, vals, 2 * L)
    if nome == "circ": return perm_series(rng, np.roll(vals, -int(rng.integers(n))), L)
    if nome == "shift": return np.roll(vals, int(rng.integers(1, n)))
    return rng.permutation(vals)

class Des:
    """Mesma estatistica do 07 (EP DK com bandwidth do cov_kwargs, correcao N/(N-k-G)), so para Ch_Caixa."""
    def __init__(self, d, x, H):
        self.x, self.n = list(x), len(d)
        self.ti = d["t"].to_numpy() - G0
        self.codes = pd.factorize(d["CNPJ_CIA"])[0]; self.nf = int(self.codes.max()) + 1
        self.cnt = np.bincount(self.codes, minlength=self.nf).astype(float)
        self.z = {c: d[c].to_numpy(float) for c in d.columns if c.endswith("_z")}
        self.tq, self.tc = np.unique(d["t"].to_numpy(), return_inverse=True); self.T = len(self.tq)
        nt = d["t"].nunique(); self.w = weights(int(cov_kwargs(H, nt).get("bandwidth", min(H + 1, nt - 1))))
        self.y = self.dm(d["y"].to_numpy(float))
    def dm(self, a): return a - (np.bincount(self.codes, weights=a, minlength=self.nf) / self.cnt)[self.codes]
    def tsum(self, a): return np.bincount(self.tc, weights=a, minlength=self.T)
    def run(self, S, SL1):
        s, sl = S[self.ti], SL1[self.ti]; cols = {}
        for v in self.x:
            if v == "Shock": cols[v] = s
            elif v == "Shock_L1": cols[v] = sl
            elif v in ZMAP: cols[v] = s * self.z[ZMAP[v]]
            else: cols[v] = self.z[v]
        names = list(cols); Xd = np.column_stack([self.dm(cols[k]) for k in names])
        XtXi = np.linalg.pinv(Xd.T @ Xd); b = XtXi @ (Xd.T @ self.y); e = self.y - Xd @ b
        corr = self.n / (self.n - Xd.shape[1] - self.nf); iC = names.index("Ch_Caixa")
        return float(b[iC]), dk_var(self.tsum((Xd @ XtXi[:, iC]) * e), self.w, corr)

def p_esquema(nome, designs, H, B, seed_rng):
    vals, L = S0[V], H + 1; ts = [[] for _ in designs]
    for k in range(B):
        g = np.full_like(S0, np.nan); g[V] = esquema(nome, vals, L, seed_rng); gl = lag(g)
        for i, D in enumerate(designs):
            e, v = D.run(g, gl); ts[i].append(e / np.sqrt(max(v, 1e-18)))
    return [np.array(t) for t in ts]

def main():
    cells = [(a, h) for a in ("Restrita", "Total") for h in (3, 5, 7)]
    if len(sys.argv) > 2: cells = [(sys.argv[1], int(sys.argv[2]))]
    ref = pd.read_csv(OUT / "07_tabela1_stud.csv"); rows = []
    for AM, H in cells:
        t0 = time.time(); ds = {}
        for linha, ex in (("Pooled", ()), ("Excluindo 2021", E21T)):
            d, x = build(AM, H, excl_q=ex); ds[linha] = Des(d, x, H)
        refs = {l: ref[(ref["amostra"] == AM) & (ref["h"] == H) & (ref["linha"] == l)].iloc[0] for l in ds}
        obs = {}
        for l, D in ds.items():
            e, v = D.run(S0, SL10); obs[l] = e / np.sqrt(v); r = refs[l]
            ok = abs(e - r["coef"]) < 1e-8 and abs(np.sqrt(v) / r["se_dk"] - 1) < 1e-6
            print(f"[gate est] {AM} h={H} {l}: coef {e:.8f} vs {r['coef']:.8f} | EP DK {np.sqrt(v):.8f} vs {r['se_dk']:.8f}")
            if not ok: sys.exit("Gate falhou: estatistica observada nao reproduz o CSV do 07. Cole as linhas [gate].")
        res = {l: dict(amostra=AM, h=H, linha=l, t_obs=obs[l], B=NPERM) for l in ds}
        for j, nome in enumerate(ESQ):
            rng = np.random.default_rng(SEED if nome == "nc" else [SEED, 11, j])      # nc: mesmo rng do 07
            tp = p_esquema(nome, list(ds.values()), H, NPERM, rng)
            for (l, _), t in zip(ds.items(), tp):
                res[l][f"p_{nome}"] = (1 + np.sum(np.abs(t) >= abs(obs[l]))) / (NPERM + 1)
            if nome == "nc":
                dif = max(abs(res[l]["p_nc"] - refs[l]["p_stud"]) for l in ds)
                print(f"[gate] nc reproduz p_stud do 07: max |dif| = {dif:.1e}")
                if dif > 1e-12: sys.exit("Gate falhou: o esquema nc nao reproduz o p_stud do 07. Cole as linhas [gate].")
        rows += list(res.values()); pd.DataFrame(rows).to_csv(OUT / "07c_tabela1_esquemas.csv", index=False)
        print(pd.DataFrame(list(res.values())).drop(columns=["amostra", "h", "B"]).to_string(index=False, float_format=lambda v: f"{v:.3f}")
              + f"\n[{AM} h={H}] {time.time() - t0:.0f}s", flush=True)
    print(f"\nSalvo em {OUT / '07c_tabela1_esquemas.csv'}")

if __name__ == "__main__":
    main()