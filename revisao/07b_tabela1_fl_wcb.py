# revisao/07b_tabela1_fl_wcb.py
# p-valores Freedman-Lane (FL), wild cluster bootstrap em blocos de h+1 trimestres (WCB bloco) e por trimestre
# (WCB trim) para as linhas "Pooled" e "Excluindo 2021" da Tabela 1 (07_tabela1_stud). Painel desbalanceado.
# uso (da raiz): python revisao\07b_tabela1_fl_wcb.py [amostra] [h]      padrao: as 6 celulas
# Reaproveita build/cov_kwargs/S0/... de placebos.py e a mesma estatistica do 07 (EP DK, correcao N/(N-k-G)).
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "revisao" / "saidas"
LOGS = ROOT / "revisao" / "logs"
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from placebos import build, cov_kwargs, S0, SL10, G0, ZMAP, V
from fechamento_b1 import E21

NBOOT, SEED = 999, 12345
E21T = tuple(int(q) for q in E21)

def weights(bw): return 1 - np.arange(bw + 1) / (bw + 1)
def lagdot(x, y, j): return float(x @ y) if j == 0 else float(x[j:] @ y[:-j])
def dk_var(q, w, c):
    return c * sum((1.0 if j == 0 else 2.0 * wj) * lagdot(q, q, j) for j, wj in enumerate(w))
def perm_series(rng, vals, L):                       # igual ao 07: blocos nao circulares, nao sobrepostos
    blocos = [vals[i:i + L] for i in range(0, len(vals), L)]
    return np.concatenate([blocos[k] for k in rng.permutation(len(blocos))])

class Cel:
    """Uma celula (amostra, h, subamostra). X e fixo; so o y muda nos bootstraps."""
    def __init__(self, d, x, H):
        self.x, self.n, self.L = list(x), len(d), H + 1
        t = d["t"].to_numpy(); ti = t - G0
        self.codes = pd.factorize(d["CNPJ_CIA"])[0]; self.nf = int(self.codes.max()) + 1
        self.cnt = np.bincount(self.codes, minlength=self.nf).astype(float)
        self.tq, self.tc = np.unique(t, return_inverse=True); self.T = len(self.tq)
        self.pos = np.searchsorted(V, ti); assert (V[self.pos] == ti).all(), "trimestre fora da serie valida do choque"
        self.TV = len(V)
        nt = d["t"].nunique(); self.bw = int(cov_kwargs(H, nt).get("bandwidth", min(H + 1, nt - 1)))
        z = {c: d[c].to_numpy(float) for c in d.columns if c.endswith("_z")}
        s, sl = S0[ti], SL10[ti]; c = {}
        for v in self.x:
            if v == "Shock": c[v] = s
            elif v == "Shock_L1": c[v] = sl
            elif v in ZMAP: c[v] = s * z[ZMAP[v]]
            else: c[v] = z[v]
        self.names = list(c)
        self.y = self.dm(d["y"].to_numpy(float))
        self.Xd = np.column_stack([self.dm(c[k]) for k in self.names]); self.k = self.Xd.shape[1]
        self.inv = np.linalg.pinv(self.Xd.T @ self.Xd)
        self.iC = self.names.index("Ch_Caixa")
        self.corr = self.n / (self.n - self.k - self.nf)
        self.Xg = self.Xd @ self.inv[:, self.iC]
        cols = [j for j in range(self.k) if j != self.iC]          # modelo restrito: beta_C = 0 imposto
        Xr = self.Xd[:, cols]
        self.fit_r = Xr @ np.linalg.lstsq(Xr, self.y, rcond=None)[0]
        self.e_r = self.y - self.fit_r
        self.E = np.full((self.nf, self.TV), np.nan); self.E[self.codes, self.pos] = self.e_r
    def dm(self, a): return a - (np.bincount(self.codes, weights=a, minlength=self.nf) / self.cnt)[self.codes]
    def tsum(self, a): return np.bincount(self.tc, weights=a, minlength=self.T)
    def stat(self, ys, bw):                                          # (coef, t) de Ch_Caixa para um y qualquer (ja within)
        b = self.inv @ (self.Xd.T @ ys); u = ys - self.Xd @ b
        v = dk_var(self.tsum(self.Xg * u), weights(bw), self.corr)
        return float(b[self.iC]), float(b[self.iC] / np.sqrt(max(v, 1e-18)))
    def p_wcb(self, rng, B, bloco):
        L = self.L if bloco else 1; bw = self.bw if bloco else 0
        t0 = abs(self.stat(self.y, bw)[1]); nb = -(-self.TV // L); ts = np.empty(B)
        for k in range(B):
            w = np.repeat(rng.choice([-1.0, 1.0], nb), L)[:self.TV][self.pos]
            ts[k] = self.stat(self.dm(self.fit_r + self.e_r * w), bw)[1]
        return (1 + np.sum(np.abs(ts) >= t0)) / (B + 1)
    def p_fl(self, rng, B):
        t0 = abs(self.stat(self.y, self.bw)[1]); ts = np.empty(B); miss = 0.0
        for k in range(B):
            src = perm_series(rng, np.arange(self.TV), self.L)
            en = self.E[self.codes, src[self.pos]]; m = np.isnan(en); en[m] = 0.0; miss += m.mean()
            ts[k] = self.stat(self.dm(self.fit_r + en), self.bw)[1]
        return (1 + np.sum(np.abs(ts) >= t0)) / (B + 1), miss / B

def gates(c, ref, rotulo):
    ok = True
    b0, t0 = c.stat(c.y, c.bw); se0 = b0 / t0
    ok &= abs(b0 - ref["coef"]) < 1e-8 and abs(se0 / ref["se_dk"] - 1) < 1e-6 and c.n == int(ref["n"])
    print(f"[gate] {rotulo}: coef {b0:.8f} vs {ref['coef']:.8f} | EP DK {se0:.8f} vs {ref['se_dk']:.8f} | n {c.n} vs {int(ref['n'])}")
    d1 = abs(c.stat(c.dm(c.fit_r + c.e_r), c.bw)[1] - t0)
    id_ = np.arange(c.TV); en = c.E[c.codes, id_[c.pos]]
    d2 = max(abs(c.stat(c.dm(c.fit_r + en), c.bw)[1] - t0), float(np.abs(en - c.e_r).max()))
    print(f"[gate] {rotulo}: identidade WCB (w=1) {d1:.1e} | identidade FL (perm. identidade) {d2:.1e}")
    return bool(ok and d1 < 1e-9 and d2 < 1e-9)

def main():
    cells = [(a, h) for a in ("Restrita", "Total") for h in (3, 5, 7)]
    if len(sys.argv) > 2: cells = [(sys.argv[1], int(sys.argv[2]))]
    ref = pd.read_csv(OUT / "07_tabela1_stud.csv"); rows = []
    for AM, H in cells:
        t0 = time.time()
        for linha, ex in (("Pooled", ()), ("Excluindo 2021", E21T)):
            d, x = build(AM, H, excl_q=ex); c = Cel(d, x, H)
            r = ref[(ref["amostra"] == AM) & (ref["h"] == H) & (ref["linha"] == linha)].iloc[0]
            if not gates(c, r, f"{AM} h={H} {linha}"): sys.exit("Gate falhou: nao confie nos p-valores. Cole as linhas [gate].")
            pf, miss = c.p_fl(np.random.default_rng([SEED, 8, 1]), NBOOT)
            rows.append(dict(amostra=AM, h=H, linha=linha, coef=r["coef"], t_obs=r["t_obs"], p_stud_07=r["p_stud"],
                             p_fl=pf, p_wcb_bloco=c.p_wcb(np.random.default_rng([SEED, 8, 2]), NBOOT, True),
                             p_wcb_trim=c.p_wcb(np.random.default_rng([SEED, 8, 3]), NBOOT, False),
                             frac_sem_par_fl=miss, n=c.n, firmas=c.nf, B=NBOOT))
        pd.DataFrame(rows).to_csv(OUT / "07b_tabela1_fl_wcb.csv", index=False)
        print(pd.DataFrame([q for q in rows if q["amostra"] == AM and q["h"] == H]).drop(columns=["amostra", "h"])
              .to_string(index=False, float_format=lambda v: f"{v:.4f}") + f"\n[{AM} h={H}] {time.time() - t0:.0f}s", flush=True)
    print(f"\nSalvo em {OUT / '07b_tabela1_fl_wcb.csv'}")

if __name__ == "__main__":
    main()