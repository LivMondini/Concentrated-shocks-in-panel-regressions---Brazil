# revisao/07_tabela1_stud.py
# uso (da raiz): python revisao\07_tabela1_stud.py [amostra] [h]     padrao: as 6 celulas
import sys, time, datetime, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "revisao" / "saidas"
LOGS = ROOT / "revisao" / "logs"
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from linearmodels.panel import PanelOLS
from placebos import build, cov_kwargs, S0, SL10, G0, ZMAP, V, lag
from fechamento_b1 import E21, E1315, REF_POOL, REF_21, REF_1315, REF_DIF

NPERM, SEED = 999, 12345
CELLS = [(a, h) for a in ("Restrita", "Total") for h in (3, 5, 7)]
if len(sys.argv) > 2:
    CELLS = [(sys.argv[1], int(sys.argv[2]))]
OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
logf = open(LOGS / f"07_tabela1_stud_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt", "w", encoding="utf-8")
def log(*a):
    s = " ".join(str(x) for x in a); print(s); logf.write(s + "\n"); logf.flush()

CONS_P = {"Pooled": {"Ch_Caixa": 1}}
CONS_J = {"Calmo": {"Ch_Caixa": 1},
          "2013-15": {"Ch_Caixa": 1, "Ch_Caixa_P1": 1},
          "2021": {"Ch_Caixa": 1, "Ch_Caixa_P2": 1},
          "Dif 2013-15 menos 2021": {"Ch_Caixa_P1": 1, "Ch_Caixa_P2": -1}}
CONS_X = {"Excluindo 2021": {"Ch_Caixa": 1}}
REFS = {"Pooled": REF_POOL, "2021": REF_21, "2013-15": REF_1315, "Dif 2013-15 menos 2021": REF_DIF}
E21T = tuple(int(q) for q in E21)

def weights(bw): return 1 - np.arange(bw + 1) / (bw + 1)
def lagdot(x, y, j): return float(x @ y) if j == 0 else float(x[j:] @ y[:-j])
def dk_var(q, w, c):
    return c * sum((1.0 if j == 0 else 2.0 * wj) * lagdot(q, q, j) for j, wj in enumerate(w))
def perm_series(rng, vals, L):
    blocos = [vals[i:i + L] for i in range(0, len(vals), L)]     # nao circular, nao sobreposto
    return np.concatenate([blocos[k] for k in rng.permutation(len(blocos))])

class Design:
    def __init__(self, d, x, H):
        self.x, self.n = list(x), len(d)
        self.t = d["t"].to_numpy(); self.ti = self.t - G0
        self.codes = pd.factorize(d["CNPJ_CIA"])[0]; self.nf = int(self.codes.max()) + 1
        self.cnt = np.bincount(self.codes, minlength=self.nf).astype(float)
        self.z = {c: d[c].to_numpy(float) for c in d.columns if c.endswith("_z")}
        self.p1 = np.isin(self.t, E1315).astype(float); self.p2 = np.isin(self.t, E21).astype(float)
        self.tq, self.tc = np.unique(self.t, return_inverse=True); self.T = len(self.tq)
        nt = d["t"].nunique(); self.kw = cov_kwargs(H, nt)
        self.w = weights(int(self.kw.get("bandwidth", min(H + 1, nt - 1))))
        self.y = self.dm(d["y"].to_numpy(float))
    def dm(self, a): return a - (np.bincount(self.codes, weights=a, minlength=self.nf) / self.cnt)[self.codes]
    def tsum(self, a): return np.bincount(self.tc, weights=a, minlength=self.T)
    def cols(self, S, SL1, reg):
        s, sl = S[self.ti], SL1[self.ti]
        c = {}
        for v in self.x:
            if v == "Shock": c[v] = s
            elif v == "Shock_L1": c[v] = sl
            elif v in ZMAP: c[v] = s * self.z[ZMAP[v]]
            else: c[v] = self.z[v]
        if reg:   # replica fechamento_b1.joint (P1=E1315, P2=E21), sem termos de entrante
            for nm, pp in (("P1", self.p1), ("P2", self.p2)):
                c[nm] = pp
                for v in self.x:
                    if v in ZMAP: c[f"{v}_{nm}"] = c[v] * pp
                c[f"Shock_{nm}"] = s * pp
        names = list(c)
        return names, np.column_stack([c[k] for k in names])
    def run(self, S, SL1, reg, cons):
        names, X = self.cols(S, SL1, reg)
        Xd = np.column_stack([self.dm(X[:, j]) for j in range(X.shape[1])])
        XtXi = np.linalg.pinv(Xd.T @ Xd)
        b = XtXi @ (Xd.T @ self.y); e = self.y - Xd @ b
        corr = self.n / (self.n - X.shape[1] - self.nf)         # correcao N/(N-k-G) do linearmodels
        out = {}
        for nm, spec in cons.items():
            c = np.zeros(len(names))
            for v, wt in spec.items(): c[names.index(v)] = wt
            out[nm] = (float(c @ b), dk_var(self.tsum((Xd @ (XtXi @ c)) * e), self.w, corr))
        return out, names, X

def lm_check(d, X, names, cons, kw):
    idx = d.set_index(["CNPJ_CIA", "t"]).index
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = PanelOLS(pd.Series(d["y"].to_numpy(float), index=idx), pd.DataFrame(X, columns=names, index=idx),
                     entity_effects=True, drop_absorbed=True, check_rank=False).fit(**kw)
    assert list(r.params.index) == names, "linearmodels absorveu colunas: mapeamento de contrastes invalido"
    P, C = r.params.to_numpy(), r.cov.to_numpy(); out = {}
    for nm, spec in cons.items():
        c = np.zeros(len(names))
        for v, wt in spec.items(): c[names.index(v)] = wt
        out[nm] = (float(c @ P), float(np.sqrt(c @ C @ c)))
    return out

try: sdref = pd.read_csv(ROOT / "permutacao_resultados.csv")
except Exception: sdref = None
rows = []
for AM, H in CELLS:
    t0 = time.time()
    d, x = build(AM, H); dx, xx = build(AM, H, excl_q=E21T)
    D, DX = Design(d, x, H), Design(dx, xx, H)
    oP, nP, XP = D.run(S0, SL10, False, CONS_P); oJ, nJ, XJ = D.run(S0, SL10, True, CONS_J)
    oX, nX, XX = DX.run(S0, SL10, False, CONS_X)
    obs = {**oP, **oJ, **oX}
    log(f"\n=== {AM} h={H} | N={D.n} firmas={D.nf} | sem 2021: N={DX.n} firmas={DX.nf} | kw={D.kw}")
    ar, ok = ("Cheia" if AM == "Total" else AM), True
    for nm, ref in REFS.items():
        dif_ = abs(obs[nm][0] - ref[(ar, H)]); ok &= dif_ < 2e-6
        log(f"[gate est] {nm}: proprio {obs[nm][0]:.6f} | ref {ref[(ar, H)]:.6f}")
    for dsub, X, names, cons, o, DD in ((d, XP, nP, CONS_P, oP, D), (d, XJ, nJ, CONS_J, oJ, D), (dx, XX, nX, CONS_X, oX, DX)):
        lm = lm_check(dsub, X, names, cons, DD.kw)
        for nm in cons:
            rel = np.sqrt(o[nm][1]) / lm[nm][1] - 1; ok &= abs(rel) < 1e-4 and abs(o[nm][0] - lm[nm][0]) < 1e-7
            log(f"[gate DK] {nm}: EP proprio/linearmodels - 1 = {rel:+.2e}")
    if not ok: sys.exit("Gate falhou nesta celula: nao confie nas permutacoes. Cole as linhas [gate].")
    rng = np.random.default_rng(SEED); vals, L = S0[V], H + 1
    acc = {nm: ([], []) for nm in obs}
    for k in range(NPERM):
        g = np.full_like(S0, np.nan); g[V] = perm_series(rng, vals, L); gl = lag(g)
        r = {**D.run(g, gl, False, CONS_P)[0], **D.run(g, gl, True, CONS_J)[0], **DX.run(g, gl, False, CONS_X)[0]}
        for nm, (e_, v_) in r.items(): acc[nm][0].append(e_); acc[nm][1].append(v_)
        if (k + 1) % 250 == 0: log(f"  {k + 1}/{NPERM}, {time.time() - t0:.0f}s")
    for nm, (est_o, var_o) in obs.items():
        nb, nse = np.array(acc[nm][0]), np.sqrt(np.maximum(np.array(acc[nm][1]), 1e-18))
        se_o = np.sqrt(var_o); t_o = est_o / se_o; tn = nb / nse
        nn, ff = (DX.n, DX.nf) if nm in CONS_X else (D.n, D.nf)
        rows.append(dict(amostra=AM, h=H, linha=nm, coef=est_o, se_dk=se_o, t_obs=t_o, n=nn, firmas=ff,
                         p_coef=(1 + np.sum(np.abs(nb) >= abs(est_o))) / (NPERM + 1),
                         p_stud=(1 + np.sum(np.abs(tn) >= abs(t_o))) / (NPERM + 1),
                         null_dp=nb.std(), t_null_q95=float(np.quantile(np.abs(tn), .95))))
        if sdref is not None and nm in ("Pooled", "Excluindo 2021", "2013-15", "2021", "Calmo"):
            m = sdref[(sdref["amostra"] == ar) & (sdref["h"] == H) & (sdref["estatistica"] == nm)]
            if len(m): rows[-1]["razao_dp_vs_permutacao_lp"] = nb.std() / float(m["sd_perm"].iloc[0])
    pd.DataFrame(rows).to_csv(OUT / "07_tabela1_stud.csv", index=False)
    log(pd.DataFrame([r for r in rows if r["amostra"] == AM and r["h"] == H]).drop(columns=["amostra", "h"])
        .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
log("\n(episodios: se_dk e so escala; a calibracao vem da permutacao. p_coef = estatistica antiga, so para o apendice.)")
log("Log em:", logf.name)