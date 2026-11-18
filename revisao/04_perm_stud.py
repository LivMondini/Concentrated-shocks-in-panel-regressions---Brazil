# revisao/04_perm_stud.py
# uso (da raiz): python revisao\04_perm_stud.py [amostra] [h] [definicao]   padrao: Restrita 5 FCA
import sys, time, datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "revisao" / "saidas"
LOGS = ROOT / "revisao" / "logs"
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd

import placebos as pb
from placebos import build, fit_full, S0, SL10, G0, ZMAP, V, lag, cov_kwargs
from ipo import Q1315
from eventos import Q2021

AM = sys.argv[1] if len(sys.argv) > 1 else "Restrita"
H = int(sys.argv[2]) if len(sys.argv) > 2 else 5
DEF = sys.argv[3] if len(sys.argv) > 3 else "FCA"
NPERM, SEED = 999, 12345
REF_SD = {("Restrita", 3): 0.0204, ("Restrita", 5): 0.0230, ("Restrita", 7): 0.0199}  # permutacao_resultados.csv
REF_P = {("Restrita", 3): 0.019, ("Restrita", 5): 0.017, ("Restrita", 7): 0.001}
tag = f"{AM}_h{H}_" + DEF.replace(", ", "_").replace(" ", "_")

OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
logf = open(LOGS / f"04_perm_stud_{tag}_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt", "w", encoding="utf-8")
def log(*a):
    s = " ".join(str(x) for x in a); print(s); logf.write(s + "\n"); logf.flush()

# ---------- entrantes ----------
f = pd.read_csv(OUT / "02_entrantes.csv", dtype={"CNPJ_CIA": str})
ENTS = {"T0": f["ent_T0"], "FCA": f["ent_fca"],
        "FCA, fallback T0": f["ent_fca"] | (~f["match_fca"] & f["ent_T0"])}
if DEF not in ENTS:
    sys.exit(f"definicao invalida: {list(ENTS)}")
ENT = set(f.loc[ENTS[DEF], "CNPJ_CIA"])

# ---------- DK escalar ----------
def weights(bw, var):
    return 1 - np.arange(bw + 1) / (bw + 1) if var == "bw+1" else 1 - np.arange(bw) / bw

def lagdot(x, y, j):
    return float(x @ y) if j == 0 else float(x[j:] @ y[:-j])

def hac_quad(qu, qv, w, c=1.0):
    """Var(g0) = A - g0*Bc + g0^2*C, serie q_t = qu_t - g0*qv_t, pesos Bartlett w."""
    A = Bc = C = 0.0
    for j, wj in enumerate(w):
        m = 1.0 if j == 0 else 2.0 * wj
        A += m * lagdot(qu, qu, j)
        Bc += m * (lagdot(qu, qv, j) + lagdot(qv, qu, j))
        C += m * lagdot(qv, qv, j)
    return c * A, c * Bc, c * C

def cvec(names, spec):
    c = np.zeros(len(names))
    for nm in spec:
        c[names.index(nm)] = 1.0
    return c

def perm_series(rng, vals, L):
    blocos = [vals[i:i + L] for i in range(0, len(vals), L)]     # nao circular, nao sobreposto
    return np.concatenate([blocos[k] for k in rng.permutation(len(blocos))])

class Design:
    def __init__(self, d, x, ent):
        self.x = list(x)
        self.t = d["t"].to_numpy()
        self.ti = self.t - G0
        self.codes = pd.factorize(d["CNPJ_CIA"])[0]
        self.nf = int(self.codes.max()) + 1
        self.cnt = np.bincount(self.codes, minlength=self.nf).astype(float)
        self.z = {c: d[c].to_numpy(float) for c in d.columns if c.endswith("_z")}
        self.rec = d["CNPJ_CIA"].isin(ent).to_numpy(float)
        self.p1 = np.isin(self.t, list(Q1315)).astype(float)
        self.p2 = np.isin(self.t, list(Q2021)).astype(float)
        self.tq, self.tc = np.unique(self.t, return_inverse=True)
        self.T = len(self.tq)
        self.y = self.dm(d["y"].to_numpy(float))

    def dm(self, a):
        return a - (np.bincount(self.codes, weights=a, minlength=self.nf) / self.cnt)[self.codes]

    def dmm(self, M):
        return np.column_stack([self.dm(M[:, j]) for j in range(M.shape[1])])

    def tsum(self, a):
        return np.bincount(self.tc, weights=a, minlength=self.T)

    def cols(self, S, SL1, modelo):
        s, sl = S[self.ti], SL1[self.ti]
        c = {}
        for v in self.x:
            if v == "Shock": c[v] = s
            elif v == "Shock_L1": c[v] = sl
            elif v in ZMAP: c[v] = s * self.z[ZMAP[v]]
            else: c[v] = self.z[v]
        if modelo == "B":       # replica ipo2.add_regimes + ipo2.add_rec(com_p2=True)
            for nm, pp in (("P1", self.p1), ("P2", self.p2)):
                c[nm] = pp
                for v in self.x:
                    if v in ZMAP: c[f"{v}_{nm}"] = c[v] * pp
                c[f"Shock_{nm}"] = s * pp
            cz, rec, p2 = self.z["Cash_Ratio_z"], self.rec, self.p2
            c["Shock_Rec"] = s * rec
            c["Cash_Ratio_z_Rec"] = cz * rec
            c["Ch_Caixa_Rec"] = c["Ch_Caixa"] * rec
            c["P2_Rec"] = p2 * rec
            c["Shock_P2_Rec"] = s * p2 * rec
            c["Cash_Ratio_z_P2_Rec"] = cz * p2 * rec
            c["Ch_Caixa_P2_Rec"] = c["Ch_Caixa"] * p2 * rec
        names = list(c)
        return names, np.column_stack([c[k] for k in names]), c

    def fit(self, X, cons, names, W=None):
        Xd = self.dmm(X)
        XtXi = np.linalg.pinv(Xd.T @ Xd)
        b = XtXi @ (Xd.T @ self.y)
        e = self.y - Xd @ b
        if W is not None:
            Wd = self.dm(W)
            bW = XtXi @ (Xd.T @ Wd)
            eW = Wd - Xd @ bW
        out = {}
        for nm, spec in cons.items():
            c = cvec(names, spec)
            s = Xd @ (XtXi @ c)
            out[nm] = dict(est=float(c @ b), qu=self.tsum(s * e),
                           qv=self.tsum(s * eW) if W is not None else np.zeros(self.T),
                           beta=float(c @ bW) if W is not None else 0.0)
        return out

CP = {"beta_C": ["Ch_Caixa"]}
CB = {"g_inc": ["Ch_Caixa", "Ch_Caixa_P2"],
      "g_ent": ["Ch_Caixa", "Ch_Caixa_Rec", "Ch_Caixa_P2", "Ch_Caixa_P2_Rec"],
      "dif_ent_inc": ["Ch_Caixa_Rec", "Ch_Caixa_P2_Rec"]}
MAPQ = {"g_inc": "B: 2021, incumbentes", "g_ent": "B: 2021, entrantes",
        "dif_ent_inc": "B: 2021, entrantes menos incumbentes"}

# ---------- observado + gate ----------
d, x = build(AM, H)
dd = Design(d, x, ENT)
nt = d["t"].nunique()
kw = cov_kwargs(H, nt)
bw = int(kw.get("bandwidth", min(H + 1, nt - 1)))
log(f"{AM} h={H} def={DEF} | N={len(d)} | T={dd.T} | entrantes no painel: {len(ENT)} | cov_kwargs={kw} | bw={bw}")

nP, XP, _ = dd.cols(S0, SL10, "P")
nB, XB, cB = dd.cols(S0, SL10, "B")
assert np.isfinite(XP).all() and np.isfinite(XB).all(), "NaN no desenho observado"
W_obs = cB["Ch_Caixa_P2"].copy()
oP = dd.fit(XP, CP, nP)
oB = dd.fit(XB, CB, nB, W=W_obs)

fp = fit_full(d, x, H)
ref_est, ref_se = float(fp.params["Ch_Caixa"]), float(fp.std_errors["Ch_Caixa"])
ref = pd.read_csv(OUT / "03_inter_def_completo.csv")
ref = ref[(ref["def"] == DEF) & (ref["amostra"] == AM) & (ref["h"] == H)].set_index("quantidade")
ok = abs(oP["beta_C"]["est"] - ref_est) < 1e-7
log(f"[gate est] beta_C proprio {oP['beta_C']['est']:.8f} | linearmodels {ref_est:.8f}")
for k, q in MAPQ.items():
    dif = abs(oB[k]["est"] - float(ref.loc[q, "estimativa"]))
    ok &= dif < 1e-6
    log(f"[gate est] {k}: proprio {oB[k]['est']:.8f} | 03_inter_def {float(ref.loc[q, 'estimativa']):.8f}")
log(f"[gate est] beta do regressor W no desenho observado (deve ser 1): {oB['g_inc']['beta']:.6f}")
if not ok:
    sys.exit("Gate de estimativas falhou: o desenho proprio nao reproduz o original.")

NOBS, NFIRM = len(d), dd.nf
CORR = {"beta_C": NOBS / (NOBS - XP.shape[1] - NFIRM)}
CORR.update({k: NOBS / (NOBS - XB.shape[1] - NFIRM) for k in CB})
log("[fator] N/(N-k-G): " + ", ".join(f"{k} {v:.5f}" for k, v in CORR.items()))
alvo = {"beta_C": ref_se, **{k: float(ref.loc[q, "se"]) for k, q in MAPQ.items()}}
alvo = {"beta_C": ref_se, **{k: float(ref.loc[q, "se"]) for k, q in MAPQ.items()}}
serie = {"beta_C": oP["beta_C"], **oB}
escolhida = None
for var in ("bw+1", "bw"):
    w_ = weights(bw, var)
    rel = {k: np.sqrt(hac_quad(serie[k]["qu"], np.zeros(dd.T), w_, CORR[k])[0]) / alvo[k] - 1 for k in alvo}
    log(f"[gate DK] pesos Bartlett '{var}': razao-1 por estatistica = " + ", ".join(f"{k} {v:+.2e}" for k, v in rel.items()))
    if escolhida is None and all(abs(v) < 1e-4 for v in rel.values()):
        escolhida = var
if escolhida is None:
    sys.exit("Gate DK falhou: o DK proprio nao reproduz o EP do linearmodels. Cole as linhas [gate DK].")
w = weights(bw, escolhida)
log(f"[gate DK] OK com pesos '{escolhida}'")

# ---------- permutacoes ----------
rng = np.random.default_rng(SEED)
vals, L = S0[V], H + 1
acc = {k: dict(est=[], A=[], Bc=[], C=[], beta=[]) for k in list(CP) + list(CB)}
t0 = time.time()
for k in range(NPERM):
    g = np.full_like(S0, np.nan); g[V] = perm_series(rng, vals, L); gl = lag(g)
    nPb, XPb, _ = dd.cols(g, gl, "P")
    nBb, XBb, _ = dd.cols(g, gl, "B")
    for nm, r in {**dd.fit(XPb, CP, nPb), **dd.fit(XBb, CB, nBb, W=W_obs)}.items():
        A, Bc, C = hac_quad(r["qu"], r["qv"], w, CORR[nm])
        a = acc[nm]
        a["est"].append(r["est"]); a["A"].append(A); a["Bc"].append(Bc); a["C"].append(C); a["beta"].append(r["beta"])
    if (k + 1) % 200 == 0:
        log(f"  {k + 1}/{NPERM} permutacoes, {time.time() - t0:.0f}s")

# ---------- gate da permutacao (esquema de blocos) ----------
sd_nulo = float(np.std(acc["beta_C"]["est"]))
p_c = (1 + np.sum(np.abs(acc["beta_C"]["est"]) >= abs(oP["beta_C"]["est"]))) / (NPERM + 1)
if (AM, H) in REF_SD:
    r_sd = sd_nulo / REF_SD[(AM, H)]
    log(f"[gate perm] DP do nulo de beta_C: {sd_nulo:.4f} (ref {REF_SD[(AM, H)]:.4f}, razao {r_sd:.2f}) | "
        f"p_perm nao studentizado: {p_c:.3f} (ref {REF_P[(AM, H)]:.3f}) -> "
        + ("OK" if 0.85 <= r_sd <= 1.15 else "<<< DIFERENTE: o esquema de blocos pode diferir de permutacao_lp.block_perm; cole essa funcao"))

# ---------- resumo ----------
rows = []
for nm, a in acc.items():
    obs = serie[nm]
    A0, _, _ = hac_quad(obs["qu"], obs["qv"], w, CORR[nm])
    se_o, est_o = np.sqrt(A0), obs["est"]
    nb, nse = np.array(a["est"]), np.sqrt(np.maximum(np.array(a["A"]), 1e-18))
    tn, t_o = nb / nse, est_o / se_o
    rows.append(dict(estatistica=nm, coef=est_o, se_dk_obs=se_o, t_obs=t_o, null_dp=nb.std(),
                     p_coef=(1 + np.sum(np.abs(nb) >= abs(est_o))) / (NPERM + 1),
                     p_stud=(1 + np.sum(np.abs(tn) >= abs(t_o))) / (NPERM + 1),
                     t_null_q95=float(np.quantile(np.abs(tn), .95))))
res = pd.DataFrame(rows)
res.insert(0, "def", DEF); res.insert(0, "h", H); res.insert(0, "amostra", AM)
res.to_csv(OUT / f"04_perm_stud_{tag}.csv", index=False)
log("\n(se_dk_obs de episodio NAO e inferencia valida; serve so de escala. p_stud = permutacao studentizada, B=999)")
log(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# ---------- intervalo de compatibilidade de g_inc por inversao ----------
a = acc["g_inc"]
alpha, beta_ = np.array(a["est"]), np.array(a["beta"])
Aa, Ba, Ca = np.array(a["A"]), np.array(a["Bc"]), np.array(a["C"])
ghat = oB["g_inc"]["est"]
Ao, Bo, Co = hac_quad(oB["g_inc"]["qu"], oB["g_inc"]["qv"], w, CORR["g_inc"])
se_o = np.sqrt(Ao)
grid = ghat + se_o * np.linspace(-40, 40, 801)
est_b = alpha[:, None] - grid[None, :] * beta_[:, None]
var_b = np.maximum(Aa[:, None] - grid[None, :] * Ba[:, None] + grid[None, :] ** 2 * Ca[:, None], 1e-18)
tb = np.abs(est_b) / np.sqrt(var_b)
var_o = np.maximum(Ao - grid * Bo + grid ** 2 * Co, 1e-18)
to = np.abs(ghat - grid) / np.sqrt(var_o)
pg = (1 + (tb >= to[None, :]).sum(axis=0)) / (NPERM + 1)
pd.DataFrame({"g0": grid, "p_stud": pg}).to_csv(OUT / f"04_ci_ginc_{tag}.csv", index=False)

def ic(alfa):
    okk = pg > alfa
    if not okk.any():
        return np.nan, np.nan, "vazio"
    i = np.flatnonzero(okk)
    nota = "contiguo" if okk[i[0]:i[-1] + 1].all() else "NAO contiguo"
    if i[0] == 0 or i[-1] == len(grid) - 1:
        nota += "; TOCA A BORDA DA GRADE"
    return grid[i[0]], grid[i[-1]], nota
for alfa in (0.05, 0.10):
    lo, hi, nota = ic(alfa)
    log(f"IC {int((1 - alfa) * 100)}% de g_inc (inversao): [{lo:.4f}, {hi:.4f}] ({nota})")

# ---------- regra pre-especificada ----------
beta_pool, g_inc, g_ent = oP["beta_C"]["est"], oB["g_inc"]["est"], oB["g_ent"]["est"]
p_inc = float(res.loc[res["estatistica"] == "g_inc", "p_stud"].iloc[0])
if g_inc >= .75 * beta_pool: regra = "nao se sustenta"
elif g_inc < .5 * g_ent and p_inc > .10: regra = "se sustenta"
else: regra = "intermediario: reportar o intervalo"
log(f"\nREGRA: g_inc/pooled = {g_inc / beta_pool:.3f} | g_inc/g_ent = {g_inc / g_ent:.3f} | p_stud(g_inc=0) = {p_inc:.3f} -> {regra}")
log("Log em:", logf.name)