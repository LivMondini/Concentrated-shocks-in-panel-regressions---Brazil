# revisao/36_fe_tempo.py
# Eq. (1) com efeitos fixos de trimestre (E1) e de setor x trimestre (E2a agregado, E2o original),
# comparada com a Eq. (1) original (E0). Mesma permutacao studentizada de 07_tabela1_stud.py.
# uso (da raiz): python revisao\36_fe_tempo.py [amostra] [h] [--B=N]
#   sem argumentos: as 6 celulas, B = 999, seed 12345.  --B=49 serve para medir tempo (saida com sufixo _B49).
# Escopo desta versao: linhas "Pooled" e "Excluindo 2021" da Tabela 1. Episodios e diferencas ficam para depois.
import sys, time, datetime, warnings, gc
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "revisao" / "saidas"
LOGS = ROOT / "revisao" / "logs"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "revisao"))
import numpy as np, pandas as pd
from linearmodels.panel import PanelOLS
import placebos as pb
from placebos import build, fit_full, cov_kwargs, S0, SL10, G0, ZMAP, V, lag, ano_de_t
from fechamento_b1 import E21
import fe_motor as fm

# ---------------- argumentos ----------------
args = [a for a in sys.argv[1:] if not a.startswith("--")]
NPERM = 999
for a in sys.argv[1:]:
    if a.startswith("--B="):
        NPERM = int(a[4:])
SEED = 12345
CELLS = [(a, h) for a in ("Restrita", "Total") for h in (3, 5, 7)]
if len(args) >= 2:
    CELLS = [(args[0], int(args[1]))]
SUF = "" if NPERM == 999 else f"_B{NPERM}"
E21T = tuple(int(q) for q in E21)

OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
logf = open(LOGS / f"36_fe_tempo{SUF}_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt", "w", encoding="utf-8")
def log(*a):
    s = " ".join(str(x) for x in a); print(s); logf.write(s + "\n"); logf.flush()

# ---------------- regras fixadas ANTES de rodar (copie para 00_prespec.md) ----------------
log("REGRAS: especificacao primaria do manuscrito = Eq. (1) original (E0), Restrita h = 5. E1, E2a e E2o sao robustez;")
log("todas as celulas e todas as variantes sao reportadas, qualquer que seja o resultado. Inferencia: p_stud (B = 999,")
log(f"seed {SEED}, blocos h+1, nao circular), a mesma permutacao em todas as especificacoes. SE DK descritivo.")
log(f"B desta execucao: {NPERM}" + ("  <<< NAO e a execucao reportavel (B != 999)" if NPERM != 999 else ""))

# ---------------- alvos de reproducao (00_gate_reproducao.py: Tabela 1 / A1 / A15) ----------------
REF_ALVO = {
    ("Restrita", 3, "Pooled"): (0.0476, 8920), ("Restrita", 5, "Pooled"): (0.0549, 8817),
    ("Restrita", 7, "Pooled"): (0.0601, 8692), ("Total", 3, "Pooled"): (0.0430, 14029),
    ("Total", 5, "Pooled"): (0.0404, 13720), ("Total", 7, "Pooled"): (0.0323, 13406),
    ("Restrita", 3, "Excluindo 2021"): (0.0223, 8140), ("Restrita", 5, "Excluindo 2021"): (0.0250, 8039),
    ("Restrita", 7, "Excluindo 2021"): (0.0344, 7921), ("Total", 3, "Excluindo 2021"): (0.0025, 12821),
    ("Total", 5, "Excluindo 2021"): (-0.0064, 12526), ("Total", 7, "Excluindo 2021"): (-0.0156, 12233),
}
REF_W2021 = {("Restrita", 3): 0.381, ("Restrita", 5): 0.387, ("Restrita", 7): 0.389,
             ("Total", 3): 0.388, ("Total", 5): 0.395, ("Total", 7): 0.397}   # Tabela A6 (E0)

# ---------------- setor ----------------
PREFIXO = "Emp. Adm. Part. - "
# PENDENTE DE CONFIRMACAO SUA: os tres mapeamentos manuais abaixo (grafia diferente da categoria-alvo).
MAPA_MANUAL = {"Const. Civil, Mat. Const. e Decoração": "Construção Civil, Mat. Constr. e Decoração",
               "Máqs., Equip., Veíc. e Peças": "Máquinas, Equipamentos, Veículos e Peças",
               "Serviços médicos": "Serviços Médicos"}

def garante_setor(d):
    if "SETOR_ATIV" not in d.columns:
        p_ = getattr(pb, "p", None)
        if p_ is None or "SETOR_ATIV" not in p_.columns:
            sys.exit("build() nao devolve SETOR_ATIV e placebos.p nao tem a coluna. Colunas de d: "
                     f"{list(d.columns)}. Cole a definicao de build em placebos.py.")
        m = p_[["CNPJ_CIA", "t", "SETOR_ATIV"]].drop_duplicates(["CNPJ_CIA", "t"])
        d2 = d.merge(m, on=["CNPJ_CIA", "t"], how="left")
        if len(d2) != len(d):
            sys.exit("merge de SETOR_ATIV mudou o numero de linhas: pare e cole este log.")
        d = d2
    if d["SETOR_ATIV"].isna().any():
        sys.exit(f"{int(d['SETOR_ATIV'].isna().sum())} linhas sem SETOR_ATIV na amostra de regressao: pare e cole este log.")
    return d

def setor_agregado(s):
    orig = s.astype(str)
    sem = orig.where(~orig.str.startswith(PREFIXO), orig.str[len(PREFIXO):])
    usados = {k: v for k, v in MAPA_MANUAL.items() if k in set(sem)}
    faltam = [v for v in usados.values() if v not in set(orig)]
    if faltam:
        sys.exit(f"mapeamento manual aponta para setor inexistente na amostra: {faltam}")
    return sem.replace(MAPA_MANUAL)

SPECS = ["E0", "E1", "E2a", "E2o"]
ROT = {"E0": "Eq.(1) original", "E1": "+ efeito fixo de trimestre",
       "E2a": "+ setor agregado x trimestre", "E2o": "+ setor original x trimestre"}

def celulas(d, spec):
    t = d["t"].astype(str)
    if spec == "E0":
        return None
    if spec == "E1":
        return d["t"].to_numpy()
    if spec == "E2o":
        return (d["SETOR_ATIV"].astype(str) + "|" + t).to_numpy()
    return (setor_agregado(d["SETOR_ATIV"]) + "|" + t).to_numpy()

# ---------------- colunas no espaco das observacoes (para conferir contra o linearmodels) ----------------
def cols_obs(d, names, S, SL1):
    ti = d["t"].to_numpy() - G0
    s, sl = S[ti], SL1[ti]
    z = {c: d[c].to_numpy(float) for c in d.columns if c.endswith("_z")}
    out = []
    for v in names:
        if v == "Shock": out.append(s)
        elif v == "Shock_L1": out.append(sl)
        elif v in ZMAP: out.append(s * z[ZMAP[v]])
        else: out.append(z[v])
    return np.column_stack(out)

def lm_fit(d, Xobs, names, kw, spec, cel):
    idx = pd.MultiIndex.from_arrays([d["CNPJ_CIA"].to_numpy(), d["t"].to_numpy()], names=["CNPJ_CIA", "t"])
    y = pd.Series(d["y"].to_numpy(float), index=idx)
    X = pd.DataFrame(Xobs, columns=names, index=idx)
    opts = dict(entity_effects=True, drop_absorbed=True, check_rank=False)
    if spec == "E1":
        opts["time_effects"] = True
    elif spec in ("E2a", "E2o"):
        opts["other_effects"] = pd.DataFrame({"celula": cel}, index=idx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = PanelOLS(y, X, **opts).fit(**kw)
    assert list(r.params.index) == names, "linearmodels absorveu colunas: contraste invalido"
    return float(r.params["Ch_Caixa"]), float(r.std_errors["Ch_Caixa"]), int(r.nobs)

# ---------------- desenho + gates ----------------
def prepara_desenho(AM, H, var, d, x, spec):
    nt = d["t"].nunique(); kw = cov_kwargs(H, nt)
    bw = int(kw.get("bandwidth", min(H + 1, nt - 1))); w = fm.pesos_bartlett(bw)
    cel = celulas(d, spec)
    D = fm.FEDesign(d, cel)
    D.prepara(d, x, ZMAP, usa_choque=(spec == "E0"))
    sq, slq = S0[D.tq - G0], SL10[D.tq - G0]
    if not np.isfinite(sq).all() or (spec == "E0" and not np.isfinite(slq).all()):
        sys.exit(f"choque ausente em trimestres da amostra ({AM} h={H} {var} {spec}): pare e cole este log.")
    X = D.cols(sq, slq)
    Xo = cols_obs(d, D.names, S0, SL10)
    ok = bool(np.allclose(X, D.res_matrix(Xo), atol=1e-8))
    log(f"  [gate base] {var} {spec}: colunas por base == projecao direta: {ok}")
    est, q = D.fit(X)
    lm_est, lm_se, lm_n = lm_fit(d, Xo, D.names, kw, spec, cel)
    ok &= abs(est - lm_est) < 1e-6 and lm_n == len(d)
    log(f"  [gate est] {var} {spec}: beta_C proprio {est:.8f} | linearmodels {lm_est:.8f} | dif {abs(est - lm_est):.1e}")
    var1, melhor = fm.dk_var(q, w, 1.0), None
    for nome, corr in D.candidatos_corr(len(D.names)).items():
        rel = np.sqrt(var1 * corr) / lm_se - 1
        log(f"  [gate DK] {var} {spec}: correcao {nome}: EP proprio/linearmodels - 1 = {rel:+.2e}")
        if melhor is None or abs(rel) < melhor[0]:
            melhor = (abs(rel), nome, corr)
    escolha = (melhor[1], melhor[2]) if melhor[0] < 1e-4 else None
    if escolha is None:
        sys.exit("Gate DK falhou: nenhuma correcao de graus de liberdade reproduz o EP do linearmodels. Cole as linhas [gate DK].")
    if spec == "E0":
        fp = fit_full(d, x, H)
        c_fp, s_fp = float(fp.params["Ch_Caixa"]), float(fp.std_errors["Ch_Caixa"])
        ok &= abs(est - c_fp) < 1e-7 and abs(np.sqrt(var1 * escolha[1]) / s_fp - 1) < 1e-4
        log(f"  [gate fit_full] {var}: coef {c_fp:.8f} | EP {s_fp:.6f}")
        ref = REF_ALVO.get((AM, H, var))
        if ref is None:
            log(f"  [gate alvo] {AM} h={H} {var}: SEM REFERENCIA no script")
        else:
            ok_ref = len(d) == ref[1] and abs(est - ref[0]) <= 0.00006
            ok &= ok_ref
            log(f"  [gate alvo] {AM} h={H} {var}: coef {est:.5f} (ref {ref[0]:.4f}) | N {len(d)} (ref {ref[1]}) -> {'OK' if ok_ref else 'DIFERE'}")
    if not ok:
        sys.exit("Gate falhou nesta celula: nao confie nos resultados. Cole as linhas [gate].")
    D.w, D.corr, D.corr_nome, D.est, D.q, D.X, D.d_n = w, escolha[1], escolha[0], est, q, X, len(d)
    D.nfirm = D.nf
    return D

rows, rows_w = [], []
for AM, H in CELLS:
    t0 = time.time()
    log(f"\n=== {AM} h={H} ===")
    dsg = {}
    for var, kwb in (("Pooled", {}), ("Excluindo 2021", {"excl_q": E21T})):
        d, x = build(AM, H, **kwb)
        d = garante_setor(d)
        log(f" {var}: N={len(d)} firmas={d['CNPJ_CIA'].nunique()} T={d['t'].nunique()}")
        for spec in SPECS:
            dsg[(var, spec)] = prepara_desenho(AM, H, var, d, x, spec)
            D = dsg[(var, spec)]
            if var == "Pooled":
                om, ct, beta = D.pesos_fwl(D.X)
                if abs(ct.sum() - D.est) > 1e-8 or abs(om.sum() - 1) > 1e-8:
                    sys.exit(f"Gate FWL falhou ({AM} h={H} {spec}): cole este log.")
                m21 = np.asarray(ano_de_t(D.tq)) == 2021
                rows_w.append(dict(amostra=AM, h=H, spec=spec, beta_C=beta, omega_2021=float(om[m21].sum()),
                                   share_c_2021=float(ct[m21].sum() / beta), omega_max=float(om.max()),
                                   n_efetivo_omega=float(1 / np.sum(om ** 2))))
                if spec == "E0" and (AM, H) in REF_W2021:
                    dif = abs(rows_w[-1]["omega_2021"] - REF_W2021[(AM, H)])
                    log(f"  [gate A6] omega_2021 {rows_w[-1]['omega_2021']:.4f} (ref {REF_W2021[(AM, H)]:.3f}) "
                        f"-> {'OK' if dif < 0.0006 else 'DIFERE'}")
    log(f" gates ok em {time.time() - t0:.0f}s; rodando {NPERM} permutacoes")

    rng = np.random.default_rng(SEED); vals, L = S0[V], H + 1
    acc = {k: ([], []) for k in dsg}
    for k in range(NPERM):
        g = np.full_like(S0, np.nan); g[V] = fm.perm_series(rng, vals, L); gl = lag(g)
        for key, D in dsg.items():
            sq = g[D.tq - G0]
            slq = gl[D.tq - G0] if D.usa_choque else None
            if not np.isfinite(sq).all() or (D.usa_choque and not np.isfinite(slq).all()):
                sys.exit(f"serie permutada gerou NaN em trimestres da amostra ({key}): cole este log.")
            e_, q_ = D.fit(D.cols(sq, slq))
            acc[key][0].append(e_); acc[key][1].append(fm.dk_var(q_, D.w, D.corr))
        if (k + 1) % 250 == 0:
            log(f"  {k + 1}/{NPERM}, {time.time() - t0:.0f}s")

    for (var, spec), D in dsg.items():
        nb = np.array(acc[(var, spec)][0]); nse = np.sqrt(np.maximum(np.array(acc[(var, spec)][1]), 1e-18))
        se_o = np.sqrt(fm.dk_var(D.q, D.w, D.corr)); t_o = D.est / se_o; tn = nb / nse
        rows.append(dict(amostra=AM, h=H, variante=var, spec=spec, rotulo=ROT[spec], coef=D.est, se_dk=se_o, t_obs=t_o,
                         n=D.d_n, firmas=D.nfirm, celulas_fe=D.nc, correcao_gl=D.corr_nome,
                         p_coef=(1 + np.sum(np.abs(nb) >= abs(D.est))) / (NPERM + 1),
                         p_stud=(1 + np.sum(np.abs(tn) >= abs(t_o))) / (NPERM + 1),
                         null_dp=nb.std(), t_null_q95=float(np.quantile(np.abs(tn), .95))))
    ref07 = OUT / "07_tabela1_stud.csv"
    if ref07.exists() and NPERM == 999:
        r7 = pd.read_csv(ref07)
        for var in ("Pooled", "Excluindo 2021"):
            m = r7[(r7["amostra"] == AM) & (r7["h"] == H) & (r7["linha"] == var)]
            mine = [r for r in rows if r["amostra"] == AM and r["h"] == H and r["variante"] == var and r["spec"] == "E0"][0]
            if len(m):
                dif = abs(float(m["p_stud"].iloc[0]) - mine["p_stud"])
                log(f"[gate perm] {var} E0: p_stud {mine['p_stud']:.4f} | 07_tabela1_stud {float(m['p_stud'].iloc[0]):.4f} "
                    f"-> {'OK' if dif <= 0.0011 else 'DIFERE'}")
            else:
                log(f"[gate perm] {var}: linha nao encontrada em 07_tabela1_stud.csv")
    pd.DataFrame(rows).to_csv(OUT / f"36_fe_tempo{SUF}.csv", index=False)
    pd.DataFrame(rows_w).to_csv(OUT / f"36_pesos_fe{SUF}.csv", index=False)
    log(pd.DataFrame([r for r in rows if r["amostra"] == AM and r["h"] == H]).drop(columns=["amostra", "h", "rotulo"])
        .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    log(f"  {time.time() - t0:.0f}s")
    del dsg, acc; gc.collect()

res = pd.DataFrame(rows)
for var in ("Pooled", "Excluindo 2021"):
    log(f"\n##### {var}: coeficiente e p_stud por especificacao #####")
    log(res[res["variante"] == var].pivot_table(index=["amostra", "h"], columns="spec", values=["coef", "p_stud"])
        .round(4).to_string())
log("\nPesos FWL (Pooled):")
log(pd.DataFrame(rows_w).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
log("\n(p_stud compartilha as mesmas permutacoes em todas as especificacoes; sem correcao de multiplicidade.)")
log("Log em:", logf.name)