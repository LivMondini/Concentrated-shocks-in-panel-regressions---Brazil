# revisao/fe_motor.py
"""
Motor de efeitos fixos absorvidos para a Eq. (1): efeito fixo de firma + efeito fixo de
"celulas" (trimestre, ou setor x trimestre). Usa apenas numpy/pandas e NAO importa nada do
repositorio, para poder ser testado isoladamente.

Ideia: o residuo de qualquer coluna `a` sobre [dummies de firma, dummies de celula] e
    M_F a - M_F D Q D' M_F a ,   Q = pinv(D' M_F D),
onde M_F tira a media da firma e D e a matriz de dummies de celula. Q e calculada uma vez
por desenho (matriz ncelulas x ncelulas); cada residualizacao depois custa uma multiplicacao
matriz-vetor. Como S_t so varia no tempo, a coluna S_t * z e uma soma linear de
1[t = tau] * z: residualizamos essas colunas-base uma vez e, em cada permutacao, a coluna
vira (base @ S_permutado). A DK usa a mesma formula de 07_tabela1_stud.py.
"""
import numpy as np
import pandas as pd


# ---------------- pecas identicas a 04_perm_stud.py / 07_tabela1_stud.py ----------------
def pesos_bartlett(bw):
    return 1 - np.arange(bw + 1) / (bw + 1)


def lagdot(x, y, j):
    return float(x @ y) if j == 0 else float(x[j:] @ y[:-j])


def dk_var(q, w, c):
    return c * sum((1.0 if j == 0 else 2.0 * wj) * lagdot(q, q, j) for j, wj in enumerate(w))


def perm_series(rng, vals, L):
    blocos = [vals[i:i + L] for i in range(0, len(vals), L)]     # nao circular, nao sobreposto
    return np.concatenate([blocos[k] for k in rng.permutation(len(blocos))])


# ---------------- auxiliares ----------------
def _pinv_sim(A, tol=1e-11):
    """pinv de matriz simetrica; 'rtol' existe no numpy >= 2.0, 'rcond' nas versoes antigas."""
    try:
        return np.linalg.pinv(A, hermitian=True, rtol=tol)
    except TypeError:
        return np.linalg.pinv(A, rcond=tol, hermitian=True)


def n_componentes(cod_f, cod_c, nf, nc):
    """Componentes conexas do grafo bipartido firma-celula. posto(FE) = nf + nc - componentes."""
    pai = list(range(nf + nc))

    def raiz(a):
        while pai[a] != a:
            pai[a] = pai[pai[a]]
            a = pai[a]
        return a

    for f, c in zip(cod_f, cod_c):
        ra, rb = raiz(int(f)), raiz(nf + int(c))
        if ra != rb:
            pai[ra] = rb
    return len({raiz(i) for i in range(nf + nc)})


class FEDesign:
    """Um desenho: firma sempre absorvida; `celulas` (rotulos por observacao) opcional."""

    def __init__(self, d, celulas=None):
        self.n = len(d)
        self.t = d["t"].to_numpy()
        self.codes = pd.factorize(d["CNPJ_CIA"])[0]
        self.nf = int(self.codes.max()) + 1
        self.cnt = np.bincount(self.codes, minlength=self.nf).astype(float)
        self.tq, self.tc = np.unique(self.t, return_inverse=True)
        self.T = len(self.tq)
        self.cc, self.nc, self.Q = None, 0, None
        self.rank_fe = self.nf
        if celulas is not None:
            self.cc, uniq = pd.factorize(np.asarray(celulas))
            self.nc = len(uniq)
            C = np.bincount(self.cc * self.nf + self.codes,
                            minlength=self.nc * self.nf).reshape(self.nc, self.nf).astype(float)
            A = np.diag(C.sum(axis=1)) - (C / self.cnt) @ C.T      # D' M_F D
            self.Q = _pinv_sim(A)
            self.rank_fe = self.nf + self.nc - n_componentes(self.codes, self.cc, self.nf, self.nc)
        self.names, self.spec, self.yt, self.usa_choque, self.c = None, None, None, None, None

    # ---- projecao ----
    def dm(self, a):
        return a - (np.bincount(self.codes, weights=a, minlength=self.nf) / self.cnt)[self.codes]

    def res(self, a):
        a1 = self.dm(a)
        if self.cc is None:
            return a1
        u = np.bincount(self.cc, weights=a1, minlength=self.nc)
        return a1 - self.dm((self.Q @ u)[self.cc])

    def res_matrix(self, M):
        return np.column_stack([self.res(M[:, j]) for j in range(M.shape[1])])

    def tsum(self, a):
        return np.bincount(self.tc, weights=a, minlength=self.T)

    # ---- preparo das colunas ----
    def _base(self, z=None):
        R = np.empty((self.n, self.T))
        for k in range(self.T):
            col = (self.tc == k).astype(float)
            R[:, k] = self.res(col if z is None else col * z)
        return R

    def prepara(self, d, x, ZMAP, usa_choque):
        """x: lista de regressores da Eq. (1). Sem choque (usa_choque=False), tira Shock e Shock_L1
        (absorvidos por efeitos fixos de tempo)."""
        self.usa_choque = usa_choque
        self.names = [v for v in x if usa_choque or v not in ("Shock", "Shock_L1")]
        z = {c: d[c].to_numpy(float) for c in d.columns if c.endswith("_z")}
        R1, Rz, self.spec = None, {}, []
        for v in self.names:
            if v in ("Shock", "Shock_L1"):
                if R1 is None:
                    R1 = self._base(None)
                self.spec.append((v, "S" if v == "Shock" else "L", R1))
            elif v in ZMAP:
                zn = ZMAP[v]
                if zn not in Rz:
                    Rz[zn] = self._base(z[zn])
                self.spec.append((v, "S", Rz[zn]))
            else:
                self.spec.append((v, "F", self.res(z[v])))
        self.yt = self.res(d["y"].to_numpy(float))
        self.c = np.zeros(len(self.names))
        self.c[self.names.index("Ch_Caixa")] = 1.0

    def cols(self, sq, slq=None):
        """sq/slq: choque e choque defasado por trimestre do desenho (alinhados a self.tq)."""
        out = []
        for nm, kind, M in self.spec:
            if kind == "F":
                out.append(M)
            elif kind == "S":
                out.append(M @ sq)
            else:
                out.append(M @ slq)
        return np.column_stack(out)

    # ---- estimacao ----
    def fit(self, X):
        XtXi = np.linalg.pinv(X.T @ X)
        b = XtXi @ (X.T @ self.yt)
        e = self.yt - X @ b
        s = X @ (XtXi @ self.c)
        return float(self.c @ b), self.tsum(s * e)

    def candidatos_corr(self, k):
        n = self.n
        return {"G=firmas": n / (n - k - self.nf),
                "G=firmas+celulas": n / (n - k - self.nf - self.nc),
                "G=posto(FE)": n / (n - k - self.rank_fe)}

    def pesos_fwl(self, X, alvo="Ch_Caixa"):
        """Pesos omega_t e contribuicoes c_t (mesma construcao de 01_pesos_fwl.py)."""
        j = self.names.index(alvo)
        Xo = np.delete(X, j, axis=1)
        g = np.linalg.lstsq(Xo, X[:, j], rcond=None)[0]
        r = X[:, j] - Xo @ g
        den = float(r @ r)
        beta = float(r @ self.yt) / den
        return self.tsum(r * r) / den, self.tsum(r * self.yt) / den, beta