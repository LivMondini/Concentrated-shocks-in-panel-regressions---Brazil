# How many periods identify the interaction? Concentrated shocks in panel regressions -- Brazil

Replication code for the Brazilian results of the paper *Concentrated shocks in panel regressions* (listed Brazilian firms, 2011Q2–2024Q3; local projections of the aggregate monetary shock × cash interaction).

Each table and figure of the paper and of the online appendix is mapped to a script and an output file in [section 5](#5-map-of-tables-and-figures).

## 1. Repository layout

```
main.py                     builds the firm-quarter panel (pipeline, step 1)
script_painel.py, calcular_q.py, ortogonalizacao.py, capex_bp.py, projecoes_locais.py
placebos.py, ipo.py, eventos.py, ciclos.py,
fechamento_b1.py, fechamento_letter.py, permutacao_lp.py,
monte_carlo_lp.py, monte_carlo_perm.py      helper modules used by the analysis
revisao/                    scripts that produce the paper's tables and figures
revisao/saidas/             small output tables (CSV) used in the paper
```

All paths are relative to the repository root; run every command from that folder.

## 2. Environment

Python 3.13. Install the pinned dependencies:

```
python -m venv .venv
# Windows: .venv\Scripts\activate        Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Main packages: numpy 2.5.4, pandas 3.0.6, statsmodels 0.15.0, linearmodels 7.0, matplotlib 3.11.2, requests 2.34.2, scipy, joblib. Random seeds are fixed (section 6); with the same package versions the results are reproducible.

## 3. Data (not included)

Raw data are **not redistributed**. Download them from the original sources and respect their terms of use.

| Input | Source | Location |
| --- | --- | --- |
| Financial statements (DFP/ITR) and company registry | CVM open data, https://dados.cvm.gov.br/ | `dados_cvm/` |
| Share prices | B3 historical quotations (COTAHIST), https://www.b3.com.br; built with `precos_b3.py`, `tickers.py` | `precos_trimestrais.csv`, `tickers.csv` |
| Macro series (selic, fiscal risk, EMBI+, VIX) | Banco Central do Brasil (SGS API), Ipeadata, Yahoo Finance | `vix_trimestral.csv`, `risco_fiscal_trimestral.csv`, `epu_brasil.csv` |
| COPOM meeting dates; DI-swap 360d | BCB / B3 | `datas_copom.csv`, `swap_di_pre_360_diario.csv` |
| Focus expectations | BCB Focus | `revisao/saidas/13_focus_reunioes.csv` |

## 4. How to run

### Step 1 — build the panel

```
python main.py
```

Runs, in order: `script_painel.py` → `calcular_q.py` → `script_painel.py` → `calcular_q.py` → `ortogonalizacao.py` → `capex_bp.py` → `projecoes_locais.py`. The first two scripts are run twice (two passes; the second pass uses the output of the first). Main output: `painel_regressao_q_bp.csv`. A log is written to `pipeline_execution.log`.

### Step 2 — tables and figures

```
python revisao/31_numeros_choque.py       # concentration of the shock
python revisao/01_pesos_fwl.py            # FWL weights of 2021
python revisao/06_blocos4.py              # four-quarter block exclusions
python revisao/06f_q4_vs_resto.py
python revisao/07_tabela1_stud.py         # pooled and episode coefficients, studentized permutation
python revisao/04_perm_stud.py
python revisao/07b_tabela1_fl_wcb.py      # alternative p-values
python revisao/07c_tabela1_esquemas.py
python revisao/36_fe_tempo.py             # time fixed effects
python revisao/09i_grade_concentracao.py  # size by concentration (T = 54, 108)
python revisao/10_diag_tamanho.py         # size diagnostics
python revisao/11_causa_distorcao.py
python revisao/12_poder.py                # power and minimum detectable effect
python revisao/05_tabelas_artigo.py       # formats Table 1
```

Monte Carlo scripts (`09b`, `09d`, `09e`, `09f`, `09g`) take the number of replications and the scenario as arguments; the exact command used for each run is recorded in `revisao/logs/`. Runs with 10,000 replications take a long time.

## 5. Map of tables and figures

| Paper item | Script | Output (`revisao/saidas/`) |
| --- | --- | --- |
| Table 1 | `01_pesos_fwl.py`, `05_tabelas_artigo.py` | `01_pesos_resumo.csv`, `05_tabelas_artigo.md` |
| Table 2, Table A1 (joint model with the 2021 episode term) | `07_tabela1_stud.py` | `07_tabela1_stud.csv` |
| Figure 1 (data) | `31_numeros_choque.py`, `06_blocos4.py` | `31_numeros_choque.csv`, `06_blocos4.csv` |
| Figure 2 (data) | `09f_consolida_mc.py` | `09f_mc_consolidado.csv` |
| Table A2 (concentration by year and episode) | `31_numeros_choque.py` | `31_numeros_choque.csv` |
| Table A3 (FWL weight of 2021 and block exclusions) | `01_pesos_fwl.py`, `06_blocos4.py`, `06f_q4_vs_resto.py` | `01_pesos_*.csv`, `06_blocos4.csv`, `06f_q4_vs_resto.csv` |
| Tables A4, A5 (time effects) | `36_fe_tempo.py` | `36_fe_tempo.csv`, `36_pesos_fe*.csv` |
| Tables A6, A7, A9, A10, A11 (Monte Carlo) | `09d_mc_ext.py`, `09e_fixedb_corrigido.py`, `09f_consolida_mc.py` | `09f_mc_consolidado.csv` |
| Table A8 (size by shock concentration, T = 54, 108) | `09i_grade_concentracao.py` | `09i_grade_T54_*.csv`, `09i_grade_T108_*.csv` |
| Table A12 (p-values by method and permutation scheme) | `07b_tabela1_fl_wcb.py`, `07c_tabela1_esquemas.py` | `07b_tabela1_fl_wcb.csv`, `07c_tabela1_esquemas.csv` |
| Tables A13, A14 (size by horizon, block partition, persistence) | `10_diag_tamanho.py` | `10_diag_h_*.csv`, `10_diag_rho_*.csv`, `10_diag_bloco_*.csv` |
| Tables A15, A16 (dependence structures; analytical size) | `11_causa_distorcao.py` | `11_interv_reps10000.csv` |
| Table A17 (size-adjusted power, minimum detectable effect) | `12_poder.py` | `12_poder_reps5000.csv` |
| Table A18 (United States and Brazil), Table A19 (episode contrasts) | Brazil column of A18: `31_numeros_choque.py`; US column and A19: see the US repository | `31_numeros_choque.csv` |

## 6. Seeds

| Seed | Used in |
| --- | --- |
| 12345 | permutation tests (`placebos.py`, `permutacao_lp.py`, `04_perm_stud.py`, `07*_tabela1*.py`, `36_fe_tempo.py`) |
| 2026 | Monte Carlo simulations (`monte_carlo_lp.py`, `09*`, `10`, `11`, `12`) |

## 7. Check your run

| Quantity | Expected | File |
| --- | --- | --- |
| Effective number of quarters (inverse Herfindahl of squared deviations) | 9.8 of 54 | `31_numeros_choque.csv`, `n_ef_hhi` = 9.797 |
| Share of 2021 in squared deviations | 34.5% | `31_numeros_choque.csv`, `share_2021` = 0.3446 |
| FWL weight of 2021Q1–Q4 | 0.38–0.40 | `01_pesos_resumo.csv`, `omega_2021` |
| Size at N_eff = 9.8, h = 3 (DK / DK 2h+2 / cluster / permutation) | 13.6 / 16.0 / 8.6 / 5.7 | `09i_grade_T54_*.csv` (Table A8) |

## 8. License

Code: MIT (see `LICENSE`). The data belong to their original providers; see their terms of use.
