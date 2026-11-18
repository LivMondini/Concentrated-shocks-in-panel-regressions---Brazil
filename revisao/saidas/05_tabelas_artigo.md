### T1. Pesos FWL exatos de beta_C (omega_t = peso do trimestre; share_c = contribuicao de 2021 a beta_C)
| amostra | h | beta_C | omega_2021 | omega_2021Q4 | share_c_2021 | between/beta_C |
|---|---|---|---|---|---|---|
| Restrita | 3 | 0.048 | 0.381 | 0.319 | 0.792 | 0.083 |
| Restrita | 5 | 0.055 | 0.387 | 0.324 | 0.855 | 0.088 |
| Restrita | 7 | 0.060 | 0.389 | 0.325 | 0.818 | 0.110 |
| Total | 3 | 0.043 | 0.388 | 0.322 | 0.861 | 0.062 |
| Total | 5 | 0.040 | 0.395 | 0.328 | 1.030 | 0.064 |
| Total | 7 | 0.032 | 0.397 | 0.330 | 1.235 | 0.076 |
*between/beta_C = parte de beta_C vinda de comparar trimestres (sem efeitos de tempo).*

### T2. Gradiente de 2021 entre incumbentes (g_inc) e entrantes (g_ent), estimativas pontuais
| def | amostra | h | g_inc | g_ent | obs_ent | inc/pooled | inc/ent | regra |
|---|---|---|---|---|---|---|---|---|
| FCA | Restrita | 3 | 0.040 | 0.052 | 718 | 0.839 | 0.765 | nao se sustenta |
| FCA | Restrita | 5 | 0.056 | 0.065 | 706 | 1.017 | 0.856 | nao se sustenta |
| FCA | Restrita | 7 | 0.041 | 0.081 | 692 | 0.686 | 0.509 | intermediario |
| FCA | Total | 3 | 0.064 | 0.076 | 1239 | 1.478 | 0.834 | nao se sustenta |
| FCA | Total | 5 | 0.071 | 0.075 | 1217 | 1.760 | 0.953 | nao se sustenta |
| FCA | Total | 7 | 0.061 | 0.094 | 1194 | 1.897 | 0.654 | nao se sustenta |
| FCA, fallback T0 | Restrita | 3 | 0.040 | 0.052 | 718 | 0.839 | 0.765 | nao se sustenta |
| FCA, fallback T0 | Restrita | 5 | 0.056 | 0.065 | 706 | 1.017 | 0.856 | nao se sustenta |
| FCA, fallback T0 | Restrita | 7 | 0.041 | 0.081 | 692 | 0.686 | 0.509 | intermediario |
| FCA, fallback T0 | Total | 3 | 0.062 | 0.042 | 1464 | 1.452 | 1.497 | nao se sustenta |
| FCA, fallback T0 | Total | 5 | 0.075 | 0.038 | 1420 | 1.865 | 1.961 | nao se sustenta |
| FCA, fallback T0 | Total | 7 | 0.063 | 0.065 | 1376 | 1.938 | 0.957 | nao se sustenta |
| T0 | Restrita | 3 | 0.038 | 0.047 | 803 | 0.793 | 0.799 | nao se sustenta |
| T0 | Restrita | 5 | 0.052 | 0.054 | 789 | 0.946 | 0.958 | nao se sustenta |
| T0 | Restrita | 7 | 0.030 | 0.073 | 772 | 0.503 | 0.414 | se sustenta* |
| T0 | Total | 3 | 0.067 | 0.028 | 1622 | 1.559 | 2.382 | nao se sustenta |
| T0 | Total | 5 | 0.083 | 0.026 | 1571 | 2.047 | 3.167 | nao se sustenta |
| T0 | Total | 7 | 0.068 | 0.044 | 1516 | 2.121 | 1.538 | nao se sustenta |
*Entrante primario = listagem FCA em 2019Q1-2021Q4. DK de 2021 nao e inferencia valida.*

### T3. Permutacao studentizada (blocos h+1 nao circulares, B=999, seed 12345)
| definicao | amostra | h | beta_C | p_perm_beta | p_stud_beta | t95_nula | g_inc | p_stud_ginc | ic95_lo | ic95_hi | p_stud_gent | p_stud_dif |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FCA | Restrita | 3 | 0.048 | 0.025 | 0.026 | 3.154 | 0.040 | 0.195 | -0.077 | 0.102 | 0.542 | 0.861 |
| FCA | Restrita | 5 | 0.055 | 0.026 | 0.023 | 2.862 | 0.056 | 0.047 | 0.005 | 0.116 | 0.529 | 0.947 |
| FCA | Restrita | 7 | 0.060 | 0.001 | 0.022 | 3.193 | 0.041 | 0.518 | -0.020 | 0.092 | 0.177 | 0.807 |
| FCA | Total | 3 | 0.043 | 0.115 | 0.066 | 3.198 | 0.064 | 0.402 | -0.101 | 0.186 | 0.566 | 0.835 |
| FCA | Total | 5 | 0.040 | 0.164 | 0.097 | 3.012 | 0.071 | 0.595 | -0.145 | 0.257 | 0.636 | 0.983 |
| FCA | Total | 7 | 0.032 | 0.221 | 0.207 | 2.877 | 0.061 | 0.759 | -0.069 | 0.178 | 0.552 | 0.954 |
| FCA, fallback T0 | Total | 3 | 0.043 | 0.115 | 0.066 | 3.198 | 0.062 | 0.357 | -0.107 | 0.164 | 0.760 | 0.814 |
| FCA, fallback T0 | Total | 5 | 0.040 | 0.164 | 0.097 | 3.012 | 0.075 | 0.514 | -0.133 | 0.267 | 0.779 | 0.765 |
| FCA, fallback T0 | Total | 7 | 0.032 | 0.221 | 0.207 | 2.877 | 0.063 | 0.725 | -0.073 | 0.181 | 0.678 | 0.993 |
*t95_nula = percentil 95 de |t| na nula (DK nominal: 1,96). IC por inversao da permutacao studentizada.*
