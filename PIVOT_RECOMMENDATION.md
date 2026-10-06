# Research Pivot: From One-Day Residual Forecasts to Level Relative Value and Residual-Path Signals

Date: 2026-10-06
Context: `step3_yield_ipca_k3_v1`, v1.1 audit (revise-v14); `RESEARCH_PAPER_IPCA_RESIDUAL_MM.md`
Papers discussed: Saha et al. (BlackRock, 2024), "Machine Learning-based Relative Valuation of Municipal Bonds", arXiv:2408.02273; Epstein, Yu and Pelger (2025), "Attention Factors for Statistical Arbitrage"; Guijarro-Ordonez, Pelger and Zanotti (2022), "Deep Learning Statistical Arbitrage".

> Access note. arXiv and its mirrors are blocked from the research environment, so the descriptions of both papers below are reconstructed from their abstracts, search snippets and prior reading. Treat method details as a reconstruction, not quotation, and verify against the PDFs before citing.

## 1. Where v1.1 leaves us

The audited results say the one-day residual is real but is not the right object for a quote overlay.

| Finding | Number | Implication |
|---|---:|---|
| Residual predicts trade vs prior evaluated mark (date x side x size neutralised) | +4.83 bp/rank, t 4.45 | The residual is information about the mark |
| Residual predicts trade vs algo quote | +0.46 bp/rank, FM t 0.51 | The algo already knows what the residual knows |
| Signal by trade recency | strong at 1 to 7 days, zero past 7 days | Lives where the desk needs the least help |
| Residual increment in charge model over side x size | about 0.004 bp | No charge-overlay case |
| AR(1) OOS daily rank IC | +0.094, t 2.67 | Weak as a per-trade forecast |
| De-circularised beta-space peer level | 20.0 bp MAE vs 51.6 bp random | The one level-based result, treated as a side check |

Two structural reasons, not estimation problems:

1. **Change vs level.** Our residual is a one-day idiosyncratic yield change. An RFQ asks whether the bond is rich or cheap now, which is a level question. Levels revert over weeks; one-day changes are mostly noise plus mark mechanics.
2. **Wrong evaluation objective.** We scored the residual as a per-trade forecast (rank IC, charge MAE). A residual with daily IC near 0.1 across thousands of bonds is a portfolio signal, and we never measured it as one.

## 2. Two reference approaches

### 2.1 BlackRock: supervised similarity in yield levels

A CatBoost model predicts a bond's yield or spread from characteristics. Similarity between two bonds is read out of the trained trees: they are peers when the model routes them to the same leaves. Each bond gets a cohort of nearest supervised neighbours; the similarity-weighted cohort yield is fair value; the deviation is the relative-value signal. Back-tests against rule-based cohorts (state, rating, maturity, coupon buckets) and heuristic distance show the supervised cohort wins. The lineage is the group's random-forest-proximity work on corporate bonds.

### 2.2 Pelger group: factor residuals plus a sequence model, trained for Sharpe

Three blocks trained together. (1) Factors from characteristics: an attention map, nonlinear and learned, with IPCA and PCA as comparison factor blocks. (2) The trailing residual path, roughly one to two months of daily residuals, read by a LongConv long-convolution sequence model that outputs a position weight per asset. (3) One objective: out-of-sample Sharpe of the factor-neutral portfolio net of a cost model, so factors are chosen for the tradability of their residuals, not for variance explained. The 2022 precursor compared a parametric Ornstein-Uhlenbeck s-score, a Fourier-feature network and a CNN plus Transformer on separately estimated residuals; even the parametric OU baseline carried a meaningful Sharpe.

### 2.3 Side by side

| | BlackRock level RV | Attention-factor stat-arb | Our v1.1 |
|---|---|---|---|
| Object | Yield level vs peers | Residual path, 30 to 60 days | One-day residual increment |
| Factor block | None (tree model absorbs it) | Attention map or IPCA | IPCA, linear, K = 3 |
| Signal model | Cohort-weighted fair value | LongConv / OU s-score | Pooled AR(1), one slope |
| Labels | Transaction prints | Returns | Evaluated marks |
| Objective | Cohort tightness, forward reversion | Portfolio Sharpe, end to end | Rank IC, per-trade MAE |
| Horizon | Weeks | Days to weeks | Next day |
| Works for untraded bonds | Yes, the cohort has prints | n/a (equities) | Weakly |
| Binding constraint | Label availability | Transaction costs | Whether a quote gets hit |

## 3. What transfers to a market maker

The stat-arb mechanics do not. Two percent of munis trade daily, there is no continuous rebalancing, and a cost model is the wrong constraint. But a market maker is a stat-arb desk whose positions are chosen by customers and whose only control is the skew. Each customer sell is an offer to open a long; each customer buy is an offer to close one. The signal's job is to decide which offers to lean into and which inventory to work out of.

Under that reading:

- the Sharpe objective becomes inventory P&L from skewing on the signal;
- the cost constraint becomes fill probability;
- the cohort fair value becomes the quote prior for bonds with no recent print, which is the segment with the widest spreads and lowest fill rates.

The cumulative residual over a window is a level deviation from factor-implied fair value. That is the bridge between the two papers inside our own framework: the thing to trade is the cumulative residual normalised by its volatility, not yesterday's increment.

## 4. Recommended direction

### 4.1 Make the level problem the main problem

- **Target.** Trade yield or spread to the MMD curve at trade time (the matched cache carries `MmdYld` and `dMmdSprdSide`), side-adjusted so dealer, customer-buy and customer-sell prints sit on one mid.
- **Features.** The current 13 instruments, the IPCA betas, and the level drivers the product table should carry: sector or use of proceeds, issuer, tax status, bank-qualified flag, insurance, issue size.
- **Model.** LightGBM or CatBoost, walk-forward on the prior 60 days of prints. Model-implied yield minus observed is the relative-value signal. This replaces the AR layer.
- **Cohorts.** Leaf co-occurrence from the trained trees defines supervised peers. Compare head to head with beta-space peers, rule-based buckets and Euclidean nearest neighbours on within-cohort dispersion of the next print and on forward reversion at 5, 10 and 20 days.

### 4.2 Time-align cohort prints with the IPCA factor path

Peer prints are days old. Roll each one forward by its own beta times the cumulative factor move before averaging, so the cohort yield is as of today. The in-panel roll-forward test cut multi-day mark error by 34% to 63%. This is a genuine addition over the BlackRock design and gives IPCA a job the one-day residual could not do.

### 4.3 Read the residual as a path and score it as a portfolio

- **Paper portfolio.** Weights proportional to the signal, hedged to zero beta through the mimicking weights $W^F$, marked by next-day residuals. Report Sharpe by activity bucket and trade recency. Half a day on existing artifacts. If flat, stop on the residual path.
- **LongConv-lite.** Pooled ridge regression of the next residual on the last 10 to 20 residuals. A long convolution without the nonlinearity, sized for 57 OOS days. Then the parametric OU s-score on the cumulative residual.
- **Re-score the factor block.** Rerun K = 1 to 4 and the pruned instrument sets, choosing by paper-portfolio Sharpe, not R².

### 4.4 Evaluate where the algo is weakest

Split every test by trade recency and by covered vs uncovered universe. The claim worth making: for names with no print in seven days, model-implied yield beats the algo quote on the next print by X bp. Report with Fama-MacBeth inference by date.

### 4.5 Keep the quantile layer, change its job

The Step 5C machinery produces q10, q50, q90. Point it at the level model so each RFQ gets a fair-value mid and a width, with the width feeding spread rather than a charge.

### 4.6 Use cohorts for inventory

Similar bonds are substitutes: hedging and axe suggestions, and a mis-mark screen on inventory.

## 5. What stays from IPCA

Three roles: hedging systematic daily moves, supplying betas as features for the level model, and time-aligning stale prints. The one-day residual and its AR forecast step back to a diagnostic. The attention block is the lowest priority: with 13 characteristics and a dominant rate factor, nonlinearity in the loading map is unlikely to be where the gain is. Deep sequence models wait for several years of panel; six months cannot train a LongConv or Transformer honestly.

## 6. Experiments in order of cost

| # | Experiment | Data | Effort | Decides |
|---|---|---|---|---|
| 1 | Paper portfolio of hedged residual positions, Sharpe by bucket | Existing residuals and $W^F$ | 0.5 day | Whether there is anything to harvest on the residual path |
| 2 | LongConv-lite ridge on 10 to 20 lags; OU s-score on cumulative residual | Existing residuals | 1 day | Whether path beats one lag |
| 3 | Walk-forward LightGBM on side-adjusted trade spreads to MMD; compare abs error of algo quote, prior mark, model-implied yield by recency bucket | Matched cache, Step 3 panel | 1 to 2 weeks | Whether the level pivot is justified |
| 4 | Supervised cohorts vs beta-space, rule-based, Euclidean peers | Output of 3 | 1 week | Which peer definition to productionise |
| 5 | Time-aligned cohorts via IPCA roll-forward | Output of 4 plus factor path | 2 to 3 days | Incremental value of IPCA in the level framework |
| 6 | Skew back-test on realised customer flow, 5/10/20-day marks | Matched trades | 1 week | The market-making analogue of Sharpe |
| 7 | Re-score K and instrument sets by downstream objective | Existing pipeline | 2 days | Production factor spec |

Gate: experiments 1 and 3 run first. Experiment 1 decides whether the residual-path track continues. Experiment 3 decides whether the level track becomes the main line. Everything else follows from those two.

## References

- Saha, P., Lyu, J., Desai, D., Chauhan, R., Jeyapaulraj, J., Chu, P., Sommer, P., Mehta, D. (2024). Machine Learning-based Relative Valuation of Municipal Bonds. arXiv:2408.02273.
- Jeyapaulraj, J. et al. (2022). Supervised similarity learning for corporate bonds using Random Forest proximities.
- Epstein, E., Yu, R., Pelger, M. (2025). Attention Factors for Statistical Arbitrage.
- Guijarro-Ordonez, J., Pelger, M., Zanotti, G. (2022). Deep Learning Statistical Arbitrage.
- Kelly, B., Pruitt, S., Su, Y. (2019). Characteristics Are Covariances. JFE.
- Kelly, B., Palhares, D., Pruitt, S. (2023). Modeling Corporate Bond Returns. JF.
- Grinold, R. (1989). The Fundamental Law of Active Management. JPM.
