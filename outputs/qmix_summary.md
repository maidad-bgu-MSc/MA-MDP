# Training Results Summary

Two training regimes were run across all 4 scenarios (`baseline`, `dense_wave`, `cross_surge`, `split_rush`):

- **Tabular (Decentralized)**: Tabular IQL, Hysteretic Q, VDN — 100 epochs, logged every 5 epochs, vs. a Fixed-Time baseline.
- **CTDE (Centralized Training, Decentralized Execution)**: QMIX — 200 episodes, eval every 20 episodes.

Delay = `|eval_reward|` throughout (lower is better).

---

# Part 1 — QMIX (CTDE) Training Results

200 episodes per scenario, eval every 20 episodes (deterministic rollout, 120 steps).
Total delay = `|eval_reward|` (lower is better).

## Final-episode eval delay (ep 200)

| Scenario     | Ep 200 delay | Best delay (episode) | Worst delay (episode) | Trend |
| :----------- | -----------: | -------------------: | --------------------: | :--- |
| baseline     |   **25,566** |       22,251  (180)  |       31,989   (20)  | Gradual improvement, narrow band ~22–32k |
| dense_wave   |   **33,614** |       33,614  (200)  |    1,618,716   (40)  | Strong learning; ~48× drop from peak |
| cross_surge  |   **81,187** |       81,187  (200)  |      971,986   (20)  | Fast initial collapse, then plateau ~85–95k |
| split_rush   |   **50,852** |       38,527  (120)  |    1,117,680   (40)  | Volatile early; settles ~45–50k |

## Per-scenario learning curves (eval delay)

```
baseline     :  31989 →  25283 →  28428 →  25928 →  25278 →  29675 →  24023 →  24047 →  22251 →  25566
dense_wave   : 667012 → 1618716 → 1185428 → 1409566 →  884039 →  330056 →   45075 →   42770 →   48025 →   33614
cross_surge  : 971986 →   97573 →   93638 →   89714 →   86835 →   91091 →   86109 →   92091 →   83984 →   81187
split_rush   :  82362 → 1117680 →  382797 →  134148 →  111776 →   38527 →   50186 →   45802 →   45512 →   50852
```

## Observations

- **baseline** is already low-delay because traffic is light; QMIX gains are small (~25% reduction).
- **dense_wave** shows the most dramatic improvement (~95% reduction by ep 200) — heavy uniform load gives the mixer a clear coordination signal.
- **cross_surge** plateaus early. After ep 40 it barely moves; suggests the QMIX mixer has converged to a local optimum or the scenario is harder to coordinate.
- **split_rush** is the noisiest (large episode-to-episode swings even late), but final ~50k is roughly an order of magnitude better than the early peak.

## Wall-clock

| Scenario     | Duration |
| :----------- | -------: |
| baseline     | 1h 23m   |
| dense_wave   | 1h 36m   |
| cross_surge  | 2h 07m   |
| split_rush   | ~1h 25m  |
| **Total**    | **~6h 30m** |

---

# Part 2 — Tabular Training Results

100 epochs per scenario, eval every 5 epochs. Three decentralized algorithms benchmarked against a fixed-time controller. Logs: `training_evaluation_log_<scenario>.csv`.

## Final-epoch delay (epoch 100) and best delay across run

### baseline (Fixed-Time = 112,204)

| Algorithm    | Epoch-100 delay | Best delay (epoch) | Beats Fixed at ep 100? |
| :----------- | --------------: | -----------------: | :---: |
| Tabular IQL  | 194,547         |  46,082 (10)       | ✗ |
| Hysteretic Q | 216,893         |  39,718 (90)       | ✗ |
| VDN          | 118,550         |  57,536 (10)       | ✗ (close, +5.6%) |

### dense_wave (Fixed-Time = 140,421)

| Algorithm    | Epoch-100 delay | Best delay (epoch) | Beats Fixed at ep 100? |
| :----------- | --------------: | -----------------: | :---: |
| Tabular IQL  | 164,344         |  71,735 (15)       | ✗ |
| Hysteretic Q | 163,873         |  78,781 (25)       | ✗ |
| VDN          | 118,243         |  79,776 (15)       | ✓ (−16%) |

### cross_surge (Fixed-Time = 254,882)

| Algorithm    | Epoch-100 delay | Best delay (epoch) | Beats Fixed at ep 100? |
| :----------- | --------------: | -----------------: | :---: |
| Tabular IQL  | 281,437         | 134,045 (50)       | ✗ |
| Hysteretic Q | 675,706         | 170,455 (50)       | ✗ (much worse) |
| VDN          | 206,500         | 159,990 (60)       | ✓ (−19%) |

### split_rush (Fixed-Time = 141,719)

| Algorithm    | Epoch-100 delay | Best delay (epoch) | Beats Fixed at ep 100? |
| :----------- | --------------: | -----------------: | :---: |
| Tabular IQL  | 163,102         |  74,694 (15)       | ✗ |
| Hysteretic Q |  90,007         |  71,896 (25)       | ✓ (−36%) |
| VDN          | 399,446         |  79,993 (80)       | ✗ (much worse) |

## Observations — tabular

- All three tabular algorithms are **highly unstable across epochs** — multi-hundred-thousand swings (e.g. Hysteretic on `split_rush` ep 80 = 2.36M). Reading epoch-100 alone is misleading; the *best* column is a better summary of capability.
- **VDN is the most consistent**: it's the only tabular method to beat Fixed-Time at epoch 100 on two scenarios (`dense_wave`, `cross_surge`).
- **Hysteretic Q** wins decisively on `split_rush` but collapses badly on `cross_surge` — high variance across scenarios.
- **Tabular IQL** never finishes ahead of Fixed-Time at epoch 100, though its *best* checkpoints are competitive — selecting via early-stopping on eval would change the picture.

---

# Cross-regime comparison (final eval, lower is better)

| Scenario     | Fixed-Time | Tab IQL (ep100) | Hysteretic (ep100) | VDN (ep100) | **QMIX (ep200)** |
| :----------- | ---------: | --------------: | -----------------: | ----------: | ---------------: |
| baseline     |    112,204 |         194,547 |            216,893 |     118,550 |       **25,566** |
| dense_wave   |    140,421 |         164,344 |            163,873 |     118,243 |       **33,614** |
| cross_surge  |    254,882 |         281,437 |            675,706 |     206,500 |       **81,187** |
| split_rush   |    141,719 |         163,102 |             90,007 |     399,446 |       **50,852** |

**QMIX (CTDE) wins on every scenario** — by a wide margin (2–6× better than the best tabular method). Caveat: QMIX trained for 200 episodes vs. tabular's 100 epochs, so this isn't a strictly equal-budget comparison, but the gap is large enough that the qualitative conclusion holds: centralized mixing + neural function approximation handles this multi-agent coordination problem substantially better than independent tabular learners.
