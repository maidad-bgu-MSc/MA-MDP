# Multi-seed aggregated results (final-epoch evaluation return)

Return = sum of system negative waiting time over the eval episode (higher / less-negative is better). Mean ± std across seeds.

| Scenario | Algorithm | Mean return | Std | n seeds |
| :--- | :--- | ---: | ---: | ---: |
| baseline | IQL | -84,689 | 22,303 | 3 |
| baseline | Hysteretic | -98,130 | 39,422 | 3 |
| baseline | VDN | -77,937 | 27,057 | 3 |
| baseline | Fixed-Time | -114,762 | 1,250 | 3 |
| baseline | QMIX | -24,850 | 1,449 | 3 |
| dense_wave | IQL | -342,021 | 265,034 | 3 |
| dense_wave | Hysteretic | -246,395 | 188,982 | 3 |
| dense_wave | VDN | -116,795 | 15,595 | 3 |
| dense_wave | Fixed-Time | -141,633 | 3,872 | 3 |
| dense_wave | QMIX | -39,109 | 2,872 | 3 |
| cross_surge | IQL | -546,157 | 467,080 | 3 |
| cross_surge | Hysteretic | -652,901 | 671,712 | 3 |
| cross_surge | VDN | -173,900 | 19,835 | 3 |
| cross_surge | Fixed-Time | -255,543 | 1,028 | 3 |
| cross_surge | QMIX | -89,355 | 3,213 | 3 |
| split_rush | IQL | -700,621 | 805,103 | 3 |
| split_rush | Hysteretic | -118,988 | 18,232 | 3 |
| split_rush | VDN | -177,383 | 72,619 | 3 |
| split_rush | Fixed-Time | -141,535 | 2,067 | 3 |
| split_rush | QMIX | -57,766 | 14,082 | 3 |
| cologne3 | IQL | -233,713 | 111,218 | 3 |
| cologne3 | Hysteretic | -407,176 | 266,496 | 3 |
| cologne3 | VDN | -113,308 | 34,405 | 3 |
| cologne3 | Fixed (native) | -43,338 | 2,119 | 3 |
| cologne3 | Fixed (round-robin) | -175,174 | 1,605 | 3 |
| grid4x4 | IQL | -1,010,640 | 5,418 | 2 |
| grid4x4 | Hysteretic | -839,815 | 102,514 | 2 |
| grid4x4 | VDN | -764,499 | 119,440 | 2 |
| grid4x4 | Fixed (native) | -31,944 | 582 | 2 |
| grid4x4 | Fixed (round-robin) | -47,530 | 786 | 2 |
