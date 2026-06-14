# Multi-seed aggregated results (evaluation return, mean ± std across seeds)

Return = sum of system negative waiting time over the eval episode (higher / less-negative is better).

- **Final** = last eval epoch.
- **Late-window** = mean over the last 5 evals (converged plateau; the fair head-to-head metric — robust to late ε-greedy noise).
- **Peak** = best single eval per seed (optimistic upper bound).

| Scenario | Algorithm | Final | Late-window | Peak | n seeds |
| :--- | :--- | ---: | ---: | ---: | ---: |
| baseline | IQL | -119,144 ± 58,633 | -118,183 ± 27,494 | -45,498 ± 8,251 | 5 |
| baseline | Hysteretic | -150,344 ± 163,663 | -99,644 ± 35,382 | -37,277 ± 2,293 | 5 |
| baseline | VDN | -35,521 ± 11,517 | -40,063 ± 14,959 | -27,531 ± 3,797 | 5 |
| baseline | Fixed-Time | -114,411 ± 1,541 | -114,411 ± 1,541 | -114,411 ± 1,541 | 5 |
| baseline | QMIX | -25,956 ± 3,540 | -24,245 ± 893 | -21,252 ± 285 | 5 |
| dense_wave | IQL | -169,004 ± 29,450 | -249,185 ± 140,002 | -92,991 ± 8,058 | 5 |
| dense_wave | Hysteretic | -117,650 ± 50,216 | -181,539 ± 95,508 | -63,923 ± 10,688 | 5 |
| dense_wave | VDN | -83,614 ± 16,807 | -118,852 ± 41,819 | -57,649 ± 5,746 | 5 |
| dense_wave | Fixed-Time | -141,633 ± 3,216 | -141,633 ± 3,216 | -141,633 ± 3,216 | 5 |
| dense_wave | QMIX | -49,603 ± 9,398 | -50,347 ± 4,573 | -36,240 ± 1,731 | 5 |
| cross_surge | IQL | -305,483 ± 202,940 | -298,661 ± 144,000 | -145,227 ± 12,312 | 5 |
| cross_surge | Hysteretic | -174,940 ± 43,596 | -422,805 ± 133,130 | -128,382 ± 7,705 | 5 |
| cross_surge | VDN | -260,125 ± 119,732 | -205,426 ± 31,761 | -129,448 ± 6,814 | 5 |
| cross_surge | Fixed-Time | -255,587 ± 1,715 | -255,587 ± 1,715 | -255,587 ± 1,715 | 5 |
| cross_surge | QMIX | -91,369 ± 4,246 | -87,819 ± 2,622 | -79,746 ± 1,060 | 5 |
| split_rush | IQL | -318,516 ± 316,965 | -257,179 ± 112,120 | -72,032 ± 12,225 | 5 |
| split_rush | Hysteretic | -169,830 ± 87,539 | -211,530 ± 171,319 | -63,645 ± 6,979 | 5 |
| split_rush | VDN | -109,227 ± 21,811 | -104,294 ± 18,076 | -58,214 ± 5,009 | 5 |
| split_rush | Fixed-Time | -141,341 ± 1,619 | -141,341 ± 1,619 | -141,341 ± 1,619 | 5 |
| split_rush | QMIX | -40,930 ± 8,581 | -44,855 ± 4,433 | -34,349 ± 2,599 | 5 |
| cologne3 | IQL | -10,589 ± 3,693 | -10,729 ± 4,086 | -6,473 ± 825 | 5 |
| cologne3 | Hysteretic | -29,617 ± 44,541 | -96,554 ± 97,862 | -8,256 ± 4,322 | 5 |
| cologne3 | VDN | -5,510 ± 626 | -5,368 ± 371 | -3,882 ± 249 | 5 |
| cologne3 | Fixed (native) | -44,047 ± 2,073 | -44,047 ± 2,073 | -44,047 ± 2,073 | 5 |
| cologne3 | Fixed (round-robin) | -175,498 ± 1,415 | -175,498 ± 1,415 | -175,498 ± 1,415 | 5 |
| cologne3 | QMIX | -11,779 ± 15,624 | -14,429 ± 13,241 | -2,947 ± 266 | 5 |
| grid4x4 | IQL | -416,788 ± 53,524 | -496,058 ± 35,573 | -272,110 ± 29,998 | 5 |
| grid4x4 | Hysteretic | -484,504 ± 90,652 | -462,960 ± 38,059 | -239,863 ± 23,343 | 5 |
| grid4x4 | VDN | -190,620 ± 43,112 | -155,544 ± 23,682 | -93,341 ± 21,451 | 5 |
| grid4x4 | Fixed (native) | -31,498 ± 587 | -31,498 ± 587 | -31,498 ± 587 | 5 |
| grid4x4 | Fixed (round-robin) | -47,571 ± 1,031 | -47,571 ± 1,031 | -47,571 ± 1,031 | 5 |
| grid4x4 | QMIX | -517,038 ± 129,346 | -451,590 ± 87,427 | -307,420 ± 16,159 | 5 |
