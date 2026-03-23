# Parallel Linesearch Parameter Sweep

## Setup

- Benchmark: `g1_fall`, Newton solver, 4096 envs, GPU, `--profile-wait 275`
- Baseline: `main` branch (sequential linesearch), FPS=218,592, GPU kernel=10,832 us
- Parameters swept: `LS_PARALLEL_K` ∈ {16, 32}, `LS_PARALLEL_N_REFINE` ∈ {1..9}
- Total: 18 experiments

## Results

### K=16

| N_REFINE | FPS | vs main | GPU (us) | vs main | LS (us) | non-LS (us) |
|----------|---------|---------|----------|---------|---------|-------------|
| 1 | 451,682 | +106.6% | 8,506 | −21.5% | 779 | 7,727 |
| 2 | 502,067 | +129.7% | 7,931 | −26.8% | 781 | 7,149 |
| **3** | **506,702** | **+131.8%** | **7,656** | **−29.3%** | **831** | **6,825** |
| 4 | 506,002 | +131.5% | 7,576 | −30.1% | 895 | 6,681 |
| 5 | 491,129 | +124.7% | 7,420 | −31.5% | 948 | 6,472 |
| 6 | 494,406 | +126.2% | 7,447 | −31.3% | 1,007 | 6,440 |
| 7 | 489,321 | +123.9% | 7,698 | −28.9% | 1,096 | 6,602 |
| 8 | 481,278 | +120.2% | 7,651 | −29.4% | 1,139 | 6,512 |
| 9 | 474,108 | +116.9% | 7,750 | −28.4% | 1,191 | 6,559 |

### K=32

| N_REFINE | FPS | vs main | GPU (us) | vs main | LS (us) | non-LS (us) |
|----------|---------|---------|----------|---------|---------|-------------|
| 1 | 473,790 | +116.7% | 8,297 | −23.4% | 763 | 7,534 |
| 2 | 511,207 | +133.9% | 7,538 | −30.4% | 769 | 6,768 |
| **3** | **511,750** | **+134.1%** | **7,345** | **−32.2%** | **828** | **6,517** |
| 4 | 509,037 | +132.9% | 7,510 | −30.7% | 898 | 6,612 |
| 5 | 502,031 | +129.7% | 7,808 | −27.9% | 960 | 6,848 |
| 6 | 496,594 | +127.2% | 7,456 | −31.2% | 1,009 | 6,448 |
| 7 | 489,837 | +124.1% | 7,669 | −29.2% | 1,089 | 6,579 |
| 8 | 483,938 | +121.4% | 7,741 | −28.5% | 1,156 | 6,586 |
| 9 | 477,384 | +118.4% | 7,806 | −27.9% | 1,223 | 6,583 |

## Analysis

### Optimal configuration: K=32, N_REFINE=3

- **FPS: 511,750 (+134.1% vs main)**
- **GPU kernel time: 7,345 us (−32.2% vs main)**
- Linesearch: 828 us (vs 4,422 us on main, −81.3%)

### Key observations

1. **N_REFINE=3 is the sweet spot for both K=16 and K=32.** Below 3, convergence is too
   slow (high non-LS time). Above 3, the extra eval kernel launches cost more than they save.

2. **K=32 is consistently better than K=16** at every N_REFINE value. The extra candidate
   density improves alpha precision. The cost of doubling K in the eval kernel is negligible
   (~7 us per call) because it's a shared-memory reduction within a single CUDA block.

3. **Diminishing returns beyond N_REFINE=3:** Each refinement pass adds ~30-50 us to the
   linesearch total but saves progressively less in non-LS kernels:
   - N=1→2: LS +6 us, non-LS −766 us (huge win from better convergence)
   - N=2→3: LS +59 us, non-LS −251 us (still worthwhile)
   - N=3→4: LS +70 us, non-LS +96 us (diminishing, crossover point)
   - N=4→5: LS +62 us, non-LS +236 us (net loss)

4. **FPS vs GPU kernel time diverge at high N_REFINE.** FPS drops faster than GPU kernel
   time increases, suggesting CPU-side overhead from the extra kernel launches matters.

5. **Linesearch cost scales linearly with N_REFINE:** ~763 us base + ~50 us per refinement
   pass (the eval kernel at ~7 us × 10 iterations, plus serial overhead).

### Trade-off curve

```
N_REFINE:  1      2      3      4      5      6      7      8      9
K=16 FPS:  452k   502k   507k   506k   491k   494k   489k   481k   474k
K=32 FPS:  474k   511k   512k   509k   502k   497k   490k   484k   477k
                          ^^^
                         PEAK
```

### Recommendation

Use **K=32, N_REFINE=3** for the best FPS. If compile time or code simplicity is a concern,
**K=16, N_REFINE=3** is nearly as good (507k vs 512k FPS, within noise) and uses half the
shared memory per block.
