# Micro-Benchmark Findings
## [TMA Multicast](https://newsletter.semianalysis.com/i/191922559/tma-multicast)
> According to NCU, the unit responsible for serving TMA multicast requests is called the L2 Request Coalescer (LRC):

> The L2 Request Coalescer (LRC) processes incoming requests for L2 and tries to coalesce read requests before forwarding them to the L2 cache. It also serves programmatic multicast requests from the SM and supports compression for writes.

Multicast can be enabled either explicitly in the instruction or implicitly by hardware behavior. Although implicit multicast may yield less efficient L2 request coalescing, it can perform similarly to explicit multicast when L2 bandwidth is not the bottleneck.

## [TMA Store](./experiments/tma.md)

- Store Cycles = 88 + Bytes / 32. 

TMA store runs at the end of each output tile loop, and when one SM processes multiple tiles, the first store can be hidden by the next tile's compute.

## [UMMA Throughput](https://newsletter.semianalysis.com/i/191922559/throughput)

> For 1SM MMA across all N sizes, we see that the smaller M=64 achieves max 50% theoretical peak throughput, and the larger M=128 achieves near 100%

> MMA supports two different AB layouts: Both input matrices stored in SMEM (SS), and matrix A stored in TMEM and matrix B stored in SMEM (TS). We observed that for M=128, while ABLayout=TS achieves near peak throughput, ABLayout=SS underperforms in smaller N sizes and catches up at N=128.
