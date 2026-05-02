# Tensor Memory Accelerator (TMA)
## Swizzle Mode
Goal: transferring 8x256B to shared memory for MMA usage.

refer to the smem descriptor format -> different swizzle mode

- baseline: swizzle none with box (8, 256B) - 2D map + 1 instr
- swizzle none with box (16, 8, 16B) - 2D map + 16 instr or 3D map + 1 instr
- swizzle 32B with box (8, 8, 32B) - 2D map + 8 instr or 3D map + 1 instr
- swizzle 64B with box (4, 8, 64B) - 2D map + 4 instr or 3D map + 1 instr
- swizzle 128B with box (2, 8, 128B) - 2D map + 2 instr or 3D map + 1 instr

## load/store latencies

```sh
python tests/tma.py --n "$n"
```

### Load (L2 Hit)
Conclusion: if Compute >= 512 cycles, L2 hit scenario can be hidden by compute

| N | Bytes | L2 Load Min | L2 Load Avg | L2 Load Max |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 128 | 350 | 350.0 | 350 |
| 2 | 256 | 350 | 350.0 | 350 |
| 4 | 512 | 350 | 350.0 | 350 |
| 8 | 1024 | 350 | 350.0 | 350 |
| 16 | 2048 | 350 | 351.8 | 354 |
| 24 | 3072 | 350 | 350.0 | 350 |
| 32 | 4096 | 350 | 350.0 | 350 |
| 64 | 8192 | 350 | 395.0 | 500 |
| 96 | 12288 | 500 | 500.0 | 500 |
| 128 | 16384 | 500 | 500.0 | 500 |
| 160 | 20480 | 500 | 500.0 | 500 |
| 192 | 24576 | 500 | 500.0 | 500 |
| 256 | 32768 | 650 | 650.0 | 650 |

### Store
Conclusion: Store Cycles = 88 + Bytes / 32

| N | Bytes | Store Cycles | Bytes / 32 | Fixed Overhead |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 128 | 92 | 4 | 88 |
| 2 | 256 | 96 | 8 | 88 |
| 4 | 512 | 104 | 16 | 88 |
| 8 | 1024 | 120 | 32 | 88 |
| 16 | 2048 | 152 | 64 | 88 |
| 24 | 3072 | 184 | 96 | 88 |
| 32 | 4096 | 216 | 128 | 88 |
| 64 | 8192 | 344 | 256 | 88 |
| 96 | 12288 | 472 | 384 | 88 |
| 128 | 16384 | 600 | 512 | 88 |
| 160 | 20480 | 728 | 640 | 88 |
| 192 | 24576 | 856 | 768 | 88 |
| 256 | 32768 | 1112 | 1024 | 88 |
