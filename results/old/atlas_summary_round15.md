# ATLAS Run Summary — 15 Rounds

Date: 2026-01-29

## Overview

- Mode: full
- Completed rounds: 15
- Total runtime: 53.1 minutes

## Phase 1 — Task Clustering

- Num Clusters: 5
- Cluster Assignments: {'0': 0, '1': 0, '2': 0, '3': 2, '4': 4, '5': 1, '6': 2, '7': 3, '8': 1}
- Silhouette Score: 0.011

## Phase 2 — Device Profiles (sample)

- Example client device profiles & allocated ranks:
  - Client 0: {'memory_mb': 2048, 'compute_ratio': 1.0, 'suggested_ranks': [4, 8]} (ranks [8, 8, 8, 8, 8, 8])
  - Client 5: {'memory_mb': 8192, 'compute_ratio': 4.0, 'suggested_ranks': [16, 32]} (ranks [32, 32, 32, 32, 32, 32])

## Round 15 — Evaluation (per-client)

- Client 0 (sst2): acc=0.8658, f1=0.8657, loss=0.3411
- Client 1 (sst2): acc=0.8578, f1=0.8576, loss=0.3398
- Client 2 (sst2): acc=0.8578, f1=0.8576, loss=0.3505
- Client 3 (mrpc): acc=0.7426, f1=0.6723, loss=0.5306
- Client 4 (mrpc): acc=0.7426, f1=0.6623, loss=0.5350
- Client 5 (mrpc): acc=0.7500, f1=0.6730, loss=0.4977
- Client 6 (cola): acc=0.7517, f1=0.6579, loss=0.5519
- Client 7 (cola): acc=0.7603, f1=0.6666, loss=0.5879
- Client 8 (cola): acc=0.7450, f1=0.6279, loss=0.6762

Round 15 average accuracy: 0.7860

Checkpoint saved: `checkpoints/atlas_round_15.pkl`

## Final per-client accuracies (after Round 15)

```
{0: 0.8658256880733946, 1: 0.8577981651376146, 2: 0.8577981651376146,
 3: 0.7426470588235294, 4: 0.7426470588235294, 5: 0.75,
 6: 0.7516778523489933, 7: 0.7603068072866731, 8: 0.7449664429530202}
```

## Notes

- The saved checkpoint `atlas_round_15.pkl` contains model states, device configs, clustering metadata, and intermediate results. Use the `--resume` flag to continue from this file.
- If you want this summary added to the repository README instead, tell me and I will append a short entry linking to this file.
