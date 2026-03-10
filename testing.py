import json
from pathlib import Path

files = {
    "atlas_dbert": "results/atlas_distilbert-base-uncased_atlas_seed42_r10.json",
    "fedavg_dbert": "results/atlas_distilbert-base-uncased_fedavg_cluster_seed42_r10.json",
    "atlas_gpt2": "results/atlas_gpt2_atlas_seed42_r10.json",
    "fedavg_gpt2": "results/atlas_gpt2_fedavg_cluster_seed42_r10.json",
}

BUDGET = {
    "max_trainable": 300_000,
    "max_adapter_mb": 0.5,
}

for name, fp in files.items():
    d = json.load(open(fp))
    weak = [k for k,v in d["device_configs"].items() if v["device_profile"]["memory_mb"] <= 2048]

    rows = []
    for k in weak:
        trainable = d["trainable_params"][k]["trainable"]
        adapter_mb = d["device_configs"][k]["adapter_memory_mb"]
        canonical = d["final_canonical"][k]
        feasible = (trainable <= BUDGET["max_trainable"]) and (adapter_mb <= BUDGET["max_adapter_mb"])
        rows.append((k, trainable, adapter_mb, canonical, feasible))

    viol_rate = 1 - sum(r[4] for r in rows)/len(rows)
    weak_mean = sum(r[3] for r in rows)/len(rows)

    print(f"\n{name}")
    print("weak rows:", rows)
    print("weak_mean_canonical:", round(weak_mean, 4))
    print("weak_budget_violation_rate:", round(viol_rate, 4))