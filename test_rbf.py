import numpy as np

fingerprints = {
    6: np.array([1.7827365398406982, 0.04275493696331978, 0.05361815169453621, 0.0694398283958435, 0.10022515058517456, 0.0781698003411293, 0.04695452004671097, 0.10560696572065353]),
    7: np.array([1.4322859048843384, 0.03839389234781265, 0.03795506805181503, 0.05675967037677765, 0.07703671604394913, 0.06135978922247887, 0.04289127141237259, 0.08610808849334717]),
    8: np.array([1.6681411266326904, 0.06650900095701218, 0.07575833797454834, 0.10781732201576233, 0.16522818803787231, 0.12756311893463135, 0.07019656896591187, 0.15376132726669312])
}

client_ids_sorted = sorted(fingerprints.keys())
fingerprint_array = np.vstack([fingerprints[cid] for cid in client_ids_sorted])
from scipy.spatial.distance import cdist
pairwise_distances_sq = cdist(fingerprint_array, fingerprint_array, metric='sqeuclidean')

print("Pairwise distances sq:")
print(pairwise_distances_sq)

clip_val = float(np.percentile(pairwise_distances_sq[pairwise_distances_sq > 0], 95.0))
pairwise_distances_sq_robust = np.log1p(np.clip(pairwise_distances_sq, 0.0, clip_val))

print("Robust distances:")
print(pairwise_distances_sq_robust)

mira_alpha = 1.0
rbf_floor = 0.05

weights = {}
for i, cid_i in enumerate(client_ids_sorted):
    neighbor_weights = []
    for j, cid_j in enumerate(client_ids_sorted):
        if i == j: continue
        d = pairwise_distances_sq_robust[i, j]
        weight = max(float(np.exp(-mira_alpha * d)), rbf_floor)
        neighbor_weights.append((cid_j, weight))
    
    total_weight = sum(w for _, w in neighbor_weights)
    neighbor_weights = [(j, w / total_weight) for j, w in neighbor_weights]
    for cid_j, w in neighbor_weights:
        weights[(cid_i, cid_j)] = w

print("Weights:")
for k, v in weights.items():
    print(f"{k}: {v:.4f}")

