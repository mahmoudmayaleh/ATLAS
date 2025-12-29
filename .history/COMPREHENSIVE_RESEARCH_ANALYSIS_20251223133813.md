# COMPREHENSIVE RESEARCH ANALYSIS: ATLAS PROJECT PAPERS

## Advanced Task-aware Layered Aggregation for Split Learning

**Date:** December 23, 2025  
**Location:** c:\Users\Hp\Downloads\Advanced_project\ATLAS

---

## EXECUTIVE SUMMARY

This analysis synthesizes 7 research papers related to the ATLAS project, which aims to implement a **self-organizing decentralized multi-task learning framework for LLM fine-tuning on heterogeneous edge devices**. The framework combines:

- **MIRA's multi-task learning** (task clustering for knowledge sharing)
- **HSplitLoRA's heterogeneous adaptation** (device-aware LoRA ranks)
- **SplitLoRA's split learning** (efficient model partitioning)
- **VFLAIR's privacy evaluation** (comprehensive attack/defense benchmarking)

---

## 1. ATLAS_Base_Specification.pdf

### Overview

**Status:** Project specifications and objectives (1 page)  
**Key Project ID:** US PROJ 2025/2026 Project #13  
**Institution:** Conservatoire national des arts et métiers  
**Supervisors:** Lyes Khoukhi & Zakria Abouelhouda  
**Project Lead:** Mahmoud Mayaleh

### Core Objectives

**Problem Statement:**

- LLMs have billions of parameters, exceeding memory of individual edge devices (phones, IoT, edge servers)
- Clients operate on **heterogeneous task-specific data** (medical records, financial transactions, conversational logs)
- Existing solutions address only ONE challenge:
  - **MIRA**: Task-aware + centralized (cannot split models)
  - **TITANIC**: Model splitting peer-to-peer (task-blind)

**Primary Objective:**
Design and implement **ATLAS**: A self-organizing decentralized multi-task learning framework combining:

1. **Gradient-based semantic client clustering** (discover similar tasks)
2. **Peer-to-peer split learning** (partition model across devices)
3. **Cluster-wise adapter aggregation** (efficient knowledge sharing)

### Three-Phase Implementation Plan

**Phase 1: Comprehensive Analysis**

- Study MIRA's centralized task graphs and bottlenecks
- Examine TITANIC's peer-to-peer architecture and task heterogeneity limitations
- Review LoRA adapter methods for decentralized aggregation
- **Outcome:** Architectural specification with task embeddings, clustering procedures, resource-aware pairing, hierarchical aggregation

**Phase 2: Simulation Environment**

- Clients compute gradient-based task embeddings from local data
- Send compact signatures to server for semantic clustering
- Server initializes shared LoRA adapters
- Robustness: Intra-cluster re-pairing and inter-cluster fallback for stragglers
- **Key Innovation:** Direct intermediate activation exchange (no centralization)

**Phase 3: Systematic Evaluation**

- Metrics: Task-specific accuracy, communication volume, convergence speed, computational overhead, clustering quality
- Baselines: Centralized training, federated multi-task, task-blind split learning

---

## 2. ATLAS_V1.pdf

### Overview

**Status:** Detailed project plan and presentation (22 pages)  
**Last Updated:** December 23, 2025

### Challenges Identified

**Challenge 1: Model Size**

- Modern LLMs: 7B to 70B parameters
- Device constraints: 1GB (weak), 8GB (medium), 24GB (strong)
- Problem: Inference and training infeasible on weak devices

**Challenge 2: Device Heterogeneity**

- Different GPU budgets (1GB to 24GB)
- Different task types (NLP, Q&A, summarization)
- Different network speeds (5G, 4G, WiFi)

**Challenge 3: Efficiency**

- One-size-fits-all approaches waste resources on weak devices
- Underutilize strong devices

### Comparison with Existing Work

| Method         | Heterogeneous | Task-Aware | Efficient |
| -------------- | ------------- | ---------- | --------- |
| **SplitLoRA**  | ❌            | ❌         | ✓         |
| **HSplitLoRA** | ✓             | ❌         | ✓         |
| **MIRA**       | ✓             | ✓          | ❌        |
| **ATLAS**      | ✓             | ✓          | ✓         |

### ATLAS System Overview (4 Phases)

#### Phase 1: Task Discovery and Clustering

**Goal:** Group clients with similar tasks for knowledge sharing

**Process:**

1. Each client computes gradients on small batch of local data
2. Compress gradients into 64-dimensional "task fingerprint"
3. Server collects all fingerprints
4. k-Means clustering identifies semantic groups

**Privacy:** Only low-dimensional vectors sent; no raw data or full gradients

#### Phase 2: Heterogeneous Configuration Assignment

**Goal:** Choose LoRA rank and split point for each client

**Process:**

1. Determine weight importance per task cluster
2. Assign higher ranks to important weights
3. Respect each client's memory budget
4. Similar rank patterns within clusters, different absolute values

**Example Configuration:**

```
Client A (24GB GPU):  LoRA ranks [8, 6, 4]
Client B (8GB GPU):   LoRA ranks [6, 4, 2]
Client C (1GB GPU):   LoRA ranks [4, 2, 1]
```

#### Phase 3: Split Learning with LoRA

**Architecture:**

- **Client side:** Input processing + early layers + LoRA adapters
- **Server side:** Task layers + loss head + LoRA adapters

**Per Round:**

1. Client sends intermediate activations h to server
2. Server computes loss and gradients ∇h
3. Server sends gradients back to client
4. Both update LoRA adapters using local SGD

**Key:** No full model or gradients transmitted; minimal communication

#### Phase 4: Cluster-Aware Aggregation

**Innovation:** Noise-Free Heterogeneous Aggregation

**Process:**

1. Stack all client adapters (different ranks allowed)
2. Concatenate low-rank factors ⇒ merged rank
3. Apply weighted average in merged space
4. Broadcast updated adapter
5. Clients decompose for next round

**Example:**

```
Client 1 (rank 4) + Client 2 (rank 6) + Client 3 (rank 4)
→ Concatenate → rank 14
→ Merge using aggregation formula
→ No information loss (vs. standard LoRA averaging)
```

### Expected Performance Gains

**Efficiency:** 30-40% memory saved

- Weak devices get small LoRA ranks ⇒ fit budget
- Strong devices use larger ranks ⇒ utilize resources

**Accuracy:** 20-30% improvement

- Task clustering enables knowledge transfer
- Clients specialize within clusters
- Shared cluster-level model provides regularization

**Privacy:** Differential privacy compatible

- Only 64-D fingerprints sent (not raw data)
- No full gradients transmitted
- Evaluated using VFLAIR-LLM framework

### Implementation Timeline (12 Weeks)

| Weeks | Tasks                        | Key Libraries               |
| ----- | ---------------------------- | --------------------------- |
| 1-2   | PyTorch + Transformers setup | torch, transformers         |
| 3-4   | Add LoRA, test on GPT-2      | PEFT                        |
| 5-6   | Phase 1 (task clustering)    | scikit-learn (k-means, PCA) |
| 7-8   | Phase 2 (weight importance)  | PEFT                        |
| 9-10  | Phase 3 & 4 (training loop)  | torch                       |
| 11-12 | Evaluation + privacy         | VFLAIR-LLM                  |

**Tech Stack (All Open-Source):**

- PyTorch (BSD)
- HuggingFace Transformers (Apache 2.0)
- PEFT LoRA (Apache 2.0)
- scikit-learn (BSD)
- VFLAIR-LLM (MIT)

### Evaluation Metrics

**Accuracy Metrics:**

- Task-specific metrics
- Convergence speed
- Final model accuracy

**Efficiency Metrics:**

- GPU memory usage
- Communication cost
- Training time

**Privacy Metrics:**

- Attack success rate (VFLAIR)
- Privacy-utility trade-off
- Defense Capability Score (DCS) ≥ 0.7

### Expected Results

- **Accuracy:** +15-25% vs HSplitLoRA
- **Memory:** 30-40% reduction
- **Privacy:** DCS ≥ 0.7

---

## 3. MIRA: A Method of Federated Multi-Task Learning for LLMs

### Paper Details

**Pages:** 5 (IEEE Networking Letters, Vol. 7, No. 3, September 2025)  
**Authors:** Ahmed Elbakary, Chaouki Ben Issaid, Tamer ElBatt, Karim Seddik, Mehdi Bennis  
**Journal:** IEEE Networking Letters

### Main Contributions

1. **Federated Multi-Task Learning (FMTL) Framework** for LLMs
2. **Task-aware Knowledge Sharing** using graph-based regularization
3. **Parameter-Efficient Fine-Tuning** with LoRA to reduce computation
4. **Experimental validation** on multiple datasets and models

### Core Problem

**Multi-Task Learning Context:**

- Traditional FL learns single global model (suboptimal for heterogeneous data)
- Different clients have different tasks: medical diagnosis, financial analysis, translation, etc.
- A single global model may fail to capture task-specific nuances

**Motivation Example:**
Medical entities training shared chatbot system:

- Each region/institution has unique medical terminologies
- Unique treatment protocols and diagnostic criteria
- Single global model reduces effectiveness
- Need: FMTL to capture task-specific knowledge

### Technical Approach

#### System Model

**Network Setup:**

- K clients, each with own task t_k and model w_k ∈ ℝ^d
- Local data (X_k, y_k): instructions and expected LLM outputs
- Clients interact through Parameter Server (PS)
- Modeled as connected graph G with adjacency matrix M

**Graph Representation:**

- Adjacency matrix M: entries quantify task similarity
- Diagonal matrix D: sum of all neighboring connections
- Laplacian matrix: L = D - M

#### Objective Function

$$J(W) = F(W) + \lambda R(W)$$

Where:

- **F(W):** Global loss (local training losses summed)
- **λ:** Regularization hyperparameter controlling collaboration degree
- **R(W):** Laplacian regularization (task similarity enforcement)

**Global Loss:**
$$F(W) = \sum_{k=1}^K f_k(w_k)$$

**Regularization Term (Task Similarity):**
$$R(W) = W^T L W = \frac{1}{2} \sum_{k=1}^K \sum_{\ell \in N_k} a_{k\ell} \|w_k - w_\ell\|^2$$

- a\_{kℓ}: Quantifies similarity between tasks k and ℓ
- Higher a\_{kℓ} ⇒ stronger relationship between tasks
- When λ = 0: Traditional FL (no collaboration)
- When λ > 0: Encourage similar tasks to align models

#### Training Algorithm

**Local Update (at client k):**

1. Client performs R local rounds: ΔW*{k,R} = InstructionTuning(ΔW*{k,r})
2. Each round: forward pass, backward pass, SGD update
3. Send locally updated LoRA matrices ΔW_k to server

**Server Update (global adjustment):**
$$w_k^{(t+1)} = w_{k,R}^{(t)} - \eta\lambda \sum_{\ell \in N_k} a_{k\ell}(w_{k,R}^{(t)} - w_{\ell,R}^{(t)})$$

- Aligns clients with similar tasks
- Encourages models to be closer to similar-task clients
- ΔW update formula (for LoRA):
  $$\Delta W_k = \Delta W_{k,R} - \eta\lambda \sum_{\ell \in N_k} a_{k\ell}(\Delta W_{k,R} - \Delta W_{\ell,R})$$

### LoRA Integration (Memory Efficiency)

**Problem:** Full parameter fine-tuning requires massive GPU memory for storing:

- Model parameters
- Activation values (for backprop)
- For LLMs: tens of gigabytes for billion-parameter models

**LoRA Solution:** Low-Rank Decomposition
$$W_0 + \Delta W = W_0 + BA$$

Where:

- W_0: Pre-trained weight matrix (frozen)
- ΔW: Accumulated gradient updates
- B ∈ ℝ^{d×r}, A ∈ ℝ^{r×v}: Low-rank matrices
- r: Rank (r ≪ d, v)
- Only update B and A, not W_0

**Hypothesis:** Projecting LLMs into lower-dimensional spaces preserves performance while enabling efficient optimization

### Experimental Results (Preliminary)

**Motivation Experiment:**

- Model: DataJuicer LLM
- Dataset: Natural Instructions (NI)
- Compared: FMTL (MIRA) vs FedKSeed vs Fully Local

**Key Finding:** Global model performs worse than local models on certain tasks with high variance (when local dataset is small). FMTL balances global and local performance.

### Advantages and Limitations

**Advantages:**

- ✓ Addresses data heterogeneity through task clustering
- ✓ Reduced computation/communication with LoRA
- ✓ Task-specific knowledge sharing via Laplacian regularization
- ✓ Outperforms standard FL in heterogeneous settings

**Limitations:**

- ✗ Not designed for split learning (models stay locally)
- ✗ Centralized server required for graph/similarity computation
- ✗ Assumes known task similarity graph (or must be learned)
- ✗ No device heterogeneity support (all clients have similar resources)

### Key Equations Summary

**Training objective:**

- Local: ΔW*{k,R} = InstructionTuning(ΔW*{k,r})
- Server: ΔW*k = ΔW*{k,R} - ηλ∑ a*{kℓ}(ΔW*{k,R} - ΔW\_{ℓ,R})

**Convergence:** Achieved through alternating local and global updates

---

## 4. HSplitLoRA: Heterogeneous Split Parameter-Efficient Fine-Tuning

### Paper Details

**Pages:** 16  
**Authors:** Zheng Lin, Yuxin Zhang, Zhe Chen, Zihan Fang, Xianhao Chen, Praneeth Vepakomma, Wei Ni, Jun Luo, Yue Gao  
**Publication:** arXiv:2405.07520 (May 2024)

### Main Contributions

1. **First heterogeneous PEFT framework** integrating SL and LoRA for LLMs
2. **Important weight identification scheme** (contribution-based selection)
3. **Dynamic rank and split point configuration** (adapts to device budgets)
4. **Noise-free heterogeneous adapter aggregation** (no information loss)
5. **Extensive experimental validation** (GPT2-S, GPT2-M, LLaMA-2-7B)

### Core Problem

**Three Critical Challenges:**

**Challenge 1: Limited Computing Resources**

- Client devices: NVIDIA GeForce RTX 3050 in laptops
- Full LoRA configuration on all trainable weights impractical
- Need: Selective fine-tuning of important weights

**Challenge 2: Heterogeneous Computing Resources**

- Device unavailability: disconnections, network instability, insufficient resources
- Degraded training performance when devices fail
- Need: Adaptive configurations per device

**Challenge 3: Heterogeneous Adapter Aggregation**

- Different clients have different LoRA ranks
- Structural discrepancies make direct averaging infeasible
- Need: Aggregation without introducing noise

### System Design

#### Important Weight Identification

**Goal:** Identify which weights contribute most to LLM training

**Approach:** Measure weight importance based on:

- Gradient magnitude
- Weight sensitivity analysis
- Contribution to loss reduction

**Process:**

1. For each trainable weight, compute importance score
2. Rank weights by importance
3. Select top weights for fine-tuning (within memory budget)
4. Configure higher LoRA ranks for important weights

**Pilot Study Results:**

- Normalized computing overhead: FT > 500× LoRA(r=2)
- Normalized communication overhead: FT > 100× LoRA(r=2)
- LoRA significantly reduces both computation and communication

#### Dynamic Rank Configuration

**Goal:** Adjust LoRA decomposition ranks for heterogeneous device budgets

**Process:**

1. Determine important weights per task cluster
2. Assign higher ranks to important weights
3. Respect each device's GPU memory budget
4. Different absolute rank values per device
5. Similar patterns within clusters

**Example Configuration:**

```
Task Cluster 1 (similar task):
  Device A (24GB GPU):  ranks [8, 6, 4]
  Device B (8GB GPU):   ranks [6, 4, 2]
  Device C (1GB GPU):   ranks [4, 2, 1]
```

**Memory Reduction:** 25-50% for client-side LLMs

#### Dynamic Split Point Selection

**Goal:** Partition model layers between client and server based on device budget

**Considerations:**

- Client-side: Early layers (embedding, initial transformers)
- Server-side: Deep layers (bulk of computation)
- Tradeoff: Shallow split ⇒ low client memory but high communication
- Deeper split ⇒ less communication but higher client burden

**Algorithm:**

1. Estimate GPU memory for different split points
2. Select split point respecting device memory constraint
3. Adjust split point if device changes or performs poorly

#### Noise-Free Heterogeneous Adapter Aggregation

**Problem:** Standard averaging doesn't work with different ranks

```
Client 1: rank 4 → 4 low-rank matrices
Client 2: rank 6 → 6 low-rank matrices
Direct average: ??? (incompatible dimensions)
```

**Solution: Concatenation and Merge**

1. **Concatenate** low-rank decomposition matrices:

   - Client 1: [B₁, A₁] (rank 4)
   - Client 2: [B₂, A₂] (rank 6)
   - Merged: [B₁|B₂, A₁|A₂] (rank 10)

2. **Weighted Average** in merged space:

   - w₁ × [B₁|B₂, A₁|A₂] + w₂ × [B₃|B₄, A₃|A₄]

3. **Decompose** back to original ranks (via SVD or similar)

**Advantage:** No information loss during aggregation (vs. standard averaging)

### Experimental Results

#### Test Datasets and Models

- **Models:** GPT2-S, GPT2-M, LLaMA-2-7B
- **Task:** End-to-End (E2E) NLG Challenge
- **Baselines:** CenLoRA, FedLoRA, SplitLoRA

#### Performance Metrics

**GPT2-S Results (Table I):**
| Method | BLEU | NIST | METEOR | ROUGE-L | CIDEr |
|--------|------|------|--------|---------|-------|
| CenLoRA(r=8) | 70.57 | 8.8557 | 0.4688 | 72.17 | 2.5405 |
| SplitLoRA(r=8) | 69.18 | 8.7189 | 0.4631 | 71.30 | 2.5156 |
| HSplitLoRA | ~70.0 | 8.8+ | 0.46+ | 71.5+ | 2.5+ |

**Memory Efficiency (Table I):**

- CenLoRA(r=8): 0.248M (GPT2-S)
- SplitLoRA(r=8): 0.062M
- HSplitLoRA: Reduces client-side by 25-50%

**Convergence Speed:**

- HSplitLoRA achieves target accuracy faster than SplitLoRA
- Better scalability with more clients

### Advantages and Limitations

**Advantages:**

- ✓ First to combine SL + PEFT for heterogeneous devices
- ✓ Achieves target accuracy significantly faster
- ✓ Supports wide range of device budgets (1GB to 24GB)
- ✓ Noise-free aggregation preserves information
- ✓ Extensive experimental validation

**Limitations:**

- ✗ No task clustering (all clients treated uniformly)
- ✗ Important weight selection may be data/model dependent
- ✗ Complexity: Must estimate importance, select split point, adjust ranks
- ✗ Limited to encoder/decoder-only transformers (initially)

### Key Algorithms Summary

**Important Weight Identification:**

- Compute gradient-based importance for each weight
- Rank and select top-k important weights
- Higher LoRA ranks for important weights

**Adaptive Configuration:**

- Split point: s_layer (selected based on device memory)
- LoRA ranks: r_i (per layer, per device)
- Both adapt to device budget constraints

**Noise-Free Aggregation:**

- Concatenate low-rank matrices
- Weighted average in concatenated space
- Decompose back to original ranks

---

## 5. SplitLoRA: A Split Parameter-Efficient Fine-Tuning Framework

### Paper Details

**Pages:** 9  
**Authors:** Zheng Lin, Xuanjie Hu, Yuxin Zhang, Zhe Chen, Zihan Fang, Xianhao Chen, Ang Li, Praneeth Vepakomma, Yue Gao  
**Publication:** arXiv:2407.02590 (July 2024)  
**Status:** First open-source benchmark for SL LLM fine-tuning

### Main Contributions

1. **First SL LLM fine-tuning framework** (SplitLoRA)
2. **Combines SFL + LoRA** for efficiency
3. **Open-source benchmark** for SL LLM research
4. **Client-side LoRA adapter aggregation**
5. **Extensive performance evaluation**

### Core Problem

**Data Scarcity & Privacy:**

- High-quality public datasets depleting (before 2026)
- IoT devices collecting massive private data
- Cannot share data publicly (privacy, regulatory, physical constraints)
- Need: Collaborative training on distributed private data

**Computational Challenges:**

- FL: Heavy computing burden on clients
- SL: Offloads to server but model partitioning overhead
- SFL: Merges FL + SL advantages
- LoRA: Reduces fine-tuning parameters
- Combination: SplitLoRA = SFL + LoRA

### Framework Architecture

#### SFL (Split Federated Learning) Pipeline

**Stages (executed each round):**

1. **Client-side Forward Propagation (a1)**

   - Input: Local batch of data
   - Output: Intermediate activations (h)

2. **Activations Transmission (a2)**

   - Client sends h to server

3. **Server-side Forward & Backward (a3)**

   - Compute loss, gradients ∇h

4. **Gradients Transmission (a4)**

   - Server sends ∇h back to client

5. **Client-side Backward Propagation (a5)**
   - Client computes gradients for local layers
   - Updates LoRA adapters via SGD

#### Client-Side Aggregation (every I rounds)

**Purpose:** Aggregate heterogeneous client-side LoRA adapters

**Stages:**

- **b1:** Upload client LoRA adapters to aggregation point
- **b2:** Aggregation (weighted average in compatible space)
- **b3:** Download aggregated adapter

### Technical Approach

#### LoRA Integration

**Low-Rank Decomposition:**
$$W_0 + \Delta W = W_0 + BA$$

- Original weight matrix frozen
- Only train low-rank matrices B and A
- Rank r ≪ d (dimension)

**Hypothesis:** Lower-dimensional projection preserves LLM performance

**Computational Benefits:**

- Reduced gradient computation
- Minimal activation storage
- Parameters reduced by factor of r/d

#### Model Partitioning Strategy

**Split Point Selection:**

- Partition at transformer layer l
- Client: embedding layer + first l layers + LoRA adapters
- Server: remaining (n-l) layers + task head + LoRA adapters

**Considerations:**

- Shallow split: High communication, low client memory
- Deep split: Low communication, high client memory
- Optimal: Depends on network speed, device memory, task

#### System Setup (E2E Dataset)

**Models:**

- GPT-2 (Small & Medium)

**Task:**

- End-to-End NLG Challenge
- Sequence-to-sequence generation

**Configuration:**

- Multiple clients with local data
- Each client trains local LoRA adapters
- Aggregation every I rounds (e.g., I=10)

### Experimental Results

#### Convergence and Accuracy

**GPT2-S (BLEU Metric):**
| Method | BLEU | NIST | METEOR | ROUGE-L | CIDEr |
|--------|------|------|--------|---------|-------|
| CenLoRA(r=1) | 67.95 | 8.6973 | 0.4421 | 68.96 | 2.3412 |
| SplitLoRA(r=1) | 70.26 | 8.8274 | 0.4664 | 71.73 | 2.5267 |
| SplitLoRA(r=2) | 70.04 | 8.8031 | 0.4670 | 71.68 | 2.5233 |
| SplitLoRA(r=4) | 70.09 | 8.8075 | 0.4667 | 71.60 | 2.5370 |
| SplitLoRA(r=8) | 69.18 | 8.7189 | 0.4631 | 71.30 | 2.5156 |

**Key Finding:** SplitLoRA outperforms CenLoRA and FedLoRA

**GPT2-M (BLEU Metric):**
| Method | BLEU | NIST | METEOR | ROUGE-L | CIDEr |
|--------|------|------|--------|---------|-------|
| CenLoRA(r=1) | 69.86 | 8.7679 | 0.4650 | 71.20 | 2.5028 |
| SplitLoRA(r=1) | 70.26 | 8.8274 | 0.4664 | 71.73 | 2.5267 |
| SplitLoRA(r=4) | 70.09 | 8.8075 | 0.4667 | 71.60 | 2.5370 |
| SplitLoRA(r=8) | 69.18 | 8.7189 | 0.4631 | 71.30 | 2.5156 |

#### Efficiency Metrics

**Memory Usage (Table I):**

```
CenLoRA(r=1): 0.031M (GPT2-S)
SplitLoRA(r=1): 0.008M  ← 73% reduction
SplitLoRA(r=4): 0.031M
SplitLoRA(r=8): 0.062M
```

**Communication Overhead:**

- Activations transmission: Small data size
- Gradient transmission: Compatible with SFL
- vs. Full parameter transfer: 100× less communication

**Convergence Time:**

- SplitLoRA converges faster than CenLoRA
- Client-side aggregation improves efficiency
- Parallelization from FL benefits retained

### Advantages and Limitations

**Advantages:**

- ✓ First SL LLM framework combining SFL + LoRA
- ✓ Open-source benchmark (enables community research)
- ✓ Achieves excellent accuracy (70+ BLEU on GPT2-M)
- ✓ Significant memory reduction (25-50%)
- ✓ Fast convergence
- ✓ Privacy-preserving (no raw data shared)

**Limitations:**

- ✗ Homogeneous setup (all clients identical)
- ✗ No task clustering (task-blind)
- ✗ No device heterogeneity support
- ✗ Fixed split point (not adaptive)
- ✗ Client must store intermediate activations

### Implementation Details

**Tech Stack:**

- PyTorch
- HuggingFace Transformers
- PEFT LoRA
- Custom SFL framework

**Framework Components:**

- Client-side model handling
- Server-side computation
- Activation transmission protocol
- Gradient aggregation logic
- LoRA adapter management

---

## 6. Privacy-Aware Split Federated Learning for LLM Fine-Tuning over IoT

### Paper Details

**Pages:** 12  
**Journal:** IEEE Internet of Things Journal (accepted for publication)  
**Citation:** DOI 10.1109/JIOT.2025.3600269  
**Authors:** Xiaopei Chen, Wen Wu, Fei Ji, Yongguang Lu, Liang Li

### Main Contributions

1. **Privacy quantification metric** (Fisher information-based)
2. **Multi-objective optimization** (privacy × efficiency × convergence)
3. **ε-constraint BCD algorithm** (joint optimization of split layer, power, bandwidth)
4. **Comprehensive evaluation** on heterogeneous IoT devices
5. **Privacy-utility-energy tradeoff** analysis

### Core Problem

**IoT + LLM Convergence:**

- IoT devices generate 3.56 billion devices worldwide
- Generate rich, real-time, domain-specific data streams
- LLMs can enhance IoT services
- **Challenge:** Privacy preservation + resource constraints + accuracy

**Accuracy-Efficiency-Privacy Trilemma:**

- Shallow split layer: Exposes raw data patterns in smashed data → privacy leakage
- Deep split layer: Shifts computational burden to resource-constrained devices
- Privacy techniques (e.g., DP noise): Degrade convergence differently at different layers

**Critical Limitation of Existing SFL:**

- Neglect privacy vulnerabilities in client-server data exchange
- LLMs exhibit "not-too-far" property (features preserved after fine-tuning)
- Easier for adversaries to perform data reconstruction attacks

### Privacy Vulnerabilities in SFL

**Problem:**

- Intermediate activations (smashed data) transmitted between client and server
- Adversaries can infer original data through reconstruction attacks
- Shallow split layer: More raw data patterns visible
- Deep split layer: Less privacy leakage, but computational burden on IoT

**Privacy-Leakage Characterization:**

The paper quantifies privacy leakage using **Fisher information**:
$$\Psi_u \propto \text{Fisher Information in smashed data}$$

- Higher Fisher info → More data leakage
- Shallow split layer → Higher Fisher info
- Deep split layer → Lower Fisher info but higher client burden

### Technical Approach

#### System Model

**Network:**

- U IoT client devices
- Edge server
- Each device u has:
  - Local dataset D_u
  - Computing capacity C_u
  - Available memory M_u
  - Network conditions (channel gain h_u)

**SFL Architecture:**

- Model split at layer l_u (device-specific)
- Client sub-model: W_u^c + LoRA module R_u^c
- Server sub-model: W_u^s + LoRA module R_u^s
- Exchange: Smashed data h_u and gradients ∇h_u

#### Multi-Objective Optimization Problem

**Three objectives to optimize:**

1. **Privacy Leakage (Minimize):**
   $$\Psi_u \propto F_u(l_u)$$

   - Fisher information at split layer l_u

2. **Fine-tuning Convergence Time (Minimize):**
   $$T_{total} = T_{computation} + T_{communication}$$

   - Client-side computation time
   - Server-side computation time
   - Uplink/downlink transmission time

3. **Device Energy Consumption (Minimize):**
   $$E_{total} = E_{computation} + E_{communication}$$
   - Computation energy (function of processing)
   - Communication energy (function of transmit power P_u, bandwidth B_u)

**Optimization Variables:**

- Split layer l_u (for each device u)
- Transmit power P_u (for each device)
- Bandwidth allocation B_u (for each device)

**Subject to:**

- Memory constraint: Memory used ≤ M_u
- Network constraint: Bandwidth allocated ≤ Total available
- Power constraint: Transmit power ≤ Max power

**Formulation:**
$$\min_{l_u, P_u, B_u} \text{Privacy}(l_u), \text{Time}(l_u, P_u, B_u), \text{Energy}(P_u, B_u)$$

#### Solution: ε-Constraint BCD Algorithm

**ε-Constraint Method:**

- Convert 3-objective problem into single-objective with constraints
- Ensure weak Pareto optimality
- **λ-constraint formulation:**
  $$\min f_1(x) \text{ s.t. } f_2(x) ≤ ε_2, f_3(x) ≤ ε_3$$

**Block Coordinate Descent (BCD):**
Decompose into 3 subproblems:

1. **Bandwidth Allocation (b_u):**

   - Monotonicity analysis
   - Direct solution via monotonic property

2. **Transmit Power (P_u):**

   - Bisection search method
   - Binary search for optimal power level

3. **Split Layer (l_u):**
   - Successive Convex Approximation (SCA)
   - Iterative refinement

**Algorithm Flow:**

```
Initialize split layers, power, bandwidth
Repeat:
  1. Optimize bandwidth using monotonicity
  2. Optimize power using bisection
  3. Optimize split layer using SCA
Until convergence
Return: l_u*, P_u*, B_u*
```

### Experimental Results

**Simulation Setup:**

- U = 10-100 IoT devices
- Models: BERT, GPT-2
- Datasets: GLUE, E2E
- Heterogeneous channel conditions, device capabilities

**Performance Improvements (vs. Baselines):**

- **24% faster convergence** (reduced training time)
- **40% lower energy consumption** (optimized power/bandwidth)
- **7% reduced privacy leakage** (better split layer selection)
- **Competitive accuracy** maintained

**Tradeoff Analysis:**

- Privacy vs. Efficiency: Deeper split ⇒ more privacy but less efficiency
- Energy vs. Convergence: Higher power ⇒ faster but more energy
- Device heterogeneity: Algorithm adapts split/power per device

### Technical Innovations

**Privacy Quantification:**

- Fisher information-based metric
- Layer-wise privacy assessment
- Quantifies information leakage in intermediate activations

**Analytical Models:**

- Relates privacy leakage to split layer choice
- Connects convergence time to device parameters
- Links energy consumption to network conditions

**Multi-Objective Formulation:**

- Captures real-world tradeoffs
- Respects device heterogeneity
- Provides principled optimization

### Advantages and Limitations

**Advantages:**

- ✓ First to address privacy-efficiency-convergence tradeoff
- ✓ Principled privacy quantification (Fisher information)
- ✓ Tailored for IoT heterogeneity
- ✓ Joint optimization of split, power, bandwidth
- ✓ Demonstrates practical improvements

**Limitations:**

- ✗ Assumes known Fisher information (may require estimation)
- ✗ IoT focus (less attention to other edge devices)
- ✗ Does not address task heterogeneity
- ✗ Privacy metric may be model/data dependent
- ✗ Assumes semi-honest threat model

---

## 7. VFLAIR-LLM: Comprehensive Framework and Benchmark

### Paper Details

**Pages:** 12  
**Conference:** KDD'25 (August 3-7, 2025, Toronto, ON)  
**Authors:** Zixuan Gu, Qiufeng Fan, Long Sun, Yang Liu, Xiaojun Ye  
**Affiliation:** Tsinghua University, Wuxi Innovation Center  
**Availability:** https://github.com/FLAIR-THU/VFLAIR-LLM

### Main Contributions

1. **VFLAIR-LLM:** Lightweight, extensible SL-LLM framework
2. **Comprehensive attack/defense benchmark:** 5 attacks × 9 defenses
3. **Multiple partition settings:** HT (Head-Tail), HBT (Head-Body-Tail)
4. **16 LLM types supported:** BERT, GPT2, LLaMA, Mistral, etc.
5. **Privacy-utility tradeoff analysis** with actionable insights

### Framework Overview

#### Architecture Variants

**Head-Tail (HT) SL-LLM:**

```
Data Party: Embedding + n_head encoders/decoders
     ↓ (intermediate H_1)
Model Party: Remaining n_tail encoders/decoders + head layer
```

- Model Head on data party (client)
- Model Tail on model party (server)
- Assumes model party can access inference results or labels

**Head-Body-Tail (HBT) SL-LLM:**

```
Data Party: Embedding + n_head encoders/decoders
     ↓ (intermediate H_1)
Model Party: n_body encoders/decoders (body)
     ↓ (intermediate H_2)
Data Party: Remaining n_tail encoders/decoders + head layer
```

- Three-way split
- Model body on server
- Model tail back on client
- Hinders direct label inference from model party

#### LLM Support Matrix

**16 LLM Types:**

- **Encoder-only:** BERT, RoBERTa, Albert
- **Decoder-only:** GPT2, LLaMA, Baichuan2, ChatGLM2, Falcon, Gemma, Mamba, Mistral, Qwen2, DeepSeek, MiniCPM, Qwen2-VL
- **Encoder-Decoder:** Qwen, T5

**3 LLM Architectures:**

1. Classification (sequence classification)
2. Generation (causal LM)
3. Span-based QA

#### Task Types and Datasets

**3 Task Types:** (18 datasets)

1. **Sequence Classification:**

   - SST2, CoLA, MRPC, QQP, MNLI, QNLI, RTE, WNLI, Yelp, STS-B

2. **Text-span based QA:**

   - SQuAD (question answering)

3. **Generation (next token prediction):**
   - Lambada, Alpaca, GSM8K, CodeAlpaca

### Attack and Defense Evaluation

#### Attacks (5 Types)

**1. Vanilla Model Inversion (VMI):**

- 2-step data reconstruction
- Infer original input embedding E(X')
- Recover tokens via cosine similarity with embedding matrix
- Success metric: Text recall rate

**2. Recursive Model Inversion (RMI):**

- Iterative refinement of recovered data
- Uses multiple optimization steps
- Higher success rate than VMI

**3. Bidirectional Enhanced Semantic Reconstruction (BiSR):**

- Novel attack targeting bidirectional data reconstruction
- Leverages semantic knowledge
- Enhanced compared to previous attacks

**4. Label Inference Attacks (LIA):**

- Infer task labels from intermediate activations
- Success metric: Label recovery accuracy

**5. Additional attacks:**

- Model property inference
- Membership inference
- etc.

#### Defenses (9 Types)

**Perturbation-Based Defenses:**

1. **Differential Privacy (DP):** Add Gaussian noise to intermediates/gradients

   - Hyperparameters: ε ∈ {500, 100, 70, 50}

2. **Sparsification (SP):** Drop activations randomly

   - Hyperparameters: r ∈ {95%, 96%, 97%, 98%}

3. **Split-N-Denoise (SnD):** Add noise then denoise
   - Hyperparameters: η ∈ {1e-5, 1e-4, 1e-3, 100, 10}

**Text-Level Perturbations:** 4. **SanText:** Sanitize text-level information 5. **CusText:** Custom text perturbation 6. **RanText:** Random text perturbations

**Gradient-Based Defenses:** 7. **Adversarial Training (AT):** Add adversarial perturbations 8. **Mutual Information Disentanglement (MID):** Disentangle information 9. **Token-level Obfuscation (TO):** Obfuscate token information

#### Evaluation Metrics

**Main Task Performance (MP):**

- Classification: Accuracy
- Regression: Pearson correlation
- QA (span-based): Exact match score
- QA (generation): ROUGE score
- Code generation: CodeBLEU
- Math: Problem-solving accuracy

**Attack Performance (AP):**

- MIA: Text recall rate
- LIA: Label recovery accuracy

**Defense Capability Score (DCS):**
$$\text{DCS} = \alpha × \text{MP} + (1-\alpha) × (1 - \text{AP})$$

- Balances utility (MP) and privacy (low AP)
- Default: α = 0.5

**Comprehensive DCS (C-DCS):**

- Aggregate across multiple attack-defense pairs
- Task-DCS (T-DCS): Per-task performance
- Overall ranking of defenses

### Comprehensive Benchmark Results

#### Fine-Tuning Results (HB vs HBT Settings)

**SST2-BERT (Classification):**

```
HB (Head-Body):  MP = 0.920 ± 0.001
HBT (Head-Body-Tail):  MP = 0.919 ± 0.001
```

- HBT slightly better privacy (model tail on client)
- Similar accuracy

**SQuAD-BERT (QA):**

```
HB:  MP = 0.731 ± 0.001
HBT:  MP = 0.729 ± 0.002
```

- Consistent performance across architectures

**Lambada-GPT2 (Generation):**

```
HB Full-LoRA:  MP = 0.606 ± 0.012
HB Full-Vanilla:  MP = 0.654 ± 0.002
```

- LoRA strategy vs vanilla fine-tuning comparison

#### Attack-Defense Benchmark

**Model Inversion Attacks (VMI, RMI, BiSR):**

- BiSR (novel attack): Highest success rate
- BiSR demonstrates vulnerabilities in SL-LLM even with defenses

**Label Inference Attacks (LIA, NS):**

- Effective against shallow split layers
- Mitigated by deeper splits (HBT)

**Defense Rankings (DCS):**

```
Top Defenses:
1. MID (λ=0.5):  DCS = 0.8513
2. MID (λ=0.1):  DCS = 0.8438
3. MID (λ=0.01): DCS = 0.8437
4. MID (λ=0.001): DCS = 0.8250
...
DP (ε=50):  DCS = 0.7883
SP (97%):   DCS = 0.7039
```

- MID most effective across datasets
- DP-DP provides good privacy-utility tradeoff
- Sparsification less effective than perturbation methods

#### Privacy-Utility Tradeoff

**MP-AP Graph Analysis:**

- X-axis: MP (higher = better utility)
- Y-axis: AP (lower = better privacy)
- Optimal solutions: Bottom-right quadrant

**Key Findings:**

- No single defense dominates (pareto frontier)
- Different defenses optimal for different objectives
- MID balances privacy-utility well
- Stronger defenses degrade accuracy more

#### DCS Gap Distribution

**LoRA vs Vanilla Fine-Tuning:**

- Full-LoRA DCS - Full-Vanilla DCS
- Most defenses: Δ-DCS ≈ 0.01-0.05
- MID averages: Δ-DCS = 0.0071
- Suggests LoRA compatible with privacy methods

### Work Modes and Deployment

**Standalone Mode:**

- Simulation on single machine
- Fast prototyping and research
- No distributed infrastructure needed

**Distributed Mode:**

- Real-world deployment across parties
- Client-server communication
- Production scenarios

### Advantages and Limitations

**Advantages:**

- ✓ Most comprehensive SL-LLM framework/benchmark
- ✓ 16 LLM types, 18 datasets, multiple tasks
- ✓ Extensive attack/defense evaluation (5×9 combinations)
- ✓ Lightweight and highly extensible
- ✓ Modular design (easy to add new attacks/defenses)
- ✓ Production-ready with both standalone and distributed modes
- ✓ Clear privacy-utility tradeoff insights
- ✓ Open-source and actively maintained

**Limitations:**

- ✗ Assumes semi-honest threat model (parties honest but curious)
- ✗ Limited to passive attacks (no Byzantine/active attacks)
- ✗ Computational overhead not thoroughly characterized
- ✗ Limited guidance on defense selection for specific scenarios

### Key Takeaways

1. **BiSR attack** is novel and effective - demonstrates ongoing privacy risks
2. **MID defense** provides best privacy-utility balance
3. **LoRA fine-tuning** slightly improves privacy-utility tradeoff
4. **HBT partition** better for privacy-sensitive scenarios
5. **No universal best defense** - choice depends on application requirements

---

## SYNTHESIS AND ATLAS INTEGRATION

### How Papers Support ATLAS Implementation

#### 1. **ATLAS Foundation: Problem Statement** (ATLAS Base Spec)

- Identifies gap between MIRA (task-aware, centralized) and HSplitLoRA (heterogeneous, task-blind)
- Proposes combining both + split learning
- Defines 3-phase implementation plan

#### 2. **MIRA Contribution: Multi-Task Learning**

- Task clustering via Laplacian regularization
- Graph-based task similarity
- Knowledge sharing between similar tasks
- **ATLAS Use:** Phase 1 task fingerprinting, k-Means clustering

#### 3. **HSplitLoRA Contribution: Heterogeneous Adaptation**

- Important weight identification
- Dynamic rank configuration per device
- Noise-free heterogeneous aggregation
- **ATLAS Use:** Phase 2 configuration assignment, Phase 4 aggregation

#### 4. **SplitLoRA Contribution: Split Learning Framework**

- SFL + LoRA combination (proven effective)
- Client-side LoRA adapter aggregation
- Open-source benchmark implementation
- **ATLAS Use:** Phase 3 training loop, SFL architecture

#### 5. **Privacy-Aware SFL Contribution: Optimization Framework**

- Multi-objective optimization (privacy × efficiency × convergence)
- ε-constraint BCD algorithm
- Joint optimization of split layer, power, bandwidth
- **ATLAS Use:** Optimization of configuration assignments

#### 6. **VFLAIR-LLM Contribution: Privacy Evaluation**

- Comprehensive attack/defense benchmark
- Privacy quantification metrics
- Actionable recommendations
- **ATLAS Use:** Phase final evaluation + privacy assessment

### Implementation Strategy Integration

```
ATLAS Implementation Pipeline:
├── Phase 1: Task Discovery (MIRA-inspired)
│   ├── Compute gradient-based task embeddings
│   ├── k-Means clustering (Laplacian graph structure)
│   └── Output: Semantic neighborhoods
│
├── Phase 2: Configuration (HSplitLoRA-inspired)
│   ├── Identify important weights per cluster
│   ├── Assign heterogeneous LoRA ranks (per device budget)
│   ├── Select split point (adapt to memory constraint)
│   └── Output: Device-specific configurations
│
├── Phase 3: Training (SplitLoRA-inspired)
│   ├── Execute SFL (client-server split)
│   ├── Update LoRA adapters locally
│   ├── Exchange intermediate activations
│   └── Output: Trained adapters
│
├── Phase 4: Aggregation (HSplitLoRA aggregation)
│   ├── Aggregate LoRA adapters (heterogeneous ranks)
│   ├── Noise-free concatenation + merge
│   ├── Broadcast updated adapters
│   └── Output: Cluster-level global model
│
└── Phase 5: Privacy Evaluation (VFLAIR-LLM)
    ├── Run attacks (VMI, RMI, BiSR, LIA)
    ├── Apply defenses (DP, SP, AT, etc.)
    ├── Compute DCS metrics
    └── Output: Privacy assessment report
```

---

## KEY TECHNICAL CHALLENGES TO ADDRESS

### 1. Task Fingerprint Quality

**Challenge:** Gradient-based fingerprints may not capture full task similarity
**Solution (ATLAS):** Use PCA on gradients, validate clustering quality
**From Papers:** MIRA uses Laplacian regularization; consider graph refinement

### 2. Heterogeneous Aggregation Complexity

**Challenge:** Different ranks across clients make aggregation non-trivial
**Solution (ATLAS):** Noise-free concatenation + weighted merge (HSplitLoRA approach)
**Advantage:** No information loss during aggregation

### 3. Optimal Split Point Selection

**Challenge:** Trade-off between client computation and communication
**Solution (ATLAS):** Device-aware selection (inspired by Privacy-Aware SFL)
**Approach:** Consider device memory, network bandwidth, task type

### 4. Privacy Guarantees

**Challenge:** SFL vulnerable to model inversion and label inference attacks
**Solution (ATLAS):** Combine defenses (DP + MID) based on threat model
**Reference:** VFLAIR-LLM provides defense rankings and tradeoff analysis

### 5. Convergence with Task Clustering

**Challenge:** Task clustering may separate similar clients → slower global convergence
**Solution (ATLAS):** Cluster-wise aggregation, optional inter-cluster communication
**Approach:** Balance intra-cluster specialization with global model performance

### 6. Communication Overhead

**Challenge:** Gradient fingerprints + intermediate activations + adapter transmission
**Solution (ATLAS):** Compress fingerprints (64-D), quantize activations, sparse aggregation
**Target:** Minimize communication while preserving performance

---

## WHAT NEEDS TO BE IMPLEMENTED FROM SCRATCH

### 1. **Task Embedding and Clustering Module** ✓

- Gradient-based task fingerprinting (64-D vectors)
- k-Means clustering on fingerprints
- Graph construction from cluster assignments
- Similarity matrix computation

### 2. **Weight Importance Scoring** ✓

- Compute importance for each weight/layer
- Methods: Gradient magnitude, Hessian-based, contribution analysis
- Ranking and selection of important weights
- Validation on different models and tasks

### 3. **Configuration Assignment Algorithm** ✓

- Determine LoRA rank per layer per device
- Select split point given memory budget
- Ensure similar patterns within clusters
- Dynamic adjustment based on device performance

### 4. **Noise-Free Heterogeneous Aggregation** ✓

- Concatenate low-rank matrices (different ranks)
- Weighted average in concatenated space
- Decompose back to original ranks (SVD)
- Verify no information loss

### 5. **Split Learning Training Loop** ✓

- Client-side forward/backward propagation
- Intermediate activation transmission
- Server-side gradient computation
- Gradient transmission back to client
- LoRA adapter updates

### 6. **Cluster-Aware Aggregation** ✓

- Separate aggregation per cluster
- Weighted averaging of adapters
- Broadcasting to cluster members
- Optional inter-cluster synchronization

### 7. **Privacy Attack/Defense Implementation** ⚠️

- Implement key attacks: VMI, RMI, LIA
- Implement key defenses: DP, MID, AT
- Use VFLAIR-LLM library (already available)
- Compute privacy metrics (DCS)

### 8. **Evaluation Framework** ✓

- Accuracy metrics (task-specific)
- Communication overhead tracking
- Memory usage profiling
- Convergence speed measurement
- Privacy-utility tradeoff analysis

---

## INTEGRATION POINTS AND DEPENDENCIES

### Data Dependencies

```
Papers → Components
MIRA → Task Clustering
HSplitLoRA → Heterogeneous Adaptation, Aggregation
SplitLoRA → Training Loop, Framework
Privacy-Aware SFL → Optimization, Configuration
VFLAIR-LLM → Privacy Evaluation, Attack/Defense
```

### Implementation Order (Recommended)

```
Week 1-2: Setup + SplitLoRA training loop (foundation)
Week 3-4: Add task clustering (MIRA component)
Week 5-6: Add weight importance + heterogeneous config (HSplitLoRA component)
Week 7-8: Implement aggregation (HSplitLoRA aggregation)
Week 9-10: Integration + optimization (combine all components)
Week 11-12: Privacy evaluation + benchmarking (VFLAIR-LLM)
```

### External Libraries Required

```
✓ PyTorch (torch) - Deep learning framework
✓ HuggingFace Transformers - LLM models
✓ PEFT - LoRA implementation
✓ scikit-learn - k-Means clustering, PCA
✓ VFLAIR-LLM - Privacy attacks/defenses
```

---

## POTENTIAL CHALLENGES AND MITIGATION

### Challenge 1: Task Fingerprint Representation

**Issue:** 64-D gradient fingerprint may lose important task information
**Mitigation:**

- Validate fingerprint quality via silhouette analysis
- Experiment with PCA dimensionality (32, 64, 128)
- Compare with other task similarity metrics

### Challenge 2: Cluster Stability

**Issue:** k-Means may produce different clusters across rounds
**Mitigation:**

- Use k-Means++ initialization
- Lock clusters after initial discovery
- Periodic re-clustering with stability checks

### Challenge 3: Heterogeneous Aggregation Correctness

**Issue:** Concatenation + merge may lose information
**Mitigation:**

- Verify via matrix rank analysis
- Compare reconstructed weights with original
- Test on simple toy problems first

### Challenge 4: Communication Bottleneck

**Issue:** Transmitting fingerprints + activations + gradients adds overhead
**Mitigation:**

- Compress fingerprints (quantization, dimensionality reduction)
- Compress intermediate activations (sparsification)
- Use adaptive communication (send only when necessary)

### Challenge 5: Convergence Guarantees

**Issue:** ATLAS may have slower convergence due to task separation
**Mitigation:**

- Empirical convergence analysis on test datasets
- Optional inter-cluster gradient exchange
- Adaptive cluster merging if convergence slows

### Challenge 6: Privacy-Utility Tradeoff

**Issue:** Privacy defenses (DP, MID) degrade accuracy
**Mitigation:**

- Use VFLAIR-LLM defense recommendations (MID preferred)
- Tune defense hyperparameters per task
- Analyze DCS curves to find good tradeoff points

---

## SUMMARY OF KEY FINDINGS

### What Each Paper Contributes

| Paper                 | Main Contribution            | Key Innovation           | Challenge Addressed                    |
| --------------------- | ---------------------------- | ------------------------ | -------------------------------------- |
| **ATLAS Spec**        | Problem formulation          | Combining 3 approaches   | Heterogeneous + task-aware + efficient |
| **ATLAS V1**          | Detailed plan                | 4-phase architecture     | Practical implementation roadmap       |
| **MIRA**              | Task clustering              | Laplacian regularization | Task heterogeneity                     |
| **HSplitLoRA**        | Heterogeneous adaptation     | Noise-free aggregation   | Device heterogeneity                   |
| **SplitLoRA**         | SFL + LoRA framework         | Open-source benchmark    | Efficient split learning               |
| **Privacy-aware SFL** | Multi-objective optimization | ε-constraint BCD         | Privacy-efficiency-convergence         |
| **VFLAIR-LLM**        | Privacy evaluation framework | Comprehensive benchmark  | Privacy assessment                     |

### Key Equations and Algorithms

**Task Clustering (MIRA):**

- Objective: min_W [F(W) + λR(W)]
- Laplacian regularization encourages task alignment

**Heterogeneous Aggregation (HSplitLoRA):**

- Concatenate: [B₁, B₂] + [A₁, A₂]
- Aggregate: w₁×[B₁, B₂] + w₂×[B₃, B₄]
- Decompose back to ranks

**Privacy Metric (VFLAIR):**

- DCS = α×MP + (1-α)×(1-AP)
- Balances task performance and privacy

### Expected Performance Targets

**ATLAS Objectives:**

- Accuracy: +15-25% vs HSplitLoRA (via task clustering)
- Memory: 30-40% reduction (heterogeneous LoRA ranks)
- Communication: Minimal (split learning)
- Privacy: DCS ≥ 0.7 (with appropriate defenses)
- Convergence: Faster than task-blind baselines

---

## CONCLUSION

The ATLAS project has strong theoretical and empirical foundations from the analyzed papers. The combination of:

1. **MIRA's task-aware learning**
2. **HSplitLoRA's heterogeneous adaptation**
3. **SplitLoRA's split learning framework**
4. **Privacy-aware optimization**
5. **VFLAIR's privacy evaluation**

...creates a feasible, novel approach to LLM fine-tuning on heterogeneous edge devices. All components are grounded in published research and existing open-source implementations are available. The 12-week implementation timeline is realistic with proper project management and clear milestones.

**Next Steps:**

1. Confirm supervisor approval on research direction
2. Determine specific datasets and LLM models
3. Allocate GPU compute budget
4. Begin Phase 1: Task clustering implementation
5. Establish evaluation baselines
