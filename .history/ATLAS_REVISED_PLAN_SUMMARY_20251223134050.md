# ATLAS Project: Revised Implementation Plan
## From TITANIC+MIRA to HSplitLoRA Multi-Task Federated Learning

---

## EXECUTIVE SUMMARY

The original ATLAS project planned to use the **TITANIC dataset with MIRA methodology** for federated learning of Large Language Models. However, due to TITANIC not being open-source and time constraints (2 months), we have **pivoted to a more implementable approach**.

### Revised Strategy
**Combine three open-source methodologies:**
1. **MIRA** - Multi-task learning via graph regularization
2. **HSpLitLoRA** - Heterogeneous split learning with LoRA
3. **SplitLoRA** - Communication-efficient federated training

### Key Advantages
- ✅ **Implementable from papers** (no proprietary data required)
- ✅ **2-month feasible timeline** (12-week structured plan)
- ✅ **Working demo** for supervisor presentation
- ✅ **Novel contribution** (combining MIRA + HSplitLoRA + SplitLoRA)
- ✅ **Comprehensive privacy evaluation** (VFLAIR benchmark)

---

## 1. PROJECT VISION

### Problem Statement
Modern federated learning struggles with:
- **Task heterogeneity:** Different clients have different tasks
- **Device heterogeneity:** Memory/compute constraints vary widely
- **Privacy risks:** Gradient leakage in multi-task scenarios
- **Communication overhead:** Large model updates expensive

### Our Solution: ATLAS
**Adaptive Task-aware Federated Learning with Heterogeneous Splitting**

Combines:
- **Task clustering** (MIRA) to group similar clients
- **Heterogeneous LoRA ranks** (HSpLitLoRA) for device adaptation
- **Split learning** (SplitLoRA) for communication efficiency
- **Privacy evaluation** (VFLAIR) for security assessment

---

## 2. TECHNICAL COMPONENTS

### Component 1: MIRA - Task Clustering

**What it does:**
- Extracts gradient fingerprints from each client
- Clusters clients into task groups using k-Means
- Groups clients with similar learning trajectories

**How it works:**
```
Client 1 gradient → 64-D fingerprint → \
Client 2 gradient → 64-D fingerprint → k-Means → Task Groups
Client 3 gradient → 64-D fingerprint → /
```

**Key equation:**
$$\mathcal{L}_{\text{task}} = \sum_{i} \|w_i\|^2 + \lambda \mathbf{w}^T L \mathbf{w}$$

where $L$ is the graph Laplacian encoding task similarities

**Benefit:** Enables task-specific model adaptation and weighting

### Component 2: HSpLitLoRA - Heterogeneous Configuration

**What it does:**
- Analyzes each device's memory and compute constraints
- Assigns different LoRA ranks to different clients
- Performs noise-free aggregation without padding

**How it works:**
```
Device Profile (2GB) → Rank 4-8
Device Profile (4GB) → Rank 8-16  
Device Profile (8GB) → Rank 16-32
```

**Key insight:**
- Standard LoRA: same rank $r$ for all clients → wastes memory on constrained devices
- HSpLitLoRA: device-aware ranks → optimal memory/accuracy trade-off

**Benefit:** 30-40% memory reduction on constrained devices

### Component 3: SplitLoRA - Communication Efficiency

**What it does:**
- Splits model computation between client and server
- Client handles feature extraction (bottom layers)
- Server handles task-specific heads (top layers)
- Only intermediate activations transmitted

**How it works:**
```
Client: [Input] → Bottom Layers → [Activation] → (Network)
Server: (Network) → [Activation] → Top Layers → [Output]
```

**Communication comparison:**
- Standard FL: $\mathcal{O}(d)$ parameters per update (millions)
- SplitLoRA: $\mathcal{O}(r + h)$ (thousands) → **10-100x reduction**

**Benefit:** Dramatically reduces communication bandwidth

### Component 4: Privacy-Aware Aggregation

**What it does:**
- Merges heterogeneous LoRA weights without noise injection
- Maintains privacy while improving accuracy
- Task-aware weighting for balanced contribution

**Key advantages:**
- No noise injection → better accuracy preservation
- Handles different LoRA ranks per device
- Task clustering ensures balanced aggregation

**Benefit:** Privacy without sacrificing accuracy

### Component 5: Privacy Evaluation

**What it does:**
- Evaluates 5 types of privacy attacks (membership inference, data reconstruction, etc.)
- Tests 9 defense mechanisms (DP, secure computation, etc.)
- Generates comprehensive privacy report

**Metrics:**
- **DCS (Detection-based Capability Score):** ≥ 0.7 target
- **Differential Privacy:** $(ε=1.0, δ=10^{-5})$
- **Attack success rates** for each attack type

**Benefit:** Demonstrates privacy guarantees to supervisors

---

## 3. SYSTEM ARCHITECTURE

### High-Level Flow

```
┌─────────────┐
│ PHASE 1     │  Extract gradients, cluster tasks
│ Clustering  │
└──────┬──────┘
       ↓
┌─────────────┐
│ PHASE 2     │  Assign heterogeneous LoRA ranks
│ Config      │
└──────┬──────┘
       ↓
┌─────────────┐
│ PHASE 3     │  Train with split learning + LoRA
│ Training    │
└──────┬──────┘
       ↓
┌─────────────┐
│ PHASE 4     │  Aggregate with task weighting
│ Aggregation │
└──────┬──────┘
       ↓
┌─────────────┐
│ PHASE 5     │  Evaluate privacy attacks/defenses
│ Privacy     │
└─────────────┘
```

### Core Modules

| Module | Function | Input | Output |
|--------|----------|-------|--------|
| **GradientExtractor** | Extract 64-D task fingerprints | Client gradients | Fingerprints |
| **TaskClusterer** | k-Means clustering on fingerprints | Fingerprints | Task groups |
| **RankAllocator** | Assign LoRA ranks per device | Device profiles | Rank configs |
| **SplitClient** | Client-side training | Local data + rank | LoRA updates |
| **SplitServer** | Server-side computation | Updates | Global state |
| **AggregationEngine** | Merge heterogeneous updates | Client updates | Global model |
| **PrivacyEvaluator** | Privacy attack/defense testing | Model weights | Privacy metrics |

---

## 4. IMPLEMENTATION TIMELINE

### Week 1-2: Task Clustering (Phase 1)
**Deliverables:**
- [ ] Gradient fingerprint extraction module
- [ ] k-Means clustering implementation  
- [ ] Task group formation and validation
- [ ] Visualization and quality metrics

**Key outputs:**
- Clustering code with unit tests
- Cluster visualization (t-SNE plots)
- Silhouette score analysis

### Week 2-3: Heterogeneous Configuration (Phase 2)
**Deliverables:**
- [ ] Device profiler (memory, compute)
- [ ] Weight importance scoring
- [ ] Dynamic rank allocation algorithm
- [ ] Configuration optimizer

**Key outputs:**
- Rank allocation code
- Memory profile analysis
- Rank assignment visualization

### Week 3-6: Split Federated Learning (Phase 3)
**Deliverables:**
- [ ] Client-side training with LoRA
- [ ] Server-side model components
- [ ] Communication protocol implementation
- [ ] Integration with GPT-2 and LLaMA

**Key outputs:**
- End-to-end training pipeline
- Convergence plots on GLUE tasks
- Communication cost analysis

### Week 6-8: Aggregation Engine (Phase 4)
**Deliverables:**
- [ ] Noise-free aggregation algorithm
- [ ] Task-aware weight computation
- [ ] Aggregation quality validation
- [ ] Privacy mechanism integration

**Key outputs:**
- Aggregation code with tests
- Aggregation quality metrics
- Privacy integration points

### Week 8-10: Privacy Evaluation (Phase 5)
**Deliverables:**
- [ ] Privacy attack implementations (5 types)
- [ ] Defense mechanisms (9 variants)
- [ ] Privacy metrics dashboard
- [ ] Final privacy report

**Key outputs:**
- Privacy attack/defense code
- DCS scores for each attack
- Comprehensive privacy report

### Week 10-12: Evaluation & Demo (Phase 6)
**Deliverables:**
- [ ] Benchmark on GLUE/SQuAD/E2E
- [ ] Heterogeneous device simulation
- [ ] Baseline comparison (Standard FL vs ATLAS)
- [ ] Live demonstration system

**Key outputs:**
- Performance comparison tables
- Interactive dashboard
- Demo-ready checkpoints
- Final presentation

---

## 5. EXPERIMENTAL SETUP

### Datasets
| Dataset | Task | Split | Size |
|---------|------|-------|------|
| GLUE | Multi-task NLP | RTE, SST-2, CoLA, STS-B | ~370K |
| SQuAD | Reading comprehension | Train/Dev | ~100K |
| E2E | Data-to-text generation | Train/Dev | ~50K |

### Models
- **GPT-2 (124M):** For rapid prototyping and debugging
- **LLaMA-7B (7B):** For production demo and benchmarking

### Simulated Clients
| Type | Count | Memory | Compute | LoRA Ranks |
|------|-------|--------|---------|-----------|
| Low-end (CPU) | 2 | 2GB | 1x | 4-8 |
| Medium (Edge GPU) | 3 | 4GB | 5x | 8-16 |
| High-end (GPU) | 5 | 8GB+ | 10x | 16-32 |

---

## 6. KEY EVALUATION METRICS

### Accuracy Metrics
- Task-specific F1, Accuracy, Pearson correlation
- Convergence speed (rounds to 95% of target)
- Final accuracy on held-out test set

### Efficiency Metrics
- Memory usage per device type
- Communication cost (MB per training round)
- Training time (wall-clock hours)
- Inference latency on edge devices

### Privacy Metrics
- **DCS (Detection-based Capability Score):** 0.0 to 1.0 (higher = more private)
- **Differential Privacy:** $(ε, δ)$ bounds
- **Attack success rates:** For each of 5 attack types
- **Defense effectiveness:** Across 9 defense mechanisms

### Performance Targets
| Metric | Target | Baseline |
|--------|--------|----------|
| Accuracy | ≥95% of centralized | Standard FL |
| Accuracy boost | +15-25% | Homogeneous LoRA |
| Memory reduction | 30-40% | Standard FL |
| Comm overhead | <5% | Centralized |
| Privacy (DCS) | ≥0.7 | Undefended |

---

## 7. TECHNOLOGY STACK

### Deep Learning
- **PyTorch 2.0+:** Core framework
- **Transformers (HuggingFace):** Pre-trained models
- **PEFT:** LoRA implementations
- **Flash Attention:** Optimization

### Machine Learning
- **scikit-learn:** Clustering (k-Means)
- **numpy/scipy:** Numerical computing
- **pandas:** Data manipulation

### Federated Learning
- **Ray:** Distributed simulation
- **PySyft:** Federated learning primitives
- **TensorBoard:** Experiment tracking

### Privacy
- **VFLAIR-LLM:** Privacy attacks/defenses (open-source)
- **opacus:** Differential privacy
- **CrypTen:** Secure MPC

### Development
- **Python 3.10+**
- **Git/GitHub:** Version control
- **Weights & Biases:** Experiment tracking
- **Jupyter Notebooks:** Interactive analysis

---

## 8. EXPECTED OUTCOMES

### Research Contributions
1. **Novel architecture:** MIRA + HSpLitLoRA + SplitLoRA combination
2. **Practical system:** End-to-end federated learning platform
3. **Comprehensive evaluation:** Privacy + accuracy + efficiency benchmarks
4. **Open-source release:** Code and trained models

### Performance Predictions
- **Accuracy:** +15-25% vs homogeneous LoRA baseline
- **Memory:** 30-40% reduction on constrained devices
- **Communication:** <5% overhead vs centralized training
- **Privacy:** DCS ≥0.7 against standard attacks
- **Inference:** <100ms latency on edge GPU

### Deliverables
- ✅ Working ATLAS implementation (5 phases)
- ✅ Benchmark datasets and baselines
- ✅ Privacy evaluation report
- ✅ Live demonstration system
- ✅ Technical documentation and code
- ✅ Research paper (optional, after implementation)

---

## 9. RISKS & MITIGATION

| Risk | Probability | Mitigation |
|------|-------------|-----------|
| Clustering fails to identify tasks | Medium | Use gradient fingerprints + weighted Laplacian |
| LoRA rank allocation suboptimal | Medium | Implement adaptive rank search at runtime |
| Privacy attacks succeed | Medium | Add differential privacy + secure aggregation |
| Communication bottleneck | Low | Split learning reduces size by 10-100x |
| Model convergence slow | Medium | Implement adaptive learning rate scheduling |
| Device simulation inaccurate | Low | Validate against real hardware when possible |
| Time runs out | Low | Prioritize core features (Phases 1-4) over Phase 5 |

### Contingency Plans
- **If clustering fails:** Use predefined task groups + manual annotations
- **If privacy weak:** Add differential privacy as separate layer
- **If memory constrained:** Reduce model size or use sparse LoRA
- **If time running low:** Demo on small model (GPT-2) instead of LLaMA

---

## 10. SUPERVISOR PRESENTATION TALKING POINTS

### Why This Approach?
1. **Addresses real problem:** Federated learning for heterogeneous multi-task settings
2. **Novel combination:** First to combine MIRA + HSpLitLoRA + SplitLoRA
3. **Implementable:** Based on published papers, no proprietary data needed
4. **Timely:** Fits within 2-month deadline with clear milestones
5. **Practical impact:** Applicable to real-world edge AI scenarios

### Key Innovation
- **MIRA clusters tasks** → better aggregation weights
- **HSpLitLoRA assigns heterogeneous ranks** → saves 30-40% memory
- **SplitLoRA reduces communication** → 10-100x bandwidth savings
- **Privacy evaluation** → demonstrates security guarantees

### Feasibility
- ✅ All methodologies are published
- ✅ Open-source libraries available (PyTorch, Transformers, PEFT)
- ✅ Clear implementation roadmap (6 phases over 12 weeks)
- ✅ Checkpoints and milestones defined
- ✅ Achievable with full-time dedication

### Expected Impact
- **Accuracy:** +15-25% vs baseline
- **Efficiency:** 30-40% memory reduction + 10-100x communication savings
- **Privacy:** DCS ≥0.7, resistant to standard attacks
- **Demo:** Working system with live results

---

## 11. FILE STRUCTURE

```
ATLAS/
├── ATLAS_Revised_Plan_Presentation.tex    # LaTeX presentation (THIS FILE)
├── ATLAS_Base_Specification.pdf           # Original requirements
├── ATLAS_V1.pdf                           # Previous plan
├── Research Papers/
│   ├── MIRA_...pdf                        # Task clustering
│   ├── HSplitLoRA.pdf                     # Heterogeneous splitting
│   ├── SplitLoRA.pdf                      # Communication efficiency
│   ├── Privacy-Aware_SFL.pdf             # Privacy mechanisms
│   └── VFLAIR-LLM.pdf                    # Privacy evaluation
└── Implementation/                         # To be created
    ├── phase1_clustering.py
    ├── phase2_configuration.py
    ├── phase3_training.py
    ├── phase4_aggregation.py
    ├── phase5_privacy.py
    └── utils/
        ├── datasets.py
        ├── models.py
        └── metrics.py
```

---

## 12. NEXT IMMEDIATE STEPS

### For Supervisor Approval (Day 1)
- [ ] Review this document
- [ ] Review LaTeX presentation
- [ ] Discuss feasibility and timeline
- [ ] Get approval on technical approach

### For Project Setup (Week 1)
- [ ] Create GitHub repository with project structure
- [ ] Set up development environment (PyTorch, Transformers)
- [ ] Download and process datasets (GLUE, SQuAD, E2E)
- [ ] Implement Phase 1 (Task Clustering)

### For Checkpoints
- **Week 2:** Task clustering working
- **Week 3:** Heterogeneous rank allocation working
- **Week 6:** End-to-end training functional
- **Week 8:** Privacy evaluation integrated
- **Week 10:** All benchmarks complete
- **Week 12:** Demo ready

---

## 13. CONCLUSION

The revised ATLAS project represents a **practical and innovative approach** to federated learning for heterogeneous multi-task LLM fine-tuning. By combining MIRA, HSpLitLoRA, and SplitLoRA methodologies, we create a system that is:

- **Novel:** First combination of these three approaches
- **Feasible:** Implementable from published papers within 2 months
- **Impactful:** Addresses real challenges in edge AI and federated learning
- **Evaluable:** Comprehensive privacy, accuracy, and efficiency benchmarks

**Status:** Ready for implementation

**Timeline:** 12 weeks with clear milestones

**Expected Demo:** Week 12 (end of 2 months)

---

## 14. QUICK REFERENCE

### Key Equations

**Task Clustering (MIRA):**
$$\mathcal{L}_{\text{task}} = \sum_{i} \|w_i\|^2 + \lambda \mathbf{w}^T L \mathbf{w}$$

**Heterogeneous LoRA:**
$$h' = h + \mathbf{A}_i \mathbf{B}_i^T x, \quad \mathbf{A}_i \in \mathbb{R}^{d \times r_i}, \mathbf{B}_i \in \mathbb{R}^{d \times r_i}$$

**Task-Aware Aggregation:**
$$w_{\text{merged}} = \sum_{g=1}^{G} \alpha_g \cdot \text{merge}(\{\mathbf{A}_i\}_{i \in \text{group}_g})$$

**Communication Efficiency:**
$$\text{Cost}_{\text{SplitLoRA}} = \mathcal{O}(r + h) \ll \mathcal{O}(d) = \text{Cost}_{\text{Standard FL}}$$

### Key Metrics

| Metric | Target | Method |
|--------|--------|--------|
| Accuracy | ≥95% of centralized | GLUE F1 scores |
| Memory | -30-40% | Peak device memory |
| Communication | <5% overhead | Bytes per round |
| Privacy (DCS) | ≥0.7 | VFLAIR benchmark |
| Convergence | 2-3x centralized | Rounds to target accuracy |

### Team Roles (Solo Project)
- Researcher: Literature review, methodology design
- Developer: Implementation of all 5 phases
- Experimenter: Running benchmarks and evaluation
- Reporter: Documentation and presentation

---

**Document Version:** 1.0 (Final)  
**Created:** 2025-12-23  
**Status:** Ready for Supervisor Review  
**Estimated Implementation Time:** 12 weeks
