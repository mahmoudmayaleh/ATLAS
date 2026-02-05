# Multi-Domain Task Configuration for ATLAS

## Publication-Quality IEEE Experiments

### Overview

For a truly **multi-task** federated learning system suitable for IEEE publication, ATLAS should handle diverse task types beyond NLP. This document describes how to configure multi-domain experiments.

---

## Supported Task Domains

### 1. Natural Language Processing (NLP)

**Available Tasks**:

- `sst2` - Stanford Sentiment Treebank (binary sentiment)
- `mrpc` - Microsoft Research Paraphrase Corpus (paraphrase detection)
- `cola` - Corpus of Linguistic Acceptability (grammar)
- `qnli` - Question Natural Language Inference
- `mnli` - Multi-Genre NLI (3-way classification)
- `wnli` - Winograd Schema Challenge
- `rte` - Recognizing Textual Entailment

**Example**:

```bash
python experiments/atlas_integrated.py \
    --tasks sst2 mrpc cola qnli \
    --clients-per-task 3 \
    --samples 5000
```

### 2. Computer Vision (Coming Soon)

**Planned Tasks**:

- `cifar10` - Image classification (10 classes)
- `mnist` - Digit recognition
- `fashion_mnist` - Fashion item classification
- `svhn` - Street View House Numbers

**Integration**:

- Use ResNet-18/MobileNet backbone
- LoRA adapters on attention/linear layers
- Same clustering/aggregation pipeline

### 3. Speech/Audio (Coming Soon)

**Planned Tasks**:

- `speech_commands` - Keyword spotting
- `librispeech` - Speech recognition
- `common_voice` - Multi-speaker ASR

**Integration**:

- Wav2Vec2 backbone
- LoRA on transformer layers
- Acoustic fingerprinting for clustering

---

## Current Implementation: NLP Only

### Why Start with NLP?

1. **Mature Infrastructure**: HuggingFace Transformers ecosystem
2. **Established Benchmarks**: GLUE tasks widely used
3. **LoRA Support**: Well-documented for language models
4. **Validation**: Easy to verify with known baselines

### Expanding to Multi-Domain

**Phase 1 works domain-agnostic**:

- Gradient fingerprinting works for any model
- Clustering based on gradient patterns, not data type
- No modification needed

**Phase 2 needs model-specific**:

- Memory formulas same: `2*d*r*b`
- Layer identification needs adaptation (e.g., ConvNets)
- Device constraints remain constant

**Phase 3 requires modality adapters**:

- Image inputs → different tokenization
- Audio inputs → spectrograms or waveforms
- Output heads different per task type

---

## Configuration Examples

### Multi-NLP Tasks (Current)

```python
# 4 diverse NLP tasks
python experiments/atlas_integrated.py \
    --mode full \
    --rounds 30 \
    --model bert-base-uncased \
    --tasks sst2 mrpc cola qnli \
    --clients-per-task 3 \
    --samples 5000
```

**Results**:

- 12 clients (4 tasks × 3 clients each)
- 4 task clusters expected
- Different task heads per cluster
- Validates task-aware aggregation

### Cross-Modal (Future)

```python
# Mixed NLP + Vision + Audio
python experiments/atlas_integrated.py \
    --mode full \
    --rounds 30 \
    --tasks sst2 cifar10 speech_commands \
    --models bert resnet18 wav2vec \
    --clients-per-task 3
```

**Challenges**:

- Different model architectures per task
- Different input preprocessing
- Common split point selection difficult

**Solution**: Use modality-specific adapters with shared orchestration

---

## Model-Task Compatibility Matrix

| Model      | SST-2 | MRPC | CoLA | QNLI | CIFAR-10 | MNIST | Speech |
| ---------- | ----- | ---- | ---- | ---- | -------- | ----- | ------ |
| DistilBERT | ✓     | ✓    | ✓    | ✓    | ✗        | ✗     | ✗      |
| BERT-base  | ✓     | ✓    | ✓    | ✓    | ✗        | ✗     | ✗      |
| RoBERTa    | ✓     | ✓    | ✓    | ✓    | ✗        | ✗     | ✗      |
| GPT-2      | ✓     | ✓    | ✓    | ✓    | ✗        | ✗     | ✗      |
| ResNet-18  | ✗     | ✗    | ✗    | ✗    | ✓        | ✓     | ✗      |
| MobileNet  | ✗     | ✗    | ✗    | ✗    | ✓        | ✓     | ✗      |
| Wav2Vec2   | ✗     | ✗    | ✗    | ✗    | ✗        | ✗     | ✓      |

---

## Task Heterogeneity Levels

### Level 1: Single Domain, Similar Tasks

```bash
--tasks sst2 mrpc  # Both sentence-pair tasks
```

- **Clustering**: May group together (similar gradients)
- **Benefit**: Mild - tasks already compatible
- **Publication value**: Medium (shows system works but not impressive)

### Level 2: Single Domain, Diverse Tasks (Current)

```bash
--tasks sst2 cola qnli mnli  # Sentiment + grammar + QA + NLI
```

- **Clustering**: 2-4 distinct clusters expected
- **Benefit**: Significant - prevents harmful task mixing
- **Publication value**: High (demonstrates task-aware FL)

### Level 3: Cross-Domain, Same Modality

```bash
--tasks sst2 medqa legalqa codeqa  # General + medical + legal + code
```

- **Clustering**: 4 distinct clusters highly likely
- **Benefit**: Maximum - domain-specific knowledge preserved
- **Publication value**: Very High (real-world heterogeneity)

### Level 4: Cross-Modal (Future Work)

```bash
--tasks sst2 cifar10 speech_commands  # Text + vision + audio
```

- **Clustering**: 3 clusters guaranteed (different modalities)
- **Benefit**: Research contribution - novel multi-modal FL
- **Publication value**: Excellent (IEEE best paper candidate)

---

## Recommended Configurations for IEEE Publication

### Configuration A: Diverse NLP (Recommended for First Paper)

```bash
python experiments/atlas_integrated.py \
    --mode full \
    --rounds 30 \
    --model bert-base-uncased \
    --tasks sst2 mrpc cola qnli \
    --clients-per-task 3 \
    --samples 5000 \
    --local-epochs 3
```

**Rationale**:

- 4 semantically different tasks
- Well-established GLUE benchmarks
- Easy to compare with FedAvg/FedProx baselines
- Strong clustering validation

### Configuration B: Domain-Specific NLP (Advanced)

```bash
python experiments/atlas_integrated.py \
    --mode full \
    --rounds 30 \
    --model bert-base-uncased \
    --tasks sst2 medical_ner legal_classification \
    --clients-per-task 4 \
    --samples 5000
```

**Rationale**:

- Demonstrates real-world heterogeneity
- Stronger motivation for task clustering
- Higher publication impact

### Configuration C: Large-Scale

```bash
python experiments/atlas_integrated.py \
    --mode full \
    --rounds 50 \
    --model bert-base-uncased \
    --tasks sst2 mrpc cola qnli mnli rte wnli \
    --clients-per-task 5 \
    --samples 5000
```

**Rationale**:

- 7 tasks × 5 clients = 35 clients (scalability)
- Comprehensive GLUE coverage
- Requires multiple Colab sessions (10+ hours)
- Maximum IEEE impact

---

## Adding New Tasks

### Step 1: Extend Dataset Map

Edit `experiments/atlas_integrated.py`:

```python
self.dataset_map = {
    # Existing NLP tasks
    'sst2': ('stanfordnlp/sst2', 'sentence', None, 2),
    'mrpc': ('nyu-mll/glue', 'sentence1', 'sentence2', 2),

    # NEW: Add your tasks here
    'imdb': ('imdb', 'text', None, 2),  # Movie reviews
    'ag_news': ('ag_news', 'text', None, 4),  # News classification
    'yelp': ('yelp_polarity', 'text', None, 2),  # Restaurant reviews
}
```

### Step 2: Verify Tokenization

```python
def _load_task_data(self, task_name: str):
    dataset_name, text_col, text_col2, num_labels = self.dataset_map[task_name]

    # Add loading logic
    if task_name == 'imdb':
        dataset = load_dataset('imdb', split='train')
        test_dataset = load_dataset('imdb', split='test')
    # ... rest of tokenization
```

### Step 3: Test

```bash
python experiments/atlas_integrated.py \
    --mode quick \
    --tasks sst2 imdb \
    --rounds 5
```

---

## Multi-Domain Roadmap

### Phase 1: NLP Only (Current)

- GLUE tasks
- Sentiment, paraphrase, grammar, QA
- 4-7 tasks supported

### Phase 2: NLP + Vision (Q2 2026)

- Add CIFAR-10, MNIST
- ResNet/MobileNet backbone
- LoRA for ConvNets
- Separate model instances per modality

### Phase 3: Full Multi-Modal (Q3 2026)

- Add speech tasks
- Unified gradient fingerprinting
- Cross-modal clustering (should separate naturally)
- True heterogeneous multi-task FL

---

## Citation for Multi-Task Datasets

If using these datasets in your IEEE paper:

**GLUE**:

```
@inproceedings{wang2018glue,
  title={GLUE: A multi-task benchmark and analysis platform for natural language understanding},
  author={Wang, Alex and others},
  booktitle={ICLR},
  year={2019}
}
```

**SST-2**:

```
@inproceedings{socher2013sst,
  title={Recursive deep models for semantic compositionality over a sentiment treebank},
  author={Socher, Richard and others},
  booktitle={EMNLP},
  year={2013}
}
```

---

## Summary

**For IEEE Publication Now**:

- Use 4 diverse NLP tasks: `sst2 mrpc cola qnli`
- 30 rounds, 5000 samples, 3 local epochs
- Demonstrate task-aware clustering and aggregation
- Compare against FedAvg, FedProx, Local Only

**Future Extensions**:

- Add domain-specific tasks (medical, legal, code)
- Implement vision tasks for true multi-domain
- Scale to 50+ clients across 10+ tasks

**Key Selling Points**:

- Multi-task (4+ different NLP tasks)
- Heterogeneous devices (2GB CPU to 16GB GPU)
- Task-aware (prevents harmful mixing)
- Personalized (client-specific models)
- Efficient (LoRA + split learning)

This configuration is **publication-ready for IEEE** and demonstrates all ATLAS innovations!
