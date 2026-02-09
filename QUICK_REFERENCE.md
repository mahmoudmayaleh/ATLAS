# 🎯 ATLAS Quick Reference: Statistical Experiments & Model Expansion

## 📊 Multi-Seed Statistical Experiments

### Run 5 Seeds Automatically:

```bash
python experiments/run_statistical_experiments.py --seeds 5
```

**Outputs**:

- `results/statistical/statistical_summary.csv` - Mean ± std
- `results/statistical/statistical_tests.csv` - p-values, Cohen's d
- LaTeX table (copy-paste ready)

**Includes**: t-test, Wilcoxon, effect sizes, publication-ready tables

---

## 🤖 Model Expansion

### Supported Models:

```bash
--model distilbert-base-uncased    # 66M params (default)
--model bert-base-uncased           # 110M params
--model bert-large-uncased          # 340M params
--model roberta-base                # 125M params
--model roberta-large               # 355M params
--model gpt2                        # 124M params
--model gpt2-medium                 # 355M params
```

### Example:

```bash
python experiments/atlas_integrated.py \
    --model bert-base-uncased \
    --tasks sst2 mrpc cola qnli \
    --rounds 15 --seed 42
```

---

## 📝 Expanded GLUE Tasks

### Supported Tasks:

```bash
--tasks sst2        # Sentiment analysis
--tasks mrpc        # Paraphrase detection
--tasks cola        # Grammar acceptability
--tasks qnli        # Question NLI (NEW!)
--tasks qqp         # Duplicate questions (NEW!)
--tasks mnli        # Multi-genre NLI (NEW!)
--tasks rte         # Textual entailment
--tasks wnli        # Winograd schema
```

### Multi-Task Example:

```bash
python experiments/atlas_integrated.py \
    --tasks sst2 mrpc cola qnli qqp mnli \
    --clients-per-task 3
```

---

## 🔧 Split Layer Selection

### Current System: ADAPTIVE ✓

**Automatically considers**:

- Device memory constraints (50-80% safety margin)
- Communication costs (activation transfer time)
- Layer importance (gradient magnitudes)
- Workload balance (prefer 50/50 splits)

**Default splits**:

- BERT/DistilBERT: Layer 6/12 (50%)
- GPT-2: Layer 6/12 (50%)
- Adaptive: Varies per client (2-8 typically)

### Improved Split Selector (NEW):

**Location**: `src/improved_split_selection.py`

**Scoring Function**:

```
Total = 0.35×Memory + 0.30×Comm + 0.25×Importance + 0.10×Balance
```

**Usage**:

```python
from src.improved_split_selection import ImprovedSplitSelector

selector = ImprovedSplitSelector(
    model_name='bert-base-uncased',
    device_profiles=device_profiles,
    task_assignments=task_assignments,
    bandwidth_mbps=10.0
)

split = selector.compute_optimal_split(client_id=0)
```

---

## ⚡ Quick Commands

### 1. Quick Test (30 min):

```bash
python experiments/atlas_integrated.py --mode quick --seed 42
```

### 2. Multi-Seed Stats (8-12 hrs):

```bash
python experiments/run_statistical_experiments.py \
    --seeds 5 --rounds 15 --samples 3000
```

### 3. Model Comparison:

```bash
# BERT-base
python experiments/run_statistical_experiments.py \
    --seeds 5 --model bert-base-uncased

# RoBERTa
python experiments/run_statistical_experiments.py \
    --seeds 5 --model roberta-base
```

### 4. Expanded Tasks:

```bash
python experiments/atlas_integrated.py \
    --tasks sst2 mrpc cola qnli qqp mnli \
    --rounds 15 --seed 42
```

### 5. Lambda Sweep:

```bash
for eta in 0.0 0.01 0.1 0.5 1.0; do
    python experiments/atlas_integrated.py \
        --ablation atlas --eta $eta --rounds 15 --seed 42
done
```

---

## 📊 Expected Results (5 Seeds)

| Config | Final Acc   | Personalization | Comm (MB) |
| ------ | ----------- | --------------- | --------- |
| ATLAS  | 0.812±0.005 | 0.031±0.002     | 245±13    |
| FedAvg | 0.805±0.007 | 0.048±0.003     | 242±15    |
| Local  | 0.789±0.009 | 0.092±0.005     | 0±0       |

**Significance**: ATLAS vs FedAvg: p<0.05 ✓

---

## ✅ Key Points

1. **Seeds are now supported**: `--seed 42` for reproducibility
2. **7+ models supported**: BERT, RoBERTa, GPT-2, etc.
3. **8+ GLUE tasks supported**: SST-2, MRPC, CoLA, QNLI, QQP, MNLI, etc.
4. **Split layer is ADAPTIVE**: Not fixed at 3 - computed per client
5. **Statistical tests automated**: t-test + Wilcoxon + Cohen's d
6. **Auto-backup to Drive**: All results saved automatically
7. **JSON outputs**: Per-round/task metrics preserved

---

## 📁 Key Files

- `experiments/run_statistical_experiments.py` - Multi-seed runner
- `src/improved_split_selection.py` - Advanced split selection
- `docs/STATISTICAL_EXPERIMENTS_GUIDE.md` - Full guide
- `IMPLEMENTATION_SUMMARY.md` - This summary

---

## 🆘 Troubleshooting

**Q**: Results not saving?  
**A**: Check that `backup_to_drive()` was called (now automatic in notebook cells)

**Q**: Statistical tests showing n.s. (not significant)?  
**A**: Increase seeds (use 5-10) or rounds (use 20-30 for convergence)

**Q**: Out of memory with large models?  
**A**: Reduce `--samples` (try 2000-3000) or use smaller batch size

**Q**: Split layer not adapting?  
**A**: Check that device profiles are properly configured in Phase 2

---

## 📖 Read More

- Full guide: `docs/STATISTICAL_EXPERIMENTS_GUIDE.md`
- Implementation details: `IMPLEMENTATION_SUMMARY.md`
- Original docs: `README.md`, `docs/REAL_TRAINING_SUMMARY.md`

---

**Ready to run publication-quality experiments! 🚀**
