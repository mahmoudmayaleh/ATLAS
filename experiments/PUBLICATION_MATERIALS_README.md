# ATLAS Publication Materials Generator

This directory contains scripts to generate IEEE-quality figures and tables for the ATLAS research paper.

## 📋 Overview

The scripts generate:
- **6 publication-ready figures** (PDF + PNG format)
- **6 results tables** (CSV + LaTeX format)

All outputs follow IEEE publication standards with proper formatting, fonts, and styling.

## 🚀 Quick Start

### Generate Everything

```bash
cd experiments
python generate_all_publication_materials.py
```

This will create:
- Figures in `../figures/`
- Tables in `../results/`

### Custom Directories

```bash
python generate_all_publication_materials.py \
    --results-dir path/to/results \
    --figures-dir path/to/figures \
    --tables-dir path/to/tables
```

### Generate Only Plots

```bash
python generate_publication_plots.py
```

### Generate Only Tables

```bash
python generate_results_tables.py
```

## 📊 Generated Figures

### Figure 1: Ablation Study (`fig1_ablation_study.pdf`)
**Purpose**: Demonstrates the impact of each ATLAS component on performance.

**Subplots**:
- (a) Accuracy convergence comparing ATLAS variants
- (b) F1 score convergence

**Shows**:
- ATLAS (Full) - complete system
- ATLAS (No Laplacian) - without graph regularization
- FedAvg (Clustered) - without split training
- Standard FL - baseline without clustering

**Usage in Paper**: Introduction section or main results to show component contributions.

---

### Figure 2: Model Comparison (`fig2_model_comparison.pdf`)
**Purpose**: Shows ATLAS generalizes across different model architectures.

**Subplots**:
- (a) Performance comparison across DistilBERT, GPT-2, and Qwen-0.5B
- (b) ATLAS vs. Standard FL for each model

**Shows**:
- Cross-architecture generalization
- Consistent improvements over baseline
- Statistical variance (shaded regions)

**Usage in Paper**: Experimental results section to demonstrate robustness.

---

### Figure 3: Communication Efficiency (`fig3_communication_efficiency.pdf`)
**Purpose**: Analyzes communication costs and efficiency gains.

**Subplots**:
- (a) Cumulative communication overhead over rounds
- (b) Communication efficiency (accuracy per MB)
- (c) Total data transferred

**Shows**:
- ATLAS reduces communication costs
- Better efficiency compared to baselines
- Trade-offs between accuracy and communication

**Usage in Paper**: Performance analysis or communication efficiency section.

---

### Figure 4: Clustering Analysis (`fig4_clustering_analysis.pdf`)
**Purpose**: Demonstrates how ATLAS handles client heterogeneity.

**Subplots**:
- (a) Per-client accuracy evolution over time
- (b) Final accuracy distribution (box plots)

**Shows**:
- Individual client trajectories
- Variance reduction through clustering
- Statistical comparison (mean, std. dev.)

**Usage in Paper**: Methodology or heterogeneity handling section.

---

### Figure 5: Eta Parameter Study (`fig5_eta_parameter_study.pdf`)
**Purpose**: Sensitivity analysis of the eta (η) parameter.

**Subplots**:
- (a) Convergence for different η values (0.0, 0.1, 0.5)
- (b) Final performance comparison

**Shows**:
- Impact of adaptation rate
- Optimal parameter selection
- Trade-offs in adaptation

**Usage in Paper**: Parameter analysis or ablation section.

---

### Figure 6: Convergence Speed (`fig6_convergence_speed.pdf`)
**Purpose**: Analyzes training efficiency and convergence speed.

**Subplots**:
- (a) Accuracy vs. cumulative training time
- (b) Average time per communication round

**Shows**:
- Wall-clock time efficiency
- Faster convergence compared to baselines
- Computational overhead analysis

**Usage in Paper**: Computational efficiency section.

---

## 📋 Generated Tables

### Table I: Main Results (`table1_main_results`)
**Content**: Performance comparison of all ATLAS variants on DistilBERT

**Columns**:
- Method name
- Average accuracy ± std. dev.
- Average F1 score ± std. dev.
- Min/max accuracy (client heterogeneity)
- Communication cost (MB)
- Training time (minutes)

**Usage**: Main results section, core performance comparison.

---

### Table II: Cross-Model Performance (`table2_cross_model`)
**Content**: ATLAS vs. Standard FL across different model architectures

**Columns**:
- Model architecture
- Method (ATLAS/Standard FL)
- Accuracy and F1 scores
- Communication and time costs

**Usage**: Generalization analysis, showing robustness across models.

---

### Table III: Ablation Study (`table3_ablation`)
**Content**: Detailed ablation showing contribution of each component

**Columns**:
- Configuration
- Description
- Performance metrics
- Delta from full system
- Communication efficiency

**Usage**: Ablation study section, component analysis.

---

### Table IV: Communication Efficiency (`table4_communication`)
**Content**: Detailed breakdown of communication costs

**Columns**:
- Total communication (MB)
- Per-round average
- Upload/download split
- Accuracy per MB
- Accuracy per minute

**Usage**: Communication efficiency analysis.

---

### Table V: Eta Sensitivity (`table5_eta_sensitivity`)
**Content**: Impact of eta parameter on performance

**Columns**:
- Eta value
- Description
- Accuracy and F1 scores
- Number of rounds
- Training time

**Usage**: Parameter sensitivity analysis.

---

### Table VI: Statistical Summary (`table6_statistical_summary`)
**Content**: High-level statistical overview across all experiments

**Columns**:
- Category (ATLAS methods, Standard FL)
- Number of experiments
- Aggregated statistics
- Average performance metrics

**Usage**: Overview section or appendix, showing overall trends.

---

## 🎨 Formatting Details

### Plot Styling
- **Font**: Times New Roman (IEEE standard)
- **Font size**: 10pt body, 11pt labels, 12pt titles
- **Colors**: IEEE-friendly color palette (colorblind-safe)
- **Line styles**: Different styles for each method (solid, dashed, dash-dot, dotted)
- **Markers**: Distinct markers for easy identification
- **Resolution**: 300 DPI for publication quality
- **Formats**: Both PDF (vector) and PNG (raster)

### Table Formatting
- **Style**: IEEE double-ruled tables
- **Formats**: Both LaTeX (.tex) and CSV (.csv)
- **Precision**: 2 decimal places for percentages, 1 for MB/time
- **Statistical notation**: Mean ± std. dev. where applicable
- **LaTeX ready**: Direct copy-paste into paper

## 📦 Dependencies

```bash
pip install numpy pandas matplotlib seaborn
```

Or use existing requirements:
```bash
pip install -r ../requirements.txt
```

## 🔧 Customization

### Modify Colors

Edit `COLORS` dictionary in `generate_publication_plots.py`:
```python
COLORS = {
    'atlas': '#1f77b4',        # Change to your preference
    'no_laplacian': '#ff7f0e',
    # ...
}
```

### Modify Plot Layout

Change figure size:
```python
fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))  # Width, height in inches
```

### Add New Plots

1. Create new method in `PublicationPlotter` class
2. Call it from `generate_all_plots()`
3. Follow existing patterns for consistency

### Modify Table Format

Edit `_generate_latex_table()` method for different LaTeX styling:
```python
column_format="l|cc|cc"  # Adjust column alignment and separators
```

## 📁 File Structure

```
experiments/
├── generate_all_publication_materials.py  # Master script
├── generate_publication_plots.py          # Plot generation
├── generate_results_tables.py             # Table generation
└── PUBLICATION_MATERIALS_README.md        # This file

Generated outputs:
├── figures/
│   ├── fig1_ablation_study.pdf
│   ├── fig1_ablation_study.png
│   ├── fig2_model_comparison.pdf
│   ├── fig2_model_comparison.png
│   └── ... (12 files total)
│
└── results/
    ├── table1_main_results.tex
    ├── table1_main_results.csv
    ├── table2_cross_model.tex
    ├── table2_cross_model.csv
    └── ... (12 files total)
```

## 🎯 Usage in LaTeX Paper

### Including Figures

```latex
\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figures/fig1_ablation_study.pdf}
\caption{Ablation study showing the impact of ATLAS components.}
\label{fig:ablation}
\end{figure}
```

### Including Tables

Simply copy the contents of the `.tex` files into your paper:

```latex
% Copy content from table1_main_results.tex
\begin{table}[!t]
\renewcommand{\arraystretch}{1.2}
...
\end{table}
```

### Two-Column Figures

For spanning both columns:
```latex
\begin{figure*}[!t]
\centering
\includegraphics[width=\textwidth]{figures/fig2_model_comparison.pdf}
\caption{Cross-model performance comparison.}
\label{fig:models}
\end{figure*}
```

## 📊 Expected Results Pattern

Your results files should follow the pattern:
```
atlas_{model}_{method}_seed{seed}_r{rounds}.json
```

Examples:
- `atlas_distilbert-base-uncased_atlas_seed42_r10.json`
- `atlas_gpt2_standard_fl_seed123_r10.json`
- `atlas_Qwen_Qwen2.5-0.5B_atlas_seed42_r10.json`

Special files:
- `atlas_integrated_full_atlas_00_eta_seed42.json` (eta=0.0)
- `atlas_integrated_full_atlas_01_eta_seed42.json` (eta=0.1)
- `atlas_integrated_full_atlas_05_eta_seed42.json` (eta=0.5)

## 🐛 Troubleshooting

### "Warning: file not found"
- Check that result files exist in the specified directory
- Verify file naming matches expected patterns
- Use `--results-dir` flag to specify correct path

### Font warnings
- Install Times New Roman font if needed
- Alternatively, modify `rcParams['font.family']` in the scripts

### Missing data points
- Ensure all experiments completed successfully
- Check JSON files for complete round_metrics data
- Verify all required fields exist in JSON structure

### Plot looks wrong
- Check that result file has expected structure
- Verify number of clients matches expectations
- Review console output for specific warnings

## 💡 Tips for Best Results

1. **Run all seeds**: For statistical significance, use multiple random seeds
2. **Consistent rounds**: Use same number of rounds for fair comparison
3. **Check quality**: Review PDFs at 100% zoom before submitting
4. **LaTeX compilation**: Test table LaTeX in your paper's preamble
5. **Color printing**: Verify colors are distinguishable in grayscale
6. **File size**: PDFs are vector format (small), PNGs for presentations

## 📚 Citations

When using these materials, cite the ATLAS paper:

```bibtex
@article{atlas2026,
  title={ATLAS: Adaptive Task-specific Learning through Adaptive Split-training},
  author={Your Name et al.},
  journal={IEEE Conference/Journal},
  year={2026}
}
```

## 🤝 Support

For issues or questions:
1. Check this README
2. Review script comments and docstrings
3. Examine example outputs
4. Verify input data format

## ✅ Checklist for Paper Submission

- [ ] Generated all 6 figures (PDF format)
- [ ] Generated all 6 tables (LaTeX format)
- [ ] Reviewed all figures for clarity
- [ ] Tested LaTeX tables in paper
- [ ] Verified all data is correct
- [ ] Checked figure resolution (300 DPI)
- [ ] Ensured colorblind-friendly colors
- [ ] Added proper captions and labels
- [ ] Cross-referenced in text
- [ ] Included in LaTeX document
- [ ] Final PDF compilation successful

---

**Last Updated**: February 2026  
**Script Version**: 1.0  
**Compatible with**: ATLAS v1.0+
