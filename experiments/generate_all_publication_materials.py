"""
Master Script for Generating All Publication Materials
Runs all plotting and table generation scripts for the ATLAS paper.
"""

import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))

from generate_publication_plots import PublicationPlotter
from generate_results_tables import ResultsTableGenerator


def main():
    """Generate all publication materials."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate all IEEE-quality figures and tables for ATLAS publication'
    )
    parser.add_argument('--results-dir', type=str, default='../results',
                       help='Directory containing results JSON files')
    parser.add_argument('--figures-dir', type=str, default='../figures',
                       help='Directory to save output figures')
    parser.add_argument('--tables-dir', type=str, default='../results',
                       help='Directory to save output tables')
    parser.add_argument('--skip-plots', action='store_true',
                       help='Skip plot generation')
    parser.add_argument('--skip-tables', action='store_true',
                       help='Skip table generation')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(" "*15 + "ATLAS Publication Materials Generator")
    print("="*70)
    print("\nThis script generates IEEE-quality figures and tables for your paper.")
    print(f"Results directory: {Path(args.results_dir).absolute()}")
    print(f"Figures directory: {Path(args.figures_dir).absolute()}")
    print(f"Tables directory: {Path(args.tables_dir).absolute()}")
    print("="*70 + "\n")
    
    # Generate plots
    if not args.skip_plots:
        print("\n📊 GENERATING PLOTS\n")
        plotter = PublicationPlotter(
            results_dir=args.results_dir,
            output_dir=args.figures_dir
        )
        plotter.generate_all_plots()
    else:
        print("\n⏭️  Skipping plots (--skip-plots flag set)\n")
    
    # Generate tables
    if not args.skip_tables:
        print("\n📋 GENERATING TABLES\n")
        table_gen = ResultsTableGenerator(
            results_dir=args.results_dir,
            output_dir=args.tables_dir
        )
        table_gen.generate_all_tables()
    else:
        print("\n⏭️  Skipping tables (--skip-tables flag set)\n")
    
    print("\n" + "="*70)
    print("✅ ALL PUBLICATION MATERIALS GENERATED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated Files:")
    print("\nFigures (PNG):")
    if not args.skip_plots:
        figures = [
            "  • fig1_ablation_accuracy.png - Ablation accuracy convergence",
            "  • fig1_ablation_f1.png - Ablation F1 convergence",
            "  • fig2_model_accuracy.png - Model architecture accuracy comparison",
            "  • fig2_model_baseline.png - ATLAS vs Standard FL baselines",
            "  • fig3_cumulative_communication.png - Cumulative communication",
            "  • fig3_communication_efficiency.png - Communication efficiency",
            "  • fig3_total_communication.png - Total data transferred",
            "  • fig4_per_client_accuracy.png - Per-client accuracy evolution",
            "  • fig4_client_distribution.png - Client accuracy distribution (boxplot)",
            "  • fig6_accuracy_vs_time.png - Accuracy vs cumulative training time",
            "  • fig6_time_per_round.png - Avg training time per round",
            "  • fig7_clustering_visualization.png - Client cluster assignments",
            "  • fig7_cluster_distribution.png - Cluster size distribution",
            "  • fig8_importance_scores.png - Layer importance scores",
            "  • fig8_importance_distribution.png - Relative layer importance (pie)",
            "  • fig9_cluster_map.png - 2D client embedding colored by cluster",
        ]
        for fig in figures:
            print(fig)
    
    print("\nTables (CSV + LaTeX):")
    if not args.skip_tables:
        tables = [
            "  • table1_main_results - Main performance comparison",
            "  • table2_cross_model - Cross-model performance",
            "  • table3_ablation - Ablation study details",
            "  • table4_communication - Communication efficiency",
            "  • table6_statistical_summary - Overall statistics",
        ]
        for table in tables:
            print(table)
    
    print("\n" + "="*70)
    print("\n📝 Next Steps:")
    print("  1. Review all generated figures and tables")
    print("  2. Copy LaTeX tables to your paper")
    print("  3. Include figures using \\includegraphics{}")
    print("  4. Adjust captions and labels as needed")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
