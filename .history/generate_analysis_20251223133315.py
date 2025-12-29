import json
import os

# Load extracted papers
json_file = r"c:\Users\Hp\Downloads\Advanced_project\ATLAS\all_papers_extracted.json"
with open(json_file, 'r', encoding='utf-8') as f:
    papers = json.load(f)

# Create comprehensive analysis document
analysis_doc = """
================================================================================
COMPREHENSIVE RESEARCH ANALYSIS: ATLAS PROJECT PAPERS
================================================================================
Date: December 23, 2025
Workspace: c:\\Users\\Hp\\Downloads\\Advanced_project\\ATLAS

Papers Analyzed:
1. ATLAS_Base_Specification.pdf (1 page)
2. ATLAS_V1.pdf (22 pages) 
3. HSplitLoRA.pdf (16 pages)
4. MIRA_A_Method_of_Federated_Multi-Task_Learning_for_Large_Language_Models.pdf (5 pages)
5. Privacy-Aware_Split_Federated_Learning_for_LLM_Fine-Tuning_over_Internet_of_Things.pdf (12 pages)
6. SplitLoRA.pdf (9 pages)
7. VFLAIR-LLM.pdf (12 pages)

================================================================================
"""

# Process each paper
for paper_name in sorted(papers.keys()):
    paper_data = papers[paper_name]
    content = paper_data['content']
    
    analysis_doc += f"\n{'='*80}\n"
    analysis_doc += f"PAPER: {paper_name}\n"
    analysis_doc += f"Total Pages: {paper_data['pages']}\n"
    analysis_doc += f"{'='*80}\n\n"
    
    # Extract first 5000 chars for analysis
    analysis_doc += f"EXTRACTED CONTENT (First 5000 characters):\n"
    analysis_doc += f"{content[:5000]}\n"
    analysis_doc += f"\n[... Content continues ...]\n\n"

# Save analysis
output_file = r"c:\Users\Hp\Downloads\Advanced_project\ATLAS\RESEARCH_ANALYSIS.txt"
with open(output_file, 'w', encoding='utf-8', errors='ignore') as f:
    f.write(analysis_doc)

print(f"Analysis saved to: {output_file}")
print(f"Total size: {len(analysis_doc)} characters")
