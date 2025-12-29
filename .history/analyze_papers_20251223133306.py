import json
import re

# Load extracted papers
with open(r"c:\Users\Hp\Downloads\Advanced_project\ATLAS\all_papers_extracted.json", 'r', encoding='utf-8') as f:
    papers = json.load(f)

# Create analysis
analysis = {}

for paper_name, paper_data in papers.items():
    content = paper_data['content']
    
    # Extract abstract/introduction (first 1000 chars)
    abstract = content[:2000]
    
    # Extract key sections
    analysis[paper_name] = {
        "pages": paper_data['pages'],
        "preview": abstract,
        "length": len(content)
    }

# Print analysis
for paper_name, info in analysis.items():
    print(f"\n{'='*80}")
    print(f"PAPER: {paper_name}")
    print(f"Pages: {info['pages']}")
    print(f"Content Size: {info['length']} chars")
    print(f"{'='*80}")
    print(info['preview'][:1500])
    print()
