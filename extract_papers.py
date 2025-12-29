import pdfplumber
import os
import sys

pdf_dir = r"c:\Users\Hp\Downloads\Advanced_project\ATLAS"
pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]

for pdf_file in sorted(pdf_files):
    pdf_path = os.path.join(pdf_dir, pdf_file)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            print(f"\n{'='*80}")
            print(f"FILE: {pdf_file}")
            print(f"Total Pages: {num_pages}")
            print('='*80)
            
            # Extract first 10 pages
            for i in range(min(10, num_pages)):
                page = pdf.pages[i]
                text = page.extract_text()
                if text:
                    print(f"\n--- PAGE {i+1} ---")
                    print(text[:2000])  # First 2000 chars per page
                    
    except Exception as e:
        print(f"Error with {pdf_file}: {str(e)}")
        import traceback
        traceback.print_exc()
