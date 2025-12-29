import pdfplumber
import os
import json

pdf_dir = r"c:\Users\Hp\Downloads\Advanced_project\ATLAS"
pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]

extracted_all = {}

for pdf_file in sorted(pdf_files):
    pdf_path = os.path.join(pdf_dir, pdf_file)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            full_text = ""
            
            # Extract all pages
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
            
            extracted_all[pdf_file] = {
                "pages": num_pages,
                "content": full_text[:15000]  # Store first 15KB per paper
            }
            print(f"✓ {pdf_file} ({num_pages} pages)")
            
    except Exception as e:
        print(f"✗ {pdf_file}: {str(e)}")

# Save all extracted content
with open(r"c:\Users\Hp\Downloads\Advanced_project\ATLAS\all_papers_extracted.json", 'w', encoding='utf-8') as f:
    json.dump(extracted_all, f, ensure_ascii=False, indent=2)

print("\nAll papers extracted successfully!")
