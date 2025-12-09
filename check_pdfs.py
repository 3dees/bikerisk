import pdfplumber
from langdetect import detect_langs

# Check both PDFs for non-English content
pdfs = [
    r'c:\csvs\UL_2271_2023.pdf',
    r'c:\csvs\SS_EN_50604_1_EN.pdf',
    r'c:\csvs\IEC 62133-2 {ed1.0}.pdf'
]

for pdf_path in pdfs:
    print(f"\n{'='*60}")
    print(f"Checking: {pdf_path.split(chr(92))[-1]}")
    print('='*60)
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            non_english_pages = []
            total_pages = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages[:10], 1):  # Check first 10 pages
                text = page.extract_text() or ""
                
                if text and len(text.strip()) > 50:
                    try:
                        langs = detect_langs(text[:500])
                        top_lang = langs[0]
                        
                        if top_lang.lang != 'en':
                            non_english_pages.append({
                                'page': page_num,
                                'lang': top_lang.lang,
                                'confidence': top_lang.prob,
                                'sample': text[:100]
                            })
                    except:
                        pass
            
            if non_english_pages:
                print(f"Found {len(non_english_pages)} non-English pages in first 10:")
                for item in non_english_pages:
                    print(f"  Page {item['page']}: {item['lang']} (confidence: {item['confidence']:.2f})")
                    print(f"    Sample: {item['sample']}...\n")
            else:
                print(f"No non-English content detected in first 10 pages")
    except Exception as e:
        print(f"Error: {e}")
