import pandas as pd
from langdetect import detect_langs

# Read the CSV file
df = pd.read_csv(r'C:\Users\vsjam\Downloads\UL_2271_2023.pdf_requirements.csv')

print(f"Total rows in CSV: {len(df)}")

# Check for non-English text
non_english = []
for idx, row in df.iterrows():
    # Check the requirement column
    text = row.get('Requirement (Clause)', '')
    if pd.notna(text) and len(str(text)) > 20:
        try:
            langs = detect_langs(str(text)[:500])
            top_lang = langs[0]
            if top_lang.lang != 'en':
                non_english.append({
                    'index': idx,
                    'language': top_lang.lang,
                    'confidence': top_lang.prob,
                    'sample': str(text)[:100]
                })
                if len(non_english) >= 10:
                    break
        except Exception as e:
            pass

if non_english:
    print(f'\nFound {len(non_english)} non-English requirements:')
    for item in non_english:
        print(f'  Row {item["index"]}: {item["language"]} (confidence: {item["confidence"]:.2f})')
        print(f'    Sample: {item["sample"]}...')
else:
    print('\nNo non-English requirements detected in CSV')
