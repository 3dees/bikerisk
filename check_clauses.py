import json
from langdetect import detect_langs

with open('outputs/harmonization_groups.json', 'r', encoding='utf-8') as f:
    groups = json.load(f)

# Check the clauses inside groups for non-English text
non_english_found = []
for group_idx, group in enumerate(groups[:50]):  # Check first 50 groups
    clauses = group.get('clauses', [])
    for clause in clauses:
        clause_text = clause.get('text', '')
        if clause_text and len(clause_text) > 30:
            try:
                langs = detect_langs(clause_text[:300])
                top_lang = langs[0]
                if top_lang.lang != 'en':
                    non_english_found.append({
                        'group': group_idx,
                        'clause': clause.get('clause_number'),
                        'standard': clause.get('standard_name'),
                        'lang': top_lang.lang,
                        'confidence': top_lang.prob,
                        'text': clause_text[:100]
                    })
            except:
                pass

if non_english_found:
    print(f'Found {len(non_english_found)} non-English clauses in first 50 groups\n')
    for item in non_english_found[:10]:
        print(f"Group {item['group']}, Clause {item['clause']} ({item['standard']})")
        print(f"  Language: {item['lang']} (confidence: {item['confidence']:.2f})")
        print(f"  Text: {item['text']}...")
        print()
else:
    print('No non-English content found in first 50 groups')
