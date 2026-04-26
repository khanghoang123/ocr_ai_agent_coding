import json

files = [
    "notebooks/08_train_experiment_B.ipynb",
    "notebooks/08_experiment_B_10000_iter.ipynb",
    "notebooks/03_train_baseline_10000_iter.ipynb",
    "notebooks/base-line_50000_iter.ipynb"
]

for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as fp:
            nb = json.load(fp)
        print(f"\n=== {f} ===")
        for cell in nb.get('cells', []):
            if cell.get('cell_type') == 'markdown':
                content = "".join(cell.get('source', []))
                if 'CER' in content or 'WER' in content:
                    print(f"--- Markdown ---\n{content.strip()}")
            elif cell.get('cell_type') == 'code':
                for out in cell.get('outputs', []):
                    if out.get('output_type') == 'stream':
                        text = "".join(out.get('text', []))
                        if 'CER' in text or 'WER' in text:
                            lines = [line.strip() for line in text.split('\n') if ('CER' in line or 'WER' in line) and 'Epoch' not in line]
                            if lines:
                                print(f"--- Output ---\n{lines[-2:]}") # last two to get final eval
    except Exception as e:
        print(f"Error reading {f}: {e}")
