import os
import json
import random
from pathlib import Path

def collect_test_files(root_dir):
    test_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        if 'train.jsonl' in filenames:
            test_files.append(Path(dirpath) / 'train.jsonl')
    return test_files

def validate_entry(entry):
    return (
        isinstance(entry, dict) and
        all(key in entry for key in ['id', 'question', 'answer']) and
        isinstance(entry['answer'], list)
    )

def load_and_filter_data(file_path):
    valid_entries = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    if validate_entry(entry):
                        valid_entries.append(entry)
                except json.JSONDecodeError:
                    print(f"JSON failed: {file_path} {line_number} ")
    except Exception as e:
        print(f"read failed: {file_path}: {str(e)}")
    return valid_entries

def sample_entries(all_entries, sample_size=300,seed=42):
    random.seed(seed)
    if len(all_entries) < sample_size:
        print(f"warning: only found {len(all_entries)} data, less than {sample_size} data")
        return all_entries
    return random.sample(all_entries, sample_size)

def main():
    input_dir = '../processed/paraphrasing'
    output_file = '../para_sampled_300.jsonl'

    test_files = collect_test_files(input_dir)
    
    all_entries = []
    for file_path in test_files:
        entries = load_and_filter_data(file_path)
        all_entries.extend(entries)
    
    sampled_data = sample_entries(all_entries)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in sampled_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"done! save {len(sampled_data)} data to {output_file}")

if __name__ == "__main__":
    main()