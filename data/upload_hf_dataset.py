"""
Upload to huggingface.
"""
import argparse
import json
from datasets import Dataset, DatasetDict, load_dataset
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default='data')
    args = parser.parse_args()
    for file_name in os.listdir(args.data_dir):
        
        if file_name.endswith('.json'):
            file_name = file_name.replace('.json', '')
            objs = json.load(open(f'{args.data_dir}/{file_name}.json'))
            print(f"#convs: {len(objs)}")
            data = Dataset.from_list(objs)
            data.push_to_hub(f"{}/{file_name}", private=True)
            print(f"Uploaded {file_name} to huggingface.")