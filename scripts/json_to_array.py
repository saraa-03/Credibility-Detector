import json
import os

# Define input/output paths
INPUT_FILE = "thesis/data/raw/metadata.json"
OUTPUT_FILE = "thesis/data/processed/metadata_array1.json"

def convert_jsonl_to_array(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    json_objects = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            json_objects.append(obj)
        except json.JSONDecodeError as e:
            print(f"⚠️ Skipping malformed line {i+1}: {e}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_objects, f, indent=2)

    print(f"✅ Converted {len(json_objects)} objects from {input_path}")
    print(f"📁 Output written to: {output_path}")

if __name__ == "__main__":
    convert_jsonl_to_array(INPUT_FILE, OUTPUT_FILE)
