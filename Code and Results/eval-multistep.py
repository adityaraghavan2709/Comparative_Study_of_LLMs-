import json 
import ollama
import csv
from typing import List

# === CONFIG ===
json_file = 'tracking_shuffled_objects_seven_objects.json'
models = [
    'deepseek-r1:1.5b', 
    'deepseek-r1:8b', 
    'llama3.2:1b', 
    'llama3.1:latest', 
    'gemma3:1b', 
    'gemma3:latest'
]
few_shot_preamble = ""
max_examples = 250

# === FUNCTIONS ===

def load_examples(json_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['examples']

def evaluate_models(models: List[str], few_shot: str, examples: List[dict]):
    print("\n Evaluating models...")
    
    # Prepare rows ahead of time
    results = []
    for ex in examples:
        results.append({
            "input": ex['input'],
            "expected": ex['target'],
            "Deepseek1.5B": "",
            "Deepseek8B": "",
            "Llama3.2_1B": "",
            "Llama3.2_8B": "",
            "Gemma3_1B": "",
            "Gemma3_4B": ""
        })

    for model in models:
        print(f"\n Loading model: {model}")

        for idx, ex in enumerate(examples):
            prompt = f"{few_shot.strip()}\n\nQ: {ex['input']}\nA:"
            print(f"\n [{model}] Q{idx + 1}: {ex['input']}")

            try:
                response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
                output = response['message']['content']
            except Exception as e:
                output = f"[ERROR: {e}]"

            print(f"📤 Output: {output}")

            # Update corresponding result row
            if "deepseek-r1:1.5b" in model:
                results[idx]["Deepseek1.5B"] = output
            elif "deepseek-r1:8b" in model:
                results[idx]["Deepseek8B"] = output
            elif "llama3.2:1b" in model:
                results[idx]["Llama3.2_1B"] = output
            elif "llama3.2:8b" in model or "llama3:8b" in model or "llama3.1" in model:
                results[idx]["Llama3.2_8B"] = output
            elif "gemma3:1b" in model or "gemma:2b" in model:
                results[idx]["Gemma3_1B"] = output
            elif "gemma3:4b" in model or "gemma3:latest" in model or "gemma:7b" in model:
                results[idx]["Gemma3_4B"] = output

    return results

def write_results_to_csv(results, output_file):
    headers = [
        "Input", "Expected Answer",
        "Deepseek1.5B", "Deepseek8B",
        "Llama3.2_1B", "Llama3.2_8B",
        "Gemma3_1B", "Gemma3_4B"
    ]

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for result in results:
            row = [
                result['input'], result['expected'],
                result['Deepseek1.5B'], result['Deepseek8B'],
                result['Llama3.2_1B'], result['Llama3.2_8B'],
                result['Gemma3_1B'], result['Gemma3_4B']
            ]
            writer.writerow(row)

# === MAIN ===

if __name__ == "__main__":
    all_examples = load_examples(json_file)
    if max_examples:
        all_examples = all_examples[:max_examples]

    all_results = evaluate_models(models, few_shot_preamble, all_examples)
    write_results_to_csv(all_results, "evaluation_results_tracking_shuffled_objects_seven_objects.csv")
    print(" Results saved to evaluation_results_tracking_shuffled_objects_seven_objects.csv")
