import json
import ollama
import csv
from typing import List
import re
import sys
import time

# === CONFIG ===
json_file = 'object_counting.json'  # Your dataset
models = ['deepseek-r1:1.5b', 'deepseek-r1:8b','gemma3:1b','gemma3:latest','llama3.2:1b','llama3.1']
few_shot_preamble = ""
max_examples = 20  # Number of examples per chunk
max_retries = 5
retry_delay = 10

# === FUNCTIONS ===

def load_examples(json_path: str):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['examples']

def extract_answer(text: str):
    try:
        numbers = [float(num) for num in re.findall(r'-?\d+\.?\d*', text)]
        if len(numbers) == 1:
            return numbers[0]
        else:
            print(f"Error: Multiple or no numerical values found in response: {text}")
            return None
    except Exception as e:
        print(f"Error extracting answer: {e}")
        return None

def evaluate_models(models: List[str], few_shot: str, examples: List[dict]):
    results = []
    print("Evaluating models...")
    count = 1
    for ex in examples:
        print('Question', count)
        row = {
            "input": ex['input'],
            "expected": ex['target'],
            "Deepseek1.5": "",
            "Deepseek8": "",
            "Gemma3:1b":"",
            "Gemma3:latest":"",
            "Llama3.2:1b":"",
            "Llama3.1":""
        }

        for model in models:
            print(f"\nEvaluating model: {model}")
            prompt = f"{few_shot.strip()}\n\nQ:{ex['input']}\nA:"
            print(f"Prompt: {prompt}")

            for attempt in range(max_retries):
                try:
                    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
                    output = response['message']['content']
                    break
                except Exception as e:
                    print(f"⚠️ Error with model {model} on attempt {attempt+1}: {e}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print(" Max retries exceeded. Skipping this example.")
                        output = "ERROR"

            if "1.5b" in model:
                row["Deepseek1.5"] = output
            elif "8b" in model:
                row["Deepseek8"] = output
            elif "gemma3:1b" in model:
                row['Gemma3:1b']=output
            elif "gemma3:latest" in model:
                row["Gemma3:latest"]=output
            elif 'llama3.2:1b' in model:
                row["Llama3.2:1b"]=output
            else:
                row["Llama3.1"]=output
        
        results.append(row)
        count += 1

    return results

def write_results_to_csv(results, output_file):
    headers = [
        "Object Counting question", "Correct Answer:",
        "Deepseek1.5 Answer","Deepseek8 Answer","Gemma1 Answer","Gemma4 Answer","Llama3.2 Answer","Llama3.1 Answer" 
    ]

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for result in results:
            row = [
                result['input'], result['expected'],
                result['Deepseek1.5'],
                result['Deepseek8'],
                result['Gemma3:1b'],
                result['Gemma3:latest'],
                result["Llama3.2:1b"],
                result["Llama3.1"]
            ]
            writer.writerow(row)

# === MAIN ===

if __name__ == "__main__":
    start_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    all_examples = load_examples(json_file)
    chunk = all_examples[start_index:start_index + max_examples]

    if not chunk:
        print("All examples processed.")
        sys.exit(0)

    all_results = evaluate_models(models, few_shot_preamble, chunk)
    write_results_to_csv(all_results, f"evaluation_results_{start_index}.csv")

    with open("checkpoint.txt", "w") as f:
        f.write(str(start_index + max_examples))

    print("Chunk complete. Exiting for restart.")
    sys.exit(0)
