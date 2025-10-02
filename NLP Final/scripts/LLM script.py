import json
import ollama
import csv
from typing import List
import re

# === CONFIG ===
json_file = 'lies.json'  # Replace with your path
models = ['gemma3:1b', 'gemma3:latest', 'llama3.2:1b', 'llama3.1:latest', 'deepseek-r1:1.5B', 'deepseek-r1:8b']  # or any aliases you use in Ollama
few_shot_preamble = "" #
max_examples = 50  # set to an integer to limit number of examples

# === FUNCTIONS ===

def load_examples(json_path: str):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['examples']

def extract_answer(text: str):
    # Extracts a single numerical value from the response, including negative numbers
    try:
        # Regular expression to match both integers and floats (including negative ones)
        numbers = [float(num) for num in re.findall(r'-?\d+\.?\d*', text)]
        
        if len(numbers) == 1:
            return numbers[0]  # Return the single numerical value
        else:
            print(f"Error: Multiple or no numerical values found in response: {text}")
            return None
    except Exception as e:
        print(f"Error extracting answer: {e}")
        return None

def evaluate_models(models: List[str], few_shot: str, examples: List[dict]):
    results = []
    print("\n🧠 Evaluating models...")

    for ex in examples:
        row = {
            "input": ex['input'],
            "expected": ex['target'],
            'Gemma3_1B': "", 
            'Gemma3_4B': "", 
            'Llama3.2_1B': "", 
            'Llama3.1_8B': "",
            'deepseek-r1:1.5B': "",
            "deepseek-r1:8b": ""
        }

        for model in models:
            try:
                print(f"\nEvaluating model: {model}")
                prompt = f"{few_shot.strip()}\n\nQ:{ex['input']}\nA:"
                print(f"Prompt: {prompt}")
                response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
                output = response['message']['content'][-30000:]
                print(f"Output from {model}: {output}")
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except:
                output = "ERROR"
                print("ERROR")

            # Store the raw output in the corresponding model's column
            if "gemma3:1b" in model:
                row["Gemma3_1B"] = output
            elif "gemma3:latest" in model:
                row["Gemma3_4B"] = output
            elif "llama3.2:1b" in model:
                row["Llama3.2_1B"] = output
            elif "llama3.1:latest" in model:
                row["Llama3.1_8B"] = output
            elif "deepseek-r1:1.5B" in model:
                row["deepseek-r1:1.5B"] = output
            elif "deepseek-r1:8b" in model:
                row["deepseek-r1:8b"] = output

        results.append(row)

    return results

def write_results_to_csv(results, output_file):
    headers = [
        "Riddle", "Correct Answer:",
        "Gemma1B Answer",
        "Gemma4B Answer", "Llama 1B Answer", "Llama 8B Answer", "Deepseek 1.5B Answer", "Deepseek 8B Answer"
    ]

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for result in results:
            row = [
                result['input'], result['expected'],
                result['Gemma3_1B'], result['Gemma3_4B'],
                result['Llama3.2_1B'], result['Llama3.1_8B'],
                result['deepseek-r1:1.5B'], result['deepseek-r1:8b']
            ]
            writer.writerow(row)

# === MAIN ===

if __name__ == "__main__":
    all_examples = load_examples(json_file)
    if max_examples:
        all_examples = all_examples[:max_examples]

    all_results = evaluate_models(models, few_shot_preamble, all_examples)
    write_results_to_csv(all_results, "evaluation_results_lies.csv")
    print("Results saved to evaluation_results_lies.csv")