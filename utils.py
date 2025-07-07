from tenacity import (
    retry,
    wait_random_exponential,
)
import json
import os
import asyncio

import numpy as np
from scipy.interpolate import interp1d


# --- Begin Hugging Face Llama3 Integration ---
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Environment variable for cache directory, defaulting to specified path
CACHE_DIR = os.getenv("HF_CACHE_DIR", "/data/jiang/vennemdp/hf_cache")
LLAMA_MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"

# Load tokenizer and model once at module import
tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_ID, cache_dir=CACHE_DIR, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    LLAMA_MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    cache_dir=CACHE_DIR,
    trust_remote_code=True
)
model.eval()

def local_complete(messages, max_new_tokens=512, temperature=0.0):
    """
    Concatenates a list of messages into a single prompt and generates
    a completion using the local Llama model.
    """
    # Build prompt by concatenating message roles and content
    prompt = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prompt += f"<|{role}|>\n{content}\n"
    prompt += "<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract assistant's reply after the last assistant tag
    return text.split("<|assistant|>")[-1].strip()
# --- End Hugging Face Llama3 Integration ---


@retry(wait=wait_random_exponential(min=1, max=60))
def query_api(messages, engine=None, **kwargs):
    """
    Queries the local Llama model with the given messages.
    """
    # Always use local Llama3 for completion
    response_text = local_complete(messages, max_new_tokens=kwargs.get("max_tokens", 512), temperature=kwargs.get("temperature", 0.0))
    # Wrap in OpenAI-like response structure if needed
    response = {"choices": [{"message": {"content": response_text}}]}
    messages.append({"role": "assistant", "content": response_text})
    return response_text, response


"""
METRICS
"""

def update_metrics(metrics, new_metrics):
    if len(metrics) == 0:
        return {
            metric: [new_metrics[metric]] for metric in new_metrics
        }
    for metric in metrics:
        metrics[metric].append(new_metrics[metric])
    return metrics


def update_test_responses(all_test_responses, new_test_responses):
    if len(all_test_responses) == 0:
        all_test_responses = []
        for t in range(len(new_test_responses)):
            all_test_responses.append(new_test_responses[t])
            all_test_responses[t]["pred"] = [all_test_responses[t]["pred"]]
            all_test_responses[t]["pred_prob"] = [all_test_responses[t]["pred_prob"]]
            all_test_responses[t]["correct_prob"] = [all_test_responses[t]["correct_prob"]]
            all_test_responses[t]["correct?"] = [all_test_responses[t]["correct?"]]
        return all_test_responses

    for t in range(len(all_test_responses)):
        all_test_responses[t]["pred"].append(new_test_responses[t]["pred"])
        all_test_responses[t]["pred_prob"].append(new_test_responses[t]["pred_prob"])
        all_test_responses[t]["correct_prob"].append(new_test_responses[t]["correct_prob"])
        all_test_responses[t]["correct?"].append(new_test_responses[t]["correct?"])
    return all_test_responses


def average_lines(lines, num_points=100):
    """
    Average a list of lines.

    Parameters:
        lines: A list of lines, where each line is a 2D numpy array with shape (n, 2),
               where n is the number of points on the line, and the 2 columns are x and y coordinates.
        num_points: The number of points to use in the averaged line. Default is 100.

    Returns:
        average_line: A 2D numpy array with shape (num_points, 2), where the columns are x and y coordinates
                      of the averaged line.
    """
    
    # Step 1: Create interpolation functions
    interp_funcs = [interp1d(line[:,0], line[:,1], kind='linear', fill_value='extrapolate') for line in lines]

    # Step 2: Define a set of x values
    min_x = min(line[:,0].min() for line in lines)
    max_x = max(line[:,0].max() for line in lines)
    x_values = np.linspace(min_x, max_x, num_points)

    # Step 3: Calculate y values at these x's for each line and average them
    y_values = np.mean([interp(x_values) for interp in interp_funcs], axis=0)

    # Step 4: Calculate errors on the mean
    y_errors = np.std([interp(x_values) for interp in interp_funcs], axis=0)

    # Now you have your average line
    average_line = np.column_stack((x_values, y_values))
    assert y_errors.shape[0] == y_values.shape[0]
    
    return average_line, y_errors
