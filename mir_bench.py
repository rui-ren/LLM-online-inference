import csv
import json
import os
import ssl
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple, List

import numpy as np
import requests
import transformers

PROMPT = "What is Deep Learning?"
MAX_NEW_TOKENS = 256
MAX_LENGTH = 4096
PARAMS = {"max_tokens": MAX_NEW_TOKENS}
THRESHOLD_TPS = 5  # Threshold for tokens per second below which we deem the query to be slow
MODEL_PATH = "mistralai/Mistral-7B-v0.1"

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
# AML MIR deployment details
AML_URI = "http://localhost:8000/generate"
AML_API_KEY = ""
AML_DEPLOYMENT_NAME = ""


def generate_text_mir():
    def allowSelfSignedHttps(allowed):
        # bypass the server certificate verification on client side
        if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context',
                                                                               None):
            ssl._create_default_https_context = ssl._create_unverified_context

    allowSelfSignedHttps(True)

    data = {
        "prompt": "Explain superconductors like I'm five years old",
        "n": 1,
        "use_beam_search": False,
        "temperature": 1.0,
        "top_p": 0.9,
        "max_tokens": 500,
        "ignore_eos": False,
        "stream": False,
    }

    headers = {
        "User-Agent": "Benchmark Client",
        "Accept": "text/event-stream",
        "Content-Type": "application/json"
    }

    body = str.encode(json.dumps(data))


    start_time = time.time()
    req = urllib.request.Request(AML_URI, body, headers)
    try:
        response = urllib.request.urlopen(req)
        result = response.read()

        end_time = time.time()
        generated_text = json.loads(result.decode("utf8"))["text"][0]
        token_count = len(tokenizer.encode(generated_text))
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))
        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))
        raise

    return (end_time - start_time) * 1000, token_count  # Convert to ms


def generate_text() -> Tuple[int, int]:
    api_url = "http://127.0.0.1:8000/generate"

    headers = {
            "User-Agent": "Benchmark Client",
            "Accept": "text/event-stream",
            "Content-Type": "application/json"
    }

    payload = {
        "prompt": PROMPT,
        **PARAMS,
        "stream": False,
    }
    start_time = time.time()
    response = requests.post(api_url, headers=headers, json=payload)
    end_time = time.time()
    latency = (end_time - start_time) * 1000  # Convert to ms

    if response.status_code != 200:
        raise ValueError(f"Error: {response.content}")

    output = json.loads(response.content)["text"][0]
    token_count = len(tokenizer.encode(output))
    return latency, token_count


def evaluate_performance(concurrent_requests: int) -> Tuple[float, float, float, float, List[float]]:
    latencies = []
    total_tokens = 0
    tokens_per_second_each_request = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        future_to_req = {executor.submit(generate_text_mir): i for i in range(concurrent_requests)}
        for future in as_completed(future_to_req):
            latency, token_count = future.result()
            latencies.append(latency)
            total_tokens += token_count

            # Calculate tokens per second for this request
            tokens_per_sec = token_count / (latency / 1000)  # latency is in ms
            tokens_per_second_each_request.append(tokens_per_sec)

    end_time = time.time()
    total_time = end_time - start_time
    throughput = concurrent_requests / total_time  # RPS (requests per second)
    tokens_per_second_overall = total_tokens / total_time  # Overall tokens per second

    p50_latency = np.percentile(latencies, 50)
    p99_latency = np.percentile(latencies, 99)

    # Count the number of requests below the token-per-second threshold
    below_threshold_count = sum(1 for tps in tokens_per_second_each_request if tps < THRESHOLD_TPS)

    return p50_latency, p99_latency, throughput, tokens_per_second_overall, below_threshold_count, tokens_per_second_each_request


concurrent_levels = [1, 2, 4, 8, 16, 32, 64, 128, 256]

print(
    "| Number of Concurrent Requests | P50 Latency (ms) | P99 Latency (ms) | Throughput (RPS) | Tokens per Second | Number of Requests Below Threshold |")
print(
    "|-------------------------------|------------------|------------------|------------------|-------------------|------------------------------------|")

# Save to file
csv_file = "performance_metrics.csv"
with open(csv_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Number of Concurrent Requests", "P50 Latency (ms)", "P99 Latency (ms)", "Throughput (RPS)",
                     "Tokens per Second"])
    for level in concurrent_levels:
        p50_latency, p99_latency, throughput, tokens_per_second_overall, below_threshold_count, tokens_per_second_each_request = evaluate_performance(
            level)
        print(
            f"| {level} | {p50_latency:.2f} | {p99_latency:.2f} | {throughput:.2f} | {tokens_per_second_overall:.2f} | {below_threshold_count:.2f} |")
        writer.writerow(
            [level, p50_latency, p99_latency, throughput, tokens_per_second_overall, below_threshold_count])
