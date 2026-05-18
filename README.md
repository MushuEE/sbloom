# Semantic Bloom Filter (sbloom)

This project explores different implementations of a "Semantic Bloom Filter" – a probabilistic data structure designed to quickly identify if a high-dimensional vector (embedding) definitely does *not* have a semantically similar match in a cache or database.

The goal is to create a fast "negative lookup gate" to protect expensive databases or caches from unnecessary queries.

## The Evolution of the Architecture

Here is a paper trail of the design choices and trade-offs explored in this repository.

### 1. Initial Implementation: Brute-Force Scan in Redis Lua
*   **Verdict**: Rejected for Production. Blocks the single-threaded Redis event loop and scales linearly $O(N)$.

### 2. Multi-Index Hashing (MIH) in Redis
*   **Result**: ~500x speedup in benchmarks.
*   **Verdict**: Great for Near-Duplicates. However, too strict for loose similarity due to bit flips.

### 3. The Industry Standard: HNSW / IVF
*   **Verdict**: Best for Search, but High Overhead. Requires significant memory and infrastructure.

### 4. In-Memory LSH Bloom Filter (The Final Iteration)
*   **Approach**: Move the gate *inside* the application memory to avoid network calls entirely on a miss.
*   **Verdict**: Demonstrates the fundamental LSH trade-off. To reduce false negatives, you increase false positives.

---

## The Breakthrough: Cost Asymmetry in Production

### The Principle of Cost Asymmetry

In a Semantic Cache architecture, the costs of errors are highly unequal:
*   **Cost of a False Positive**: You check the cache, miss, and then do inference anyway. You wasted a cheap network call to the cache (microseconds and fractions of a cent).
*   **Cost of a False Negative**: You assume the query is new, skip the cache, and go straight to the LLM. You wasted **seconds of latency** and **actual dollars** on tokens for an answer you already had.

### Conclusion: The 8x4 Config is a Winner

Because **Inference is orders of magnitude more expensive than Cache lookups**, we want to minimize False Negatives at all costs. The **8 tables / 4 planes** configuration achieves a **~98% success rate** for 0.78 similarity, making it a highly viable, low-cost gatekeeper.

---

## Future Research: The Eviction Challenge

A standard Bloom Filter does not support **deletion**. As items are evicted from the main cache, the Bloom Filter retains their bits, leading to a slow rise in False Positives.

### Cuckoo Filters
As a future area of research, we propose exploring **Cuckoo Filters** to replace the Bloom Filter. Cuckoo Filters support deletion natively without the memory overhead of counters, making them ideal for dynamic caches with high eviction rates.

## Files
*   `sbloom.py`: Redis implementations.
*   `lsh_bloom.py`: In-memory LSH implementation.
*   `test_lsh.py`: Verification script.
