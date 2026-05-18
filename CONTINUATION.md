# Continuation Instructions - Semantic Bloom Filter

We have implemented a Semantic Bloom Filter caching layer in Python with Redis. The goal is to compare the performance of searching for similar binary-quantized vectors in Python vs. in a Redis Lua script.

## Current State

### Implementation Details
- **Binary Quantization**: 1536-dimensional float vectors are converted to 192-byte binary strings by taking the sign of each dimension (`float > 0 -> 1`, `else 0`) and packing bits into bytes using NumPy.
- **Distance Metric**: Hamming distance is used to approximate cosine similarity.
- **Threshold Formula**: The target cosine similarity $S$ is mapped to a Hamming distance threshold $D$ using the formula $D = N \frac{\arccos(S)}{\pi}$, where $N = 1536$.
- **Storage Strategy**: Vectors are stored as separate keys with the prefix `sbloom:vec:1536:<id>` to facilitate scanning and pipelined fetching.

### Components
- `sbloom.py`: Implements the `SemanticBloomFilter` class.
    - **App-side Lookup (`check_via_app`)**: Scans for all keys matching the prefix, fetches them using Redis pipelines, and computes the Hamming distance in Python using `int.bit_count()`.
    - **Lua-side Lookup (`check_via_lua`)**: Executes an embedded Lua script on the Redis server. The script finds keys matching the prefix using `KEYS`, iterates over them, performs bitwise XOR using the `bit.bxor` library, and counts bits using a hardcoded popcount lookup table. It returns 1 (found) as soon as a match within the threshold is found, short-circuiting the search.
- `benchmark.py`: A script to generate 10,000 synthetic vectors, populate Redis, and run 100 queries using both methods to compare QPS, latency, and estimated payload size.

## Next Steps on the New Machine

1. **Ensure Redis is Running**:
   The benchmark script expects Redis at `localhost:6379`. Ensure it is running before proceeding.

2. **Run the Benchmark**:
   Ensure dependencies (`redis`, `numpy`) are installed in your Python environment, then run:
   ```bash
   python3 benchmark.py
   ```

3. **Verify Results**:
   - Check the "Hits" count at the bottom of the report. They should be identical or very close between App-side and Lua-side.
   - Compare QPS and Latency. The Lua-side should be significantly faster due to reduced network payload and in-memory processing.

4. **Potential Improvements**:
   - If global keyspace pollution or `KEYS` performance is a concern, consider switching to a Redis Hash (e.g., `HVALS` in Lua), though that would require adjusting the pipeline logic in `check_via_app`.
