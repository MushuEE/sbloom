import time
import numpy as np
import redis
from sbloom import SemanticBloomFilter


def generate_synthetic_data(n_vectors, dimension):
    """Generates random normalized vectors."""
    print(
        f"Generating {n_vectors} synthetic vectors of dimension {dimension}..."
    )
    # Generate random vectors
    data = np.random.randn(n_vectors, dimension).astype(np.float32)
    # Normalize to unit length (common for embeddings)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data = data / norms
    return data


def run_benchmark():
    # Connect to local Redis
    redis_client = redis.Redis(host="localhost", port=6379, db=0)

    try:
        redis_client.ping()
    except redis.ConnectionError:
        print(
            "Error: Could not connect to Redis at localhost:6379. Please ensure Redis is running."
        )
        return

    # Flush Redis to start fresh
    print("Flushing Redis...")
    redis_client.flushdb()

    dimension = 1536
    n_vectors = 10000
    n_queries = 100
    target_similarity = 0.9

    vectors = generate_synthetic_data(n_vectors, dimension)
    queries = generate_synthetic_data(n_queries, dimension)

    sbf = SemanticBloomFilter(redis_client, dimension=dimension)

    # 1. Measure Population
    print("Populating filter...")
    start_mem = int(redis_client.info("memory")["used_memory"])

    start_time = time.time()
    for i in range(n_vectors):
        sbf.add(f"v{i}", vectors[i])
    pop_time = time.time() - start_time

    end_mem = int(redis_client.info("memory")["used_memory"])
    mem_used = end_mem - start_mem

    print(f"Population took {pop_time:.2f} seconds.")
    print(f"Memory used: {mem_used / 1024 / 1024:.2f} MB")
    print(f"Memory per vector: {mem_used / n_vectors:.2f} bytes")

    # 2. Benchmark check_via_app
    print("\nBenchmarking check_via_app...")
    app_latencies = []
    app_hits = 0

    start_time = time.time()
    for i in range(n_queries):
        q_start = time.time()
        hit = sbf.check_via_app(queries[i], target_similarity)
        app_latencies.append(time.time() - q_start)
        if hit:
            app_hits += 1
    app_total_time = time.time() - start_time
    app_qps = n_queries / app_total_time

    # 3. Benchmark check_via_lua
    print("Benchmarking check_via_lua...")
    lua_latencies = []
    lua_hits = 0

    start_time = time.time()
    for i in range(n_queries):
        q_start = time.time()
        hit = sbf.check_via_lua(queries[i], target_similarity)
        lua_latencies.append(time.time() - q_start)
        if hit:
            lua_hits += 1
    lua_total_time = time.time() - start_time
    lua_qps = n_queries / lua_total_time

    # Calculate percentiles
    def get_percentiles(latencies):
        l_ms = [l * 1000 for l in latencies]
        return {
            "avg": np.mean(l_ms),
            "p95": np.percentile(l_ms, 95),
            "p99": np.percentile(l_ms, 99),
        }

    app_stats = get_percentiles(app_latencies)
    lua_stats = get_percentiles(lua_latencies)

    # Estimate payload sizes
    # App side: fetches all keys (SCAN) + pipeline GETs for all keys.
    # Key size ~30 bytes, Value size 192 bytes. Total ~220 bytes per key.
    # 10,000 keys * 220 bytes ~ 2.2 MB per query.
    # Lua side: sends query (192 bytes) + meta. Receives 1 byte. ~ 200 bytes per query.
    app_payload_est = "2.2 MB"
    lua_payload_est = "200 bytes"

    print("\n" + "=" * 40)
    print("PERFORMANCE REPORT")
    print("=" * 40)
    print(f"Dataset: {n_vectors} vectors (1536-dim)")
    print(f"Queries: {n_queries}")
    print(f"Target Similarity: {target_similarity}")
    print("-" * 40)
    print(f"{'Metric':<25} {'App-side':<15} {'Lua-side':<15}")
    print("-" * 40)
    print(f"{'Queries per Second (QPS)':<25} {app_qps:<15.2f} {lua_qps:<15.2f}")
    print(
        f"{'Avg Latency (ms)':<25} {app_stats['avg']:<15.2f} {lua_stats['avg']:<15.2f}"
    )
    print(
        f"{'P95 Latency (ms)':<25} {app_stats['p95']:<15.2f} {lua_stats['p95']:<15.2f}"
    )
    print(
        f"{'P99 Latency (ms)':<25} {app_stats['p99']:<15.2f} {lua_stats['p99']:<15.2f}"
    )
    print("-" * 40)
    print(f"Total Network Payload Size Differences (Estimated):")
    print(f"  App-side: ~{app_payload_est} per query (fetches all data)")
    print(f"  Lua-side: ~{lua_payload_est} per query (in-memory on server)")
    print("-" * 40)
    print(f"Memory footprint per vector: {mem_used / n_vectors:.2f} bytes")
    print(f"Total Redis memory used: {mem_used / 1024 / 1024:.2f} MB")
    print("=" * 40)

    print(f"Hits: App={app_hits}, Lua={lua_hits}")


if __name__ == "__main__":
    run_benchmark()
