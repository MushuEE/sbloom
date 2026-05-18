import math
import numpy as np
import redis


def binary_quantize(vector: np.ndarray) -> bytes:
    """Quantizes a float vector into a binary string.

    If a float is > 0, sets the bit to 1, otherwise 0.
    Packs bits efficiently into bytes.
    """
    vec = np.asarray(vector)
    bits = (vec > 0).astype(np.uint8)
    packed = np.packbits(bits)
    return packed.tobytes()


class SemanticBloomFilter:

    def __init__(self, redis_client: redis.Redis, dimension: int = 1536):
        """Initializes the Semantic Bloom Filter.

        Args:
            redis_client: An initialized redis-py client.
            dimension: Dimensionality of the vectors (default 1536).
        """
        self.redis = redis_client
        self.dimension = dimension
        self.prefix = f"sbloom:vec:{dimension}:"

        # Register Lua script
        self.lua_script = self.redis.register_script(self._get_lua_script())

    def _get_lua_script(self) -> str:
        """Returns the Lua script for in-Redis Hamming distance check."""
        return """
        local query = ARGV[1]
        local threshold = tonumber(ARGV[2])
        local prefix = ARGV[3]
        
        -- Find all keys with the prefix
        local keys = redis.call('KEYS', prefix .. '*')
        
        -- Popcount lookup table for fast bit counting
        local popcount = {0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
                          1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
                          1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
                          2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
                          1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
                          2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
                          3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
                          3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
                          4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8}
        
        for i=1, #keys do
            local val = redis.call('GET', keys[i])
            if val then
                local dist = 0
                for j=1, #query do
                    local b1 = string.byte(query, j)
                    local b2 = string.byte(val, j)
                    local xor = bit.bxor(b1, b2)
                    dist = dist + popcount[xor + 1]
                end
                if dist <= threshold then
                    return 1 -- Found
                end
            end
        end
        return 0 -- Not found
        """

    def _compute_threshold(self, target_cosine_similarity: float) -> int:
        """Computes the Hamming distance threshold based on cosine similarity.

        Uses the relation D = N * acos(S) / pi.
        """
        # Clip to avoid math domain error due to precision
        target_cosine_similarity = max(-1.0, min(1.0, target_cosine_similarity))
        angle = math.acos(target_cosine_similarity)
        return int(self.dimension * angle / math.pi)

    def add(self, vector_id: str, vector_floats: np.ndarray):
        """Quantizes and stores the vector in Redis."""
        packed = binary_quantize(vector_floats)
        key = f"{self.prefix}{vector_id}"
        self.redis.set(key, packed)

    def add_mih(self, vector_id: str, vector_floats: np.ndarray):
        """Quantizes and stores the vector in Redis using Multi-Index Hashing."""
        packed = binary_quantize(vector_floats)
        
        num_blocks = 16
        block_size = (self.dimension // 8) // num_blocks

        
        for i in range(num_blocks):
            start = i * block_size
            end = start + block_size
            block = packed[start:end]
            
            # Store in Redis Set
            # Key: sbloom:mih:block:{i}:{block_hex}
            key = f"sbloom:mih:block:{i}:{block.hex()}"
            self.redis.sadd(key, vector_id)
            
        # Also store the full vector for verification
        key = f"{self.prefix}{vector_id}"
        self.redis.set(key, packed)


    def check_mih(self, query_floats: np.ndarray) -> dict:
        """Checks for similar vectors using Multi-Index Hashing and returns scores.
        
        Returns a dict mapping vector_id to the fraction of matching blocks (match/total).
        """
        query_packed = binary_quantize(query_floats)
        
        num_blocks = 16
        block_size = (self.dimension // 8) // num_blocks

        
        # Pipeline SMEMBERS to get candidates from all matching blocks
        pipe = self.redis.pipeline()
        for i in range(num_blocks):
            start = i * block_size
            end = start + block_size
            block = query_packed[start:end]
            key = f"sbloom:mih:block:{i}:{block.hex()}"
            pipe.smembers(key)
            
        results = pipe.execute()
        
        # Count frequencies of each candidate
        from collections import Counter
        counts = Counter()
        for res in results:
            if res:
                for vid in res:
                    counts[vid.decode('utf-8')] += 1
                    
        # Convert to scores (match/total)
        scores = {vid: count / num_blocks for vid, count in counts.items()}
        return scores

    def check_via_app(self, query_floats: np.ndarray, target_cosine_similarity: float) -> bool:

        """Checks for similar vectors in Python after fetching all from Redis."""
        query_packed = binary_quantize(query_floats)
        threshold = self._compute_threshold(target_cosine_similarity)

        # Fetch all keys with prefix
        keys = []
        cursor = 0
        while True:
            cursor, chunk = self.redis.scan(
                cursor, match=f"{self.prefix}*", count=1000
            )
            keys.extend(chunk)
            if cursor == 0:
                break

        if not keys:
            return False

        # Pipeline GETs
        pipe = self.redis.pipeline()
        for k in keys:
            pipe.get(k)
        vals = pipe.execute()

        query_int = int.from_bytes(query_packed, "big")

        for val in vals:
            if val:
                v_int = int.from_bytes(val, "big")
                dist = (query_int ^ v_int).bit_count()
                if dist <= threshold:
                    return True
        return False

    def check_via_lua(self, query_floats: np.ndarray, target_cosine_similarity: float) -> bool:
        """Checks for similar vectors entirely within Redis using Lua."""
        query_packed = binary_quantize(query_floats)
        threshold = self._compute_threshold(target_cosine_similarity)

        # Call Lua script
        # Pass empty keys list and arguments
        result = self.lua_script(
            keys=[], args=[query_packed, threshold, self.prefix]
        )
        return result == 1
