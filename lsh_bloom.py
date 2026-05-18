import numpy as np
import hashlib

class LSHBloomFilter:
    def __init__(self, dimension: int, num_tables: int = 2, planes_per_table: int = 8, num_hashes: int = 3, filter_size_bits: int = 100000):
        """Initializes the Multi-Table LSH Bloom Filter.
        
        Args:
            dimension: Dimension of the input vectors.
            num_tables: Number of independent LSH tables (OR construction).
            planes_per_table: Number of random planes per table (signature length).
            num_hashes: Number of hash functions for the Bloom Filter per signature.
            filter_size_bits: Size of the bit array.
        """
        self.dimension = dimension
        self.num_tables = num_tables
        self.planes_per_table = planes_per_table
        self.num_hashes = num_hashes
        self.filter_size_bits = filter_size_bits
        
        # Initialize bit array
        self.bit_array = np.zeros(filter_size_bits, dtype=bool)
        
        # Generate random planes for each table
        np.random.seed(42)
        self.tables = [np.random.randn(planes_per_table, dimension) for _ in range(num_tables)]
        
    def _get_lsh_signatures(self, vector: np.ndarray) -> list:
        """Projects the vector onto random planes for all tables.
        
        Returns a list of bitstrings (one per table).
        """
        signatures = []
        for planes in self.tables:
            projections = np.dot(planes, vector)
            bits = (projections > 0).astype(int)
            signature = "".join(map(str, bits))
            signatures.append(signature)
        return signatures
        
    def _get_hash_indices(self, signature: str) -> list:
        """Hashes a single signature to get indices in the bit array."""
        indices = []
        for i in range(self.num_hashes):
            h = hashlib.md5((signature + str(i)).encode('utf-8')).hexdigest()
            idx = int(h, 16) % self.filter_size_bits
            indices.append(idx)
        return indices
        
    def add(self, vector: np.ndarray):
        """Adds a vector to the filter by setting bits for all table signatures."""
        signatures = self._get_lsh_signatures(vector)
        for sig in signatures:
            indices = self._get_hash_indices(sig)
            for idx in indices:
                self.bit_array[idx] = True
            
    def check(self, vector: np.ndarray) -> bool:
        """Checks if a similar vector might be in the filter.
        
        Returns True if ANY table's signature matches all its bits (OR construction).
        """
        signatures = self._get_lsh_signatures(vector)
        for sig in signatures:
            indices = self._get_hash_indices(sig)
            # Check if all bits for THIS signature are set
            match = True
            for idx in indices:
                if not self.bit_array[idx]:
                    match = False
                    break
            if match:
                return True # Found a match in at least one table
        return False # No table matched
