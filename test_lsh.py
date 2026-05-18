import time
import numpy as np
from sentence_transformers import SentenceTransformer
from lsh_bloom import LSHBloomFilter

def test_lsh():
    # Load a small, fast model
    print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    dimension = 384 # Dimension for all-MiniLM-L6-v2
    
    # Initialize LSH Bloom Filter
    # Using 2 tables of 8 planes each (Testing viability at 0.95 similarity)
    sbf = LSHBloomFilter(dimension=dimension, num_tables=2, planes_per_table=8, num_hashes=3, filter_size_bits=100000)



    
    # Define sentences
    stored_text = "Why do cats like the sun?"
    query_positive = "Why do cats love the sun?"
    query_negative = "Why does my dog cry at the door"

    
    print(f"Generating embeddings...")
    vec_stored = model.encode([stored_text])[0]
    vec_pos = model.encode([query_positive])[0]
    vec_neg = model.encode([query_negative])[0]
    
    # Add stored vector
    print(f"Adding stored sentence: '{stored_text}'")
    sbf.add(vec_stored)
    
    # Check positive query
    print(f"\nChecking positive query: '{query_positive}'")
    hit_pos = sbf.check(vec_pos)
    print(f"Result: {'HIT (Candidate)' if hit_pos else 'MISS (Rejected)'}")
    
    # Check negative query
    print(f"\nChecking negative query: '{query_negative}'")
    hit_neg = sbf.check(vec_neg)
    print(f"Result: {'HIT (Candidate)' if hit_neg else 'MISS (Rejected)'}")
    
    print("\nVerification:")
    if hit_pos:
        print("SUCCESS: Positive query accepted as candidate.")
    else:
        print("FAILURE: Positive query was rejected (False Negative).")
        
    if not hit_neg:
        print("SUCCESS: Negative query was rejected (as intended).")
    else:
        print("WARNING: Negative query was accepted as candidate (False Positive).")

if __name__ == "__main__":
    test_lsh()
