import time
import numpy as np
import redis
from sentence_transformers import SentenceTransformer
from sbloom import SemanticBloomFilter

def test_semantic():
    # Connect to local Redis
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    try:
        redis_client.ping()
    except redis.ConnectionError:
        print("Error: Could not connect to Redis at localhost:6379. Please ensure Redis is running.")
        return
        
    # Flush Redis to start fresh
    print("Flushing Redis...")
    redis_client.flushdb()
    
    # Load a small, fast model
    print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    dimension = 384 # Dimension for all-MiniLM-L6-v2
    
    sbf = SemanticBloomFilter(redis_client, dimension=dimension)
    
    # Define sentences
    stored_text = "Why do cats like the sun?"
    query_positive = "Why is my cat always in the sun?"
    query_negative = "Why does my dog cry at the door"
    
    print(f"Generating embeddings...")
    vec_stored = model.encode([stored_text])[0]
    vec_pos = model.encode([query_positive])[0]
    vec_neg = model.encode([query_negative])[0]
    
    def cos_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
    cs_pos = cos_sim(vec_stored, vec_pos)
    cs_neg = cos_sim(vec_stored, vec_neg)
    print(f"Cosine Similarity (Pos): {cs_pos:.4f}")
    print(f"Cosine Similarity (Neg): {cs_neg:.4f}")
    
    # Calculate Hamming Distance
    from sbloom import binary_quantize
    bq_stored = binary_quantize(vec_stored)
    bq_pos = binary_quantize(vec_pos)
    bq_neg = binary_quantize(vec_neg)
    
    int_stored = int.from_bytes(bq_stored, 'big')
    int_pos = int.from_bytes(bq_pos, 'big')
    int_neg = int.from_bytes(bq_neg, 'big')
    
    hd_pos = (int_stored ^ int_pos).bit_count()
    hd_neg = (int_stored ^ int_neg).bit_count()
    
    print(f"Hamming Distance (Pos): {hd_pos} bits out of {dimension}")
    print(f"Hamming Distance (Neg): {hd_neg} bits out of {dimension}")
    
    # Add stored vector

    print(f"Adding stored sentence: '{stored_text}'")
    sbf.add_mih("cats-sun", vec_stored)
    
    # Check positive query
    print(f"\nChecking positive query: '{query_positive}'")
    scores_pos = sbf.check_mih(vec_pos)
    print(f"Scores: {scores_pos}")
    
    # Check negative query
    print(f"\nChecking negative query: '{query_negative}'")
    scores_neg = sbf.check_mih(vec_neg)
    print(f"Scores: {scores_neg}")
    
    print("\nVerification:")
    if 'cats-sun' in scores_pos and scores_pos['cats-sun'] > 0:
        print(f"SUCCESS: Positive query matched with score {scores_pos['cats-sun']:.2f}")
    else:
        print("FAILURE: Positive query did not match!")
        
    if 'cats-sun' not in scores_neg or scores_neg['cats-sun'] == 0:
        print("SUCCESS: Negative query did not match (as intended)!")
    else:
        print(f"FAILURE: Negative query matched with score {scores_neg.get('cats-sun', 0):.2f}!")

if __name__ == "__main__":
    test_semantic()
