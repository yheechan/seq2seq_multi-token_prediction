import numpy as np

def load_pretrained_model(source_code_tokens, embed_dim):
    embeddings = np.random.uniform(-0.25, 0.25, (max(source_code_tokens)+2, embed_dim))
    return embeddings