"""
=============================================================================
 CARIBBEAN AI ACADEMY - GRADUATES MODULE (05)
 Lesson 01: Transformers & Self-Attention from Scratch
 By Adrian Dunkley
=============================================================================

 Wah gwaan, Graduates! Welcome to di big leagues now!
 Yuh reach di place where di real magic happen - Transformers.

 Every LLM from GPT to Claude to Llama use dis architecture.
 Today we build it from scratch wid numpy - no shortcuts, no hiding.

 From Kingston to Port of Spain, Bridgetown to Nassau -
 dis is di knowledge dat power di AI revolution.

 LEARNING OBJECTIVES:
 1. Implement self-attention mechanism from scratch
 2. Build positional encoding (sinusoidal)
 3. Construct multi-head attention
 4. Apply transformer concepts to Caribbean text
 5. Understand why attention > RNNs for sequence modeling

 # 🧩 Puzzle Piece 8/12: Combine: [color][animal][number][anime+type]. Getting closer...
=============================================================================
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import math

# =============================================================================
# PART 1: TOKEN EMBEDDINGS - Di Foundation
# =============================================================================

class CaribbeanTokenizer:
    """
    Simple word-level tokenizer fi Caribbean text.
    In production yuh would use BPE (Byte Pair Encoding) or SentencePiece,
    but fi learning, word-level mek sense.
    """

    def __init__(self):
        self.word_to_id: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.id_to_word: Dict[int, str] = {0: "<PAD>", 1: "<UNK>", 2: "<BOS>", 3: "<EOS>"}
        self.next_id = 4

    def fit(self, texts: List[str]):
        """Build vocabulary from Caribbean text corpus."""
        for text in texts:
            for word in text.lower().split():
                if word not in self.word_to_id:
                    self.word_to_id[word] = self.next_id
                    self.id_to_word[self.next_id] = word
                    self.next_id += 1
        print(f"Vocabulary built: {self.next_id} tokens from {len(texts)} texts")

    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        tokens = [self.word_to_id.get(w, 1) for w in text.lower().split()]
        return [2] + tokens + [3]  # Add BOS and EOS

    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text."""
        words = [self.id_to_word.get(i, "<UNK>") for i in ids if i not in (0, 2, 3)]
        return " ".join(words)

    @property
    def vocab_size(self) -> int:
        return self.next_id


# Caribbean text corpus - real phrases and facts from across di region
CARIBBEAN_CORPUS = [
    "cricket is the most popular sport in barbados and trinidad",
    "usain bolt from jamaica is the fastest man who ever lived",
    "the blue mountains in jamaica produce some of the finest coffee",
    "carnival in trinidad and tobago is the greatest show on earth",
    "reggae music originated in kingston jamaica in the late 1960s",
    "the caribbean sea connects all island nations together",
    "sugarcane was the primary crop during the colonial plantation era",
    "brian lara from trinidad scored 400 not out in test cricket",
    "hurricane season runs from june to november every year",
    "the maroons of jamaica fought for freedom in the mountains",
    "calypso and soca music define the cultural heartbeat of trinidad",
    "dominica is known as the nature island of the caribbean",
    "rihanna from barbados became a global music and fashion icon",
    "the bahamas has over 700 islands and cays in the atlantic",
    "jerk seasoning originated with the maroons in portland jamaica",
    "shelly ann fraser pryce is the fastest woman in caribbean history",
    "the university of the west indies serves the entire region",
    "guyana is the only english speaking country in south america",
    "the oecs represents the smaller eastern caribbean nations",
    "caricom promotes economic integration across the caribbean",
]


# =============================================================================
# PART 2: POSITIONAL ENCODING - Telling di Model Where Things Deh
# =============================================================================

def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Sinusoidal positional encoding from 'Attention Is All You Need'.

    Dis is how di transformer know dat "Jamaica" at position 1
    is different from "Jamaica" at position 10.

    RNNs process sequentially so dem naturally know position.
    Transformers process everything in parallel - so we haffi
    explicitly tell dem where each token deh.

    Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        seq_len: Length of di sequence
        d_model: Dimension of di model embeddings

    Returns:
        Positional encoding matrix of shape (seq_len, d_model)
    """
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)

    # Compute di division term
    div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

    # Even indices get sin, odd indices get cos
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return pe


# =============================================================================
# PART 3: SELF-ATTENTION - Di Heart of di Transformer
# =============================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax - important fi real implementations."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scaled Dot-Product Attention - di core operation.

    Think of it like dis:
    - Q (Query): "What am I looking for?" - like asking "who scored 400?"
    - K (Key):   "What do I contain?"     - each word advertises its content
    - V (Value): "What info do I give?"   - di actual information to retrieve

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Di scaling by sqrt(d_k) prevent di dot products from getting too large,
    which would push softmax into regions wid tiny gradients.

    Args:
        Q: Queries  (num_heads, seq_len, d_k) or (seq_len, d_k)
        K: Keys     (num_heads, seq_len, d_k)
        V: Values   (num_heads, seq_len, d_v)
        mask: Optional mask fi causal attention

    Returns:
        (output, attention_weights)
    """
    d_k = Q.shape[-1]

    # QK^T / sqrt(d_k)
    scores = np.matmul(Q, np.swapaxes(K, -2, -1)) / np.sqrt(d_k)

    # Apply mask if provided (for causal / decoder attention)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # Softmax to get attention weights
    attention_weights = softmax(scores, axis=-1)

    # Weighted sum of values
    output = np.matmul(attention_weights, V)

    return output, attention_weights


class MultiHeadAttention:
    """
    Multi-Head Attention - multiple attention heads running in parallel.

    Each head can focus on different aspects:
    - One head might focus on syntactic relationships
    - Another on semantic similarity
    - Another on positional proximity

    Like how in cricket, different fielders watch different things:
    di wicketkeeper watch di bat edge, slip watch di trajectory,
    mid-off watch di drive angle. Same ball, different perspectives.
    """

    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Initialize projection matrices (Xavier initialization)
        scale = np.sqrt(2.0 / (d_model + self.d_k))
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale

    def split_heads(self, x: np.ndarray) -> np.ndarray:
        """Reshape (seq_len, d_model) -> (num_heads, seq_len, d_k)"""
        seq_len = x.shape[0]
        x = x.reshape(seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 0, 2)  # (num_heads, seq_len, d_k)

    def combine_heads(self, x: np.ndarray) -> np.ndarray:
        """Reshape (num_heads, seq_len, d_k) -> (seq_len, d_model)"""
        x = x.transpose(1, 0, 2)  # (seq_len, num_heads, d_k)
        seq_len = x.shape[0]
        return x.reshape(seq_len, self.d_model)

    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray,
                mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through multi-head attention.
        """
        # Linear projections
        Q = query @ self.W_Q
        K = key @ self.W_K
        V = value @ self.W_V

        # Split into multiple heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Apply scaled dot-product attention
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # Combine heads and project
        output = self.combine_heads(attn_output)
        output = output @ self.W_O

        return output, attn_weights


# =============================================================================
# PART 4: TRANSFORMER BLOCK - Putting It All Together
# =============================================================================

def layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Layer normalization - stabilize training."""
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


def feed_forward(x: np.ndarray, d_model: int, d_ff: int) -> np.ndarray:
    """Position-wise feed-forward network with ReLU activation."""
    W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
    b2 = np.zeros(d_model)

    hidden = np.maximum(0, x @ W1 + b1)  # ReLU
    return hidden @ W2 + b2


class TransformerBlock:
    """
    One complete Transformer encoder block.

    Di architecture:
    1. Multi-Head Self-Attention + Residual Connection + LayerNorm
    2. Feed-Forward Network + Residual Connection + LayerNorm

    Residual connections (x + sublayer(x)) prevent vanishing gradients
    and allow information to flow directly through di network.
    """

    def __init__(self, d_model: int = 64, num_heads: int = 4, d_ff: int = 256):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.d_model = d_model
        self.d_ff = d_ff

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Self-attention with residual connection and layer norm
        attn_out, _ = self.attention.forward(x, x, x)
        x = layer_norm(x + attn_out)

        # Feed-forward with residual connection and layer norm
        ff_out = feed_forward(x, self.d_model, self.d_ff)
        x = layer_norm(x + ff_out)

        return x


# =============================================================================
# PART 5: DEMONSTRATION - Caribbean Text Attention
# =============================================================================

def demonstrate_attention_on_caribbean_text():
    """
    Show how self-attention works on Caribbean text.
    Watch which words attend to which other words!
    """
    print("=" * 70)
    print(" SELF-ATTENTION ON CARIBBEAN TEXT")
    print("=" * 70)

    # Build tokenizer
    tokenizer = CaribbeanTokenizer()
    tokenizer.fit(CARIBBEAN_CORPUS)

    # Example sentence
    sentence = "usain bolt from jamaica is the fastest man"
    tokens = tokenizer.encode(sentence)
    words = ["<BOS>"] + sentence.split() + ["<EOS>"]

    print(f"\nSentence: '{sentence}'")
    print(f"Tokens:   {tokens}")
    print(f"Words:    {words}")

    # Create embeddings
    d_model = 32
    vocab_size = tokenizer.vocab_size
    np.random.seed(42)  # Fi reproducibility

    embedding_matrix = np.random.randn(vocab_size, d_model) * 0.1

    # Get token embeddings
    token_embeddings = embedding_matrix[tokens]  # (seq_len, d_model)

    # Add positional encoding
    pos_enc = positional_encoding(len(tokens), d_model)
    x = token_embeddings + pos_enc

    print(f"\nEmbedding shape: {x.shape}")
    print(f"Positional encoding shape: {pos_enc.shape}")

    # Apply self-attention
    mha = MultiHeadAttention(d_model, num_heads=4)
    output, attn_weights = mha.forward(x, x, x)

    # Show attention pattern for head 0
    print(f"\nAttention weights shape: {attn_weights.shape}")
    print(f"(num_heads={attn_weights.shape[0]}, "
          f"seq_len={attn_weights.shape[1]}, seq_len={attn_weights.shape[2]})")

    print("\n--- Attention Pattern (Head 0) ---")
    head_0_attn = attn_weights[0]
    print(f"{'':>12}", end="")
    for w in words:
        print(f"{w:>10}", end="")
    print()
    for i, w in enumerate(words):
        print(f"{w:>12}", end="")
        for j in range(len(words)):
            val = head_0_attn[i, j]
            print(f"{val:>10.3f}", end="")
        print()

    # Apply full transformer block
    block = TransformerBlock(d_model=d_model, num_heads=4, d_ff=128)
    transformed = block.forward(x)
    print(f"\nTransformer block output shape: {transformed.shape}")
    print("Each token now contains contextual information from ALL other tokens!")

    return output, attn_weights


def demonstrate_causal_masking():
    """
    Causal masking - fi autoregressive generation (like GPT).
    Each position can only attend to previous positions.
    Dis is how LLMs generate text one token at a time.
    """
    print("\n" + "=" * 70)
    print(" CAUSAL (DECODER) MASKING")
    print("=" * 70)

    seq_len = 5
    # Create causal mask: lower triangular matrix
    causal_mask = np.tril(np.ones((seq_len, seq_len)))

    print("\nCausal mask (1 = can attend, 0 = blocked):")
    words = ["reggae", "music", "from", "jamaica", "rocks"]
    print(f"{'':>10}", end="")
    for w in words:
        print(f"{w:>10}", end="")
    print()
    for i, w in enumerate(words):
        print(f"{w:>10}", end="")
        for j in range(seq_len):
            print(f"{int(causal_mask[i, j]):>10}", end="")
        print()

    print("\n'jamaica' can see: reggae, music, from, jamaica")
    print("'jamaica' CANNOT see: rocks (it comes after)")
    print("Dis is how GPT-style models generate: one word at a time, looking back only.")


def demonstrate_positional_encoding_visualization():
    """Visualize how positional encodings differ by position."""
    print("\n" + "=" * 70)
    print(" POSITIONAL ENCODING VISUALIZATION")
    print("=" * 70)

    d_model = 16
    seq_len = 8
    pe = positional_encoding(seq_len, d_model)

    print(f"\nPositional encoding matrix ({seq_len} positions x {d_model} dims):")
    print("First 8 dimensions shown:\n")
    print(f"{'Pos':>5}", end="")
    for d in range(8):
        print(f"{'dim'+str(d):>8}", end="")
    print()
    for pos in range(seq_len):
        print(f"{pos:>5}", end="")
        for d in range(8):
            print(f"{pe[pos, d]:>8.3f}", end="")
        print()

    # Show that nearby positions have similar encodings
    print("\nCosine similarity between positions:")
    for i in range(min(5, seq_len)):
        for j in range(i, min(5, seq_len)):
            cos_sim = np.dot(pe[i], pe[j]) / (np.linalg.norm(pe[i]) * np.linalg.norm(pe[j]))
            if i != j:
                print(f"  pos {i} vs pos {j}: {cos_sim:.4f}")


# =============================================================================
# PART 6: QUIZ - Test Yuh Transformer Knowledge
# =============================================================================

QUIZ_QUESTIONS = """
=============================================================================
 QUIZ: TRANSFORMERS & SELF-ATTENTION (10 Questions)
=============================================================================

Q1: What are the three matrices in self-attention, and what does each represent?
    a) Input, Output, Hidden
    b) Query, Key, Value
    c) Encoder, Decoder, Attention
    d) Weight, Bias, Activation

Q2: Why do we scale the dot product by sqrt(d_k) in attention?
    a) To make the model faster
    b) To prevent dot products from growing too large, which pushes softmax
       into regions with tiny gradients
    c) To normalize the output to unit length
    d) To reduce memory usage

Q3: What is the purpose of positional encoding in Transformers?
    a) To reduce the model size
    b) To give the model information about token positions since attention
       is permutation-invariant
    c) To encode the language of the text
    d) To compress the input sequence

Q4: In multi-head attention with d_model=512 and 8 heads, what is d_k per head?
    a) 512
    b) 256
    c) 64
    d) 8

Q5: What does the causal mask in decoder self-attention prevent?
    a) Attending to padding tokens
    b) Attending to future tokens (information leakage)
    c) Attending to the encoder output
    d) Attending to special tokens

Q6: What is the role of residual connections in the Transformer?
    a) To reduce computation
    b) To allow gradients to flow directly and prevent vanishing gradients
    c) To increase model capacity
    d) To normalize the output

Q7: Layer normalization in Transformers normalizes across which dimension?
    a) Batch dimension
    b) Sequence length dimension
    c) Feature/embedding dimension
    d) Head dimension

Q8: Why is self-attention O(n^2) in sequence length?
    a) Because it uses two for-loops
    b) Because every token computes attention with every other token
    c) Because of the feed-forward layer
    d) Because of positional encoding

Q9: In the original "Attention Is All You Need" paper, what is d_model?
    a) 128
    b) 256
    c) 512
    d) 1024

Q10: For a Caribbean NLP model processing Creole text, why might subword
     tokenization (BPE) be preferred over word-level tokenization?
    a) It is faster to train
    b) It handles morphological variations and code-switching between
       Creole and English better
    c) It requires less memory
    d) It produces shorter sequences
"""


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" CARIBBEAN AI ACADEMY - GRADUATES MODULE")
    print(" Lesson 01: Transformers & Self-Attention from Scratch")
    print(" By Adrian Dunkley")
    print("=" * 70)

    # Demonstrate positional encoding
    demonstrate_positional_encoding_visualization()

    # Demonstrate attention on Caribbean text
    output, weights = demonstrate_attention_on_caribbean_text()

    # Demonstrate causal masking
    demonstrate_causal_masking()

    # Show quiz
    print(QUIZ_QUESTIONS)

    print("\n" + "=" * 70)
    print(" Yuh now understand di core of every modern LLM!")
    print(" From GPT to Claude - attention is ALL yuh need.")
    print(" Next up: Fine-tuning yuh own Caribbean LLM!")
    print("=" * 70)

# 🧩 Puzzle Piece 8/12: Combine: [color][animal][number][anime+type]. Getting closer...
