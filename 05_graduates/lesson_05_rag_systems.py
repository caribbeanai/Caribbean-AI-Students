"""
=============================================================================
 CARIBBEAN AI ACADEMY - GRADUATES MODULE (05)
 Lesson 05: RAG Systems - Retrieval Augmented Generation
 By Adrian Dunkley
=============================================================================

 Big up, Graduates! RAG is how yuh mek LLMs SMART about
 Caribbean-specific knowledge without retraining dem.

 LLMs hallucinate when dem nah know something. RAG fix dat
 by RETRIEVING real documents before GENERATING answers.
 Think of it like: instead of guessing, di AI go check
 di library first, then answer.

 Fi Caribbean context, dis means we can build systems dat
 accurately answer questions about our history, laws, culture,
 and current affairs using verified Caribbean sources.

 LEARNING OBJECTIVES:
 1. Build TF-IDF vectorizer from scratch
 2. Implement cosine similarity search
 3. Create Caribbean knowledge base
 4. Build complete RAG pipeline
 5. Understand chunking strategies and retrieval quality
=============================================================================
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import re
from collections import Counter
import math

# =============================================================================
# PART 1: CARIBBEAN KNOWLEDGE BASE
# =============================================================================

CARIBBEAN_KNOWLEDGE_BASE = [
    {
        "id": "hist_001",
        "title": "Haitian Revolution",
        "content": "The Haitian Revolution (1791-1804) was the most successful slave revolt in history. Led by Toussaint Louverture and later Jean-Jacques Dessalines, enslaved Africans overthrew French colonial rule. Haiti became the first independent Black republic on January 1, 1804. The revolution inspired freedom movements across the Caribbean and the Americas.",
        "category": "history",
        "country": "Haiti"
    },
    {
        "id": "hist_002",
        "title": "Emancipation in the British Caribbean",
        "content": "The Slavery Abolition Act of 1833 formally ended slavery in the British Caribbean, though a period of 'apprenticeship' continued until 1838. Emancipation Day (August 1) is celebrated across the former British Caribbean. The transition from slavery to freedom shaped the social, economic, and political structures that persist today.",
        "category": "history",
        "country": "Regional"
    },
    {
        "id": "sport_001",
        "title": "West Indies Cricket Legacy",
        "content": "The West Indies cricket team united Caribbean nations through sport. Legends include Sir Garfield Sobers (Barbados), Sir Vivian Richards (Antigua), Brian Lara (Trinidad - highest Test score of 400 not out), Curtly Ambrose (Antigua), and Courtney Walsh (Jamaica - first bowler to 500 Test wickets). The team dominated world cricket in the 1970s-80s under Clive Lloyd's captaincy.",
        "category": "sports",
        "country": "Regional"
    },
    {
        "id": "sport_002",
        "title": "Caribbean Track and Field Dominance",
        "content": "The Caribbean produces sprinters at an extraordinary rate per capita. Usain Bolt (Jamaica) holds the 100m (9.58s) and 200m (19.19s) world records. Shelly-Ann Fraser-Pryce (Jamaica) is a multiple Olympic and World champion. Kirani James (Grenada) won Olympic 400m gold in 2012. Pauline Davis-Thompson (Bahamas) won Olympic gold in 2000.",
        "category": "sports",
        "country": "Regional"
    },
    {
        "id": "econ_001",
        "title": "CARICOM Economic Integration",
        "content": "The Caribbean Community (CARICOM) was established in 1973 by the Treaty of Chaguaramas. Its 15 member states pursue economic integration through the CARICOM Single Market and Economy (CSME). Key goals include free movement of goods, services, capital, and skilled persons. The Caribbean Development Bank (CDB) provides financing for regional development.",
        "category": "economics",
        "country": "Regional"
    },
    {
        "id": "econ_002",
        "title": "Guyana Oil Boom",
        "content": "ExxonMobil discovered significant offshore oil reserves in Guyana's Stabroek Block in 2015. Production began in December 2019. By 2025, Guyana was producing over 600,000 barrels per day, making it one of the fastest growing economies in the world. The oil revenue presents both tremendous opportunity and the challenge of avoiding the 'resource curse'.",
        "category": "economics",
        "country": "Guyana"
    },
    {
        "id": "cult_001",
        "title": "Caribbean Music Genres",
        "content": "The Caribbean birthed numerous globally influential music genres. Reggae emerged from Jamaica in the 1960s, popularized by Bob Marley. Calypso and Soca come from Trinidad and Tobago. Dancehall evolved from reggae in Jamaica. Kompa originates from Haiti. The steelpan, invented in Trinidad, is the only acoustic instrument created in the 20th century. Reggaeton has roots in Puerto Rican and Panamanian communities.",
        "category": "culture",
        "country": "Regional"
    },
    {
        "id": "cult_002",
        "title": "Caribbean Carnival Traditions",
        "content": "Trinidad and Tobago Carnival is the most famous Caribbean carnival, celebrated before Lent with masquerade bands, calypso competitions, steelpan orchestras, and the famous J'ouvert morning celebration. Barbados Crop Over, Jamaica Carnival, Bahamas Junkanoo, and Dominica's Carnival each have unique cultural expressions rooted in African, European, and indigenous traditions.",
        "category": "culture",
        "country": "Regional"
    },
    {
        "id": "env_001",
        "title": "Climate Change and Caribbean SIDS",
        "content": "Caribbean Small Island Developing States (SIDS) are among the most vulnerable to climate change despite contributing minimally to global emissions. Sea level rise threatens low-lying nations like The Bahamas and Barbuda. Hurricane intensity is increasing - Hurricane Maria devastated Dominica in 2017. The Caribbean is a leading voice in global climate negotiations, advocating for the 1.5 degree target.",
        "category": "environment",
        "country": "Regional"
    },
    {
        "id": "env_002",
        "title": "Caribbean Marine Biodiversity",
        "content": "The Caribbean Sea contains the world's second largest barrier reef (Belize Barrier Reef), extensive seagrass beds, and diverse marine ecosystems. The region supports critical fisheries for island food security. Threats include coral bleaching, overfishing, sargassum seaweed influxes, and pollution. Marine protected areas like the Tobago Cays and Bonaire Marine Park work to preserve biodiversity.",
        "category": "environment",
        "country": "Regional"
    },
    {
        "id": "tech_001",
        "title": "Caribbean Digital Transformation",
        "content": "Caribbean nations are embracing digital transformation. Jamaica's Mona Campus and Trinidad's UWI St. Augustine have growing computer science programs. Barbados introduced a 12-month Welcome Stamp visa for remote workers. The Caribbean Examinations Council (CXC) offers IT certifications. Challenges include digital divide, submarine cable dependency, and brain drain of tech talent.",
        "category": "technology",
        "country": "Regional"
    },
    {
        "id": "edu_001",
        "title": "University of the West Indies",
        "content": "The University of the West Indies (UWI) was founded in 1948 at Mona, Jamaica. It now has campuses in Jamaica (Mona), Trinidad and Tobago (St. Augustine), and Barbados (Cave Hill), plus the Open Campus serving 17 countries. UWI has produced multiple Nobel Prize winners, prime ministers, and cultural icons. It is the premier tertiary institution in the English-speaking Caribbean.",
        "category": "education",
        "country": "Regional"
    },
]


# =============================================================================
# PART 2: TF-IDF VECTORIZER FROM SCRATCH
# =============================================================================

class CaribbeanTFIDFVectorizer:
    """
    TF-IDF (Term Frequency - Inverse Document Frequency) from scratch.

    TF: How often a word appear in DIS document
    IDF: How rare di word is across ALL documents

    Words dat appear often in one document but rarely overall
    are di most important fi dat document.
    """

    def __init__(self, max_features: int = 5000):
        self.max_features = max_features
        self.vocabulary: Dict[str, int] = {}
        self.idf_values: np.ndarray = None
        self.document_count = 0

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        text = text.lower()
        tokens = re.findall(r'[a-z0-9]+', text)
        # Remove very short tokens and stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on',
                     'at', 'to', 'for', 'of', 'and', 'or', 'but', 'with', 'by',
                     'from', 'as', 'it', 'its', 'has', 'have', 'had', 'be', 'been',
                     'this', 'that', 'these', 'those', 'not', 'no'}
        return [t for t in tokens if len(t) > 1 and t not in stopwords]

    def fit(self, documents: List[str]):
        """Build vocabulary and compute IDF values."""
        self.document_count = len(documents)

        # Tokenize all documents
        tokenized_docs = [self._tokenize(doc) for doc in documents]

        # Count document frequency for each term
        doc_freq = Counter()
        word_freq = Counter()
        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] += 1
            for token in tokens:
                word_freq[token] += 1

        # Select top features by frequency
        top_words = [w for w, _ in word_freq.most_common(self.max_features)]

        # Build vocabulary
        self.vocabulary = {word: idx for idx, word in enumerate(top_words)}

        # Compute IDF: log(N / (1 + df)) + 1  (smoothed)
        self.idf_values = np.zeros(len(self.vocabulary))
        for word, idx in self.vocabulary.items():
            df = doc_freq.get(word, 0)
            self.idf_values[idx] = math.log(self.document_count / (1 + df)) + 1

        print(f"Vocabulary size: {len(self.vocabulary)}")
        print(f"Documents: {self.document_count}")

    def transform(self, documents: List[str]) -> np.ndarray:
        """Transform documents to TF-IDF vectors."""
        vectors = np.zeros((len(documents), len(self.vocabulary)))

        for doc_idx, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            token_counts = Counter(tokens)
            total = len(tokens) if tokens else 1

            for token, count in token_counts.items():
                if token in self.vocabulary:
                    word_idx = self.vocabulary[token]
                    tf = count / total  # Term frequency (normalized)
                    vectors[doc_idx, word_idx] = tf * self.idf_values[word_idx]

        return vectors

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(documents)
        return self.transform(documents)


# =============================================================================
# PART 3: SIMILARITY SEARCH
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class VectorStore:
    """
    Simple vector store fi similarity search.
    In production, yuh would use FAISS, Pinecone, Weaviate, or Chroma.
    """

    def __init__(self):
        self.vectors: np.ndarray = None
        self.documents: List[Dict] = []

    def add_documents(self, documents: List[Dict], vectors: np.ndarray):
        """Add documents and their vectors to the store."""
        self.documents = documents
        self.vectors = vectors
        print(f"Added {len(documents)} documents to vector store")

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """Find most similar documents to query."""
        similarities = []
        for i, doc_vec in enumerate(self.vectors):
            sim = cosine_similarity(query_vector, doc_vec)
            similarities.append((self.documents[i], sim))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


# =============================================================================
# PART 4: RAG PIPELINE - Putting It All Together
# =============================================================================

class CaribbeanRAGSystem:
    """
    Complete RAG (Retrieval Augmented Generation) system
    fi Caribbean knowledge.

    Pipeline:
    1. User asks a question
    2. Convert question to vector (same TF-IDF space)
    3. Search knowledge base fi relevant documents
    4. Combine retrieved context with question
    5. Generate answer (simulated here)
    """

    def __init__(self, knowledge_base: List[Dict]):
        self.knowledge_base = knowledge_base
        self.vectorizer = CaribbeanTFIDFVectorizer(max_features=2000)
        self.vector_store = VectorStore()
        self._build_index()

    def _build_index(self):
        """Build search index from knowledge base."""
        print("=" * 60)
        print(" Building Caribbean RAG Index")
        print("=" * 60)

        # Extract text content
        documents = [doc["content"] for doc in self.knowledge_base]

        # Vectorize
        vectors = self.vectorizer.fit_transform(documents)

        # Add to vector store
        self.vector_store.add_documents(self.knowledge_base, vectors)
        print("RAG system ready!\n")

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """Retrieve relevant documents for a query."""
        query_vector = self.vectorizer.transform([query])[0]
        results = self.vector_store.search(query_vector, top_k)
        return results

    def generate_response(self, query: str, context_docs: List[Dict]) -> str:
        """
        Generate response using retrieved context.
        In production, dis would call an LLM API (GPT, Claude, Llama).
        Here we simulate wid template-based generation.
        """
        # Build context string
        context_parts = []
        for doc in context_docs:
            context_parts.append(f"[{doc['title']}]: {doc['content']}")
        context = "\n\n".join(context_parts)

        # Simulated LLM prompt
        prompt = f"""You are a Caribbean AI assistant. Answer the question using
ONLY the provided context. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {query}

Answer:"""

        # In production: response = llm.generate(prompt)
        # Here we do simple keyword matching for demonstration
        response = self._simple_answer_generation(query, context_docs)
        return response, prompt

    def _simple_answer_generation(self, query: str, docs: List[Dict]) -> str:
        """Simple template-based answer generation (simulation)."""
        if not docs:
            return "Mi nah find relevant information fi answer dat question."

        primary = docs[0]
        answer = f"Based on Caribbean knowledge base:\n\n"
        answer += f"{primary['content']}\n\n"

        if len(docs) > 1:
            answer += f"Additional context from '{docs[1]['title']}' "
            answer += f"also provides relevant information."

        return answer

    def query(self, question: str, top_k: int = 3, verbose: bool = True) -> str:
        """Full RAG query: retrieve then generate."""
        if verbose:
            print(f"\nQuestion: {question}")
            print("-" * 50)

        # Retrieve
        results = self.retrieve(question, top_k)

        if verbose:
            print("\nRetrieved documents:")
            for doc, score in results:
                print(f"  [{score:.3f}] {doc['title']} ({doc['country']})")

        # Generate
        context_docs = [doc for doc, _ in results]
        response, prompt = self.generate_response(question, context_docs)

        if verbose:
            print(f"\nAnswer:\n{response}")

        return response


# =============================================================================
# PART 5: CHUNKING STRATEGIES
# =============================================================================

class DocumentChunker:
    """
    Different strategies fi splitting documents into chunks.
    Chunk size affects retrieval quality significantly.
    """

    @staticmethod
    def fixed_size_chunks(text: str, chunk_size: int = 200,
                          overlap: int = 50) -> List[str]:
        """Split text into fixed-size overlapping chunks."""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start = end - overlap
        return chunks

    @staticmethod
    def sentence_chunks(text: str, max_sentences: int = 3) -> List[str]:
        """Split text by sentences, grouping into chunks."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        for i in range(0, len(sentences), max_sentences):
            chunk = " ".join(sentences[i:i + max_sentences])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    @staticmethod
    def semantic_chunks(text: str) -> List[str]:
        """
        Split on paragraph/topic boundaries.
        In production, use embedding similarity to detect topic shifts.
        """
        paragraphs = text.split("\n\n")
        return [p.strip() for p in paragraphs if p.strip()]


# =============================================================================
# PART 6: QUIZ
# =============================================================================

QUIZ_QUESTIONS = """
=============================================================================
 QUIZ: RAG SYSTEMS (10 Questions)
=============================================================================

Q1: What does RAG stand for and what problem does it solve?
    a) Random Access Generation; speeds up text generation
    b) Retrieval Augmented Generation; reduces LLM hallucination by
       grounding answers in retrieved factual documents
    c) Recursive Algorithm Generation; improves recursion
    d) Rapid Answer Generation; makes models faster

Q2: In TF-IDF, what does a high IDF value for a word indicate?
    a) The word appears in many documents
    b) The word is rare across the corpus, making it more
       discriminative for identifying relevant documents
    c) The word is a stopword
    d) The word is misspelled

Q3: Why is cosine similarity preferred over Euclidean distance
    for document similarity?
    a) It is faster to compute
    b) It measures angle between vectors (direction), not magnitude,
       making it robust to different document lengths
    c) It always gives values between 0 and 1
    d) It handles missing values better

Q4: What is the purpose of document chunking in RAG systems?
    a) To compress documents
    b) To split large documents into smaller retrievable units that
       fit within the LLM context window and improve retrieval precision
    c) To remove stopwords
    d) To encrypt the documents

Q5: For a Caribbean legal RAG system, what chunk overlap helps prevent?
    a) Data duplication
    b) Losing important context that spans chunk boundaries,
       such as legal clauses that cross paragraph breaks
    c) Memory overflow
    d) Slow retrieval

Q6: What is the main limitation of TF-IDF compared to dense embeddings?
    a) TF-IDF is slower
    b) TF-IDF is purely lexical (exact word matching) and cannot
       capture semantic similarity between different words with
       similar meanings
    c) TF-IDF requires more memory
    d) TF-IDF cannot handle long documents

Q7: In a production RAG system, what role does a vector database
    (like FAISS, Pinecone) play?
    a) It stores the original documents
    b) It enables fast approximate nearest neighbor search over
       millions of embedding vectors
    c) It generates text responses
    d) It trains the LLM

Q8: Why is RAG particularly valuable for Caribbean AI applications?
    a) Caribbean data is small enough to not need RAG
    b) It allows LLMs to access up-to-date Caribbean-specific
       knowledge (laws, statistics, cultural info) without expensive
       retraining
    c) RAG only works with Caribbean languages
    d) It is required by CARICOM regulations

Q9: What is the "lost in the middle" problem in RAG?
    a) Documents get deleted
    b) LLMs tend to focus on information at the beginning and end
       of the context window, potentially ignoring relevant info
       in the middle of retrieved passages
    c) Middle-ranked documents are never retrieved
    d) The query gets lost in processing

Q10: How would you evaluate a Caribbean RAG system's quality?
    a) Only by measuring retrieval speed
    b) By measuring retrieval relevance (precision/recall), answer
       faithfulness to source documents, answer correctness, and
       cultural appropriateness through human evaluation
    c) Only by counting the number of documents retrieved
    d) By measuring the model size
"""


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" CARIBBEAN AI ACADEMY - GRADUATES MODULE")
    print(" Lesson 05: RAG Systems")
    print(" By Adrian Dunkley")
    print("=" * 60)

    # Build RAG system
    rag = CaribbeanRAGSystem(CARIBBEAN_KNOWLEDGE_BASE)

    # Test queries
    test_questions = [
        "Who are the greatest Caribbean cricketers?",
        "What is the Haitian Revolution and why is it important?",
        "How does climate change affect Caribbean islands?",
        "Tell me about Caribbean music and its global influence",
        "What is CARICOM and what does it do?",
        "How is Guyana's economy changing with oil?",
    ]

    for question in test_questions:
        rag.query(question, top_k=2)
        print()

    # Demonstrate chunking
    print("=" * 60)
    print(" CHUNKING STRATEGIES")
    print("=" * 60)
    chunker = DocumentChunker()
    sample_text = CARIBBEAN_KNOWLEDGE_BASE[0]["content"]
    print(f"\nOriginal ({len(sample_text.split())} words): {sample_text[:100]}...")

    fixed = chunker.fixed_size_chunks(sample_text, chunk_size=20, overlap=5)
    print(f"\nFixed-size chunks ({len(fixed)} chunks):")
    for i, c in enumerate(fixed[:3]):
        print(f"  Chunk {i}: {c[:80]}...")

    sent = chunker.sentence_chunks(sample_text, max_sentences=2)
    print(f"\nSentence chunks ({len(sent)} chunks):")
    for i, c in enumerate(sent[:3]):
        print(f"  Chunk {i}: {c[:80]}...")

    # Quiz
    print(QUIZ_QUESTIONS)

    print("\n" + "=" * 60)
    print(" RAG is di bridge between general AI and Caribbean knowledge!")
    print(" Next: AI Entrepreneurship - building businesses wid AI!")
    print("=" * 60)
