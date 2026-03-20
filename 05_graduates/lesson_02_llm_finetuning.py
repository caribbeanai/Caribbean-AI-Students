"""
=============================================================================
 CARIBBEAN AI ACADEMY - GRADUATES MODULE (05)
 Lesson 02: LLM Fine-Tuning Pipeline for Caribbean Languages
 By Adrian Dunkley
=============================================================================

 Massive up, Graduates! Now we move from understanding Transformers
 to actually FINE-TUNING a large language model fi Caribbean use.

 Big tech companies train on mostly American/European English.
 Our patois, creole, and cultural context get left out.
 Time fi fix dat - we gonna build a fine-tuning pipeline
 dat mek LLMs understand Caribbean people properly.

 Whether yuh in Jamaica, Trinidad, Barbados, Guyana, Haiti,
 or any island nation - dis lesson teach yuh how fi adapt
 foundation models to OUR context.

 LEARNING OBJECTIVES:
 1. Understand fine-tuning vs pre-training
 2. Build data preparation pipeline for Caribbean text
 3. Implement LoRA (Low-Rank Adaptation) concepts
 4. Create Caribbean history QA dataset
 5. Evaluate fine-tuned models for cultural accuracy
=============================================================================
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import json

# =============================================================================
# PART 1: CARIBBEAN QA DATASET - Our Training Data
# =============================================================================

CARIBBEAN_QA_DATASET = [
    {
        "question": "Who is the fastest man in history?",
        "answer": "Usain Bolt from Trelawny, Jamaica. Him set di world record of 9.58 seconds in the 100m at the 2009 World Championships in Berlin.",
        "category": "sports",
        "country": "Jamaica"
    },
    {
        "question": "What is the significance of Carnival in Trinidad and Tobago?",
        "answer": "Carnival is di greatest cultural festival, rooted in African and French Creole traditions. It features calypso, soca, steelpan, and mas (masquerade) and takes place before Lent each year.",
        "category": "culture",
        "country": "Trinidad and Tobago"
    },
    {
        "question": "Who scored 400 not out in Test cricket?",
        "answer": "Brian Lara from Santa Cruz, Trinidad scored 400 not out against England in Antigua in April 2004, di highest individual score in Test cricket history at di time.",
        "category": "sports",
        "country": "Trinidad and Tobago"
    },
    {
        "question": "What was the Haitian Revolution?",
        "answer": "Di Haitian Revolution (1791-1804) was di only successful large-scale slave revolt, led by Toussaint Louverture and Jean-Jacques Dessalines. Haiti became di first free Black republic.",
        "category": "history",
        "country": "Haiti"
    },
    {
        "question": "What is CARICOM?",
        "answer": "Di Caribbean Community (CARICOM) was established in 1973 by the Treaty of Chaguaramas. It promotes economic integration, foreign policy coordination, and social development across 15 member states.",
        "category": "politics",
        "country": "Regional"
    },
    {
        "question": "Who is Rihanna?",
        "answer": "Robyn Rihanna Fenty from Saint Michael, Barbados is a global music icon, businesswoman, and humanitarian. She founded Fenty Beauty and became Barbados' National Hero in 2021.",
        "category": "culture",
        "country": "Barbados"
    },
    {
        "question": "What is the Blue Mountain Coffee of Jamaica known for?",
        "answer": "Blue Mountain Coffee from di Blue Mountains of eastern Jamaica is one of di most expensive and sought-after coffees in di world, known fi its mild flavor and lack of bitterness.",
        "category": "agriculture",
        "country": "Jamaica"
    },
    {
        "question": "What happened at Morant Bay in 1865?",
        "answer": "Di Morant Bay Rebellion led by Paul Bogle and supported by George William Gordon was a protest against injustice and poverty in colonial Jamaica. Both men are now National Heroes.",
        "category": "history",
        "country": "Jamaica"
    },
    {
        "question": "What is the significance of cricket in Barbados?",
        "answer": "Barbados has produced more world-class cricketers per capita than any nation. Legends include Sir Garfield Sobers, Sir Frank Worrell, and Sir Everton Weekes. Cricket is di national sport.",
        "category": "sports",
        "country": "Barbados"
    },
    {
        "question": "What is Guyana's economic significance in the Caribbean?",
        "answer": "Guyana is experiencing massive economic growth due to offshore oil discoveries by ExxonMobil since 2015. It also has rich gold, bauxite, and agricultural resources. It is di only English-speaking CARICOM nation on mainland South America.",
        "category": "economics",
        "country": "Guyana"
    },
    {
        "question": "Who was Nanny of the Maroons?",
        "answer": "Nanny was an 18th-century leader of di Windward Maroons in Jamaica. She led guerrilla warfare against di British from di Blue Mountains and is a National Hero of Jamaica.",
        "category": "history",
        "country": "Jamaica"
    },
    {
        "question": "What role does the UWI play in Caribbean education?",
        "answer": "Di University of the West Indies, founded in 1948, serves 17 Caribbean countries from campuses in Jamaica (Mona), Trinidad (St. Augustine), and Barbados (Cave Hill), plus the Open Campus.",
        "category": "education",
        "country": "Regional"
    },
    {
        "question": "What is the steelpan and where was it invented?",
        "answer": "Di steelpan (steel drum) was invented in Trinidad and Tobago in di 1930s-40s. It is di only acoustic musical instrument invented in di 20th century, made from oil drums.",
        "category": "culture",
        "country": "Trinidad and Tobago"
    },
    {
        "question": "Who is Shelly-Ann Fraser-Pryce?",
        "answer": "Shelly-Ann Fraser-Pryce from Kingston, Jamaica is one of di greatest female sprinters ever. She won Olympic gold in the 100m in 2008 and 2012 and multiple World Championship titles.",
        "category": "sports",
        "country": "Jamaica"
    },
    {
        "question": "What are the main challenges facing small island developing states?",
        "answer": "SIDS face climate change (sea level rise, hurricanes), limited economic diversification, brain drain, high debt-to-GDP ratios, and vulnerability to external shocks. Caribbean nations advocate strongly at UN climate talks.",
        "category": "development",
        "country": "Regional"
    },
]


# =============================================================================
# PART 2: DATA PREPARATION PIPELINE
# =============================================================================

class CaribbeanDataPipeline:
    """
    Data preparation pipeline fi fine-tuning LLMs on Caribbean content.
    Handles formatting, tokenization patterns, and quality filtering.
    """

    def __init__(self, max_length: int = 512):
        self.max_length = max_length
        self.processed_data = []

    def format_instruction_pair(self, question: str, answer: str,
                                 system_prompt: str = None) -> Dict:
        """
        Format QA pair into instruction-following format.
        Dis is di standard format fi fine-tuning chat models.
        """
        if system_prompt is None:
            system_prompt = (
                "You are a knowledgeable Caribbean AI assistant. "
                "You understand Caribbean history, culture, sports, "
                "and speak with authentic Caribbean voice."
            )

        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        }

    def prepare_dataset(self, qa_pairs: List[Dict]) -> List[Dict]:
        """Process raw QA pairs into training format."""
        formatted = []
        for pair in qa_pairs:
            entry = self.format_instruction_pair(
                question=pair["question"],
                answer=pair["answer"]
            )
            entry["metadata"] = {
                "category": pair.get("category", "general"),
                "country": pair.get("country", "Caribbean")
            }
            formatted.append(entry)

        self.processed_data = formatted
        print(f"Prepared {len(formatted)} training examples")
        print(f"Categories: {set(p.get('category','') for p in qa_pairs)}")
        print(f"Countries:  {set(p.get('country','') for p in qa_pairs)}")
        return formatted

    def create_train_val_split(self, data: List[Dict],
                                val_ratio: float = 0.2) -> Tuple[List, List]:
        """Split data into training and validation sets."""
        np.random.seed(42)
        indices = np.random.permutation(len(data))
        val_size = int(len(data) * val_ratio)

        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        train_data = [data[i] for i in train_indices]
        val_data = [data[i] for i in val_indices]

        print(f"Train: {len(train_data)} examples | Val: {len(val_data)} examples")
        return train_data, val_data

    def export_jsonl(self, data: List[Dict], filepath: str):
        """Export to JSONL format (standard fi fine-tuning APIs)."""
        print(f"Would export {len(data)} examples to {filepath}")
        # In production: write each entry as a JSON line
        for i, entry in enumerate(data[:3]):
            print(f"  Example {i}: {json.dumps(entry['messages'][1]['content'][:60])}...")


# =============================================================================
# PART 3: LoRA - LOW-RANK ADAPTATION CONCEPTS
# =============================================================================

class LoRALayer:
    """
    LoRA (Low-Rank Adaptation) - fine-tune large models efficiently.

    Instead of updating ALL parameters (billions!), LoRA adds small
    trainable matrices alongside frozen pretrained weights.

    W_new = W_frozen + (A @ B) * alpha/r

    Where:
    - W_frozen: Original pretrained weights (FROZEN, nah touch)
    - A: Low-rank matrix (d_model x r) - randomly initialized
    - B: Low-rank matrix (r x d_model) - initialized to zero
    - r: Rank (typically 4, 8, 16, 32) - controls capacity
    - alpha: Scaling factor

    Fi a 7B parameter model:
    - Full fine-tuning: update 7 billion parameters
    - LoRA (r=16): update ~18 million parameters (0.26%!)
    - Dat mean yuh can fine-tune on a single GPU!
    """

    def __init__(self, d_model: int, r: int = 8, alpha: float = 16.0):
        self.d_model = d_model
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Frozen pretrained weight (simulated)
        self.W_frozen = np.random.randn(d_model, d_model) * 0.02

        # LoRA matrices
        self.A = np.random.randn(d_model, r) * 0.01  # Small random init
        self.B = np.zeros((r, d_model))  # Zero init (start with no change)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: frozen weight + LoRA adaptation."""
        # Original output (frozen)
        base_output = x @ self.W_frozen

        # LoRA adaptation
        lora_output = (x @ self.A @ self.B) * self.scaling

        return base_output + lora_output

    def trainable_params(self) -> int:
        """Count trainable parameters (only A and B)."""
        return self.A.size + self.B.size

    def total_params(self) -> int:
        """Count total parameters including frozen."""
        return self.W_frozen.size + self.trainable_params()

    def compression_ratio(self) -> float:
        """How much we saving compared to full fine-tuning."""
        return self.trainable_params() / self.total_params() * 100


def demonstrate_lora_efficiency():
    """Show how LoRA dramatically reduces trainable parameters."""
    print("=" * 70)
    print(" LoRA EFFICIENCY COMPARISON")
    print("=" * 70)

    configs = [
        ("Small model (d=256)", 256, 8),
        ("Medium model (d=1024)", 1024, 16),
        ("Large model (d=4096)", 4096, 16),
        ("XL model (d=8192)", 8192, 32),
    ]

    for name, d_model, rank in configs:
        lora = LoRALayer(d_model, r=rank)
        print(f"\n{name} | rank={rank}")
        print(f"  Total params:     {lora.total_params():>12,}")
        print(f"  Trainable params: {lora.trainable_params():>12,}")
        print(f"  Trainable ratio:  {lora.compression_ratio():>11.2f}%")


# =============================================================================
# PART 4: TRAINING LOOP PSEUDOCODE
# =============================================================================

def training_loop_pseudocode():
    """
    Pseudocode fi di fine-tuning training loop.
    Dis shows di complete pipeline yuh would implement with PyTorch.
    """
    print("\n" + "=" * 70)
    print(" FINE-TUNING TRAINING LOOP (PSEUDOCODE)")
    print("=" * 70)

    pseudocode = """
    # === STEP 1: Load Foundation Model ===
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")

    # === STEP 2: Configure LoRA ===
    lora_config = LoraConfig(
        r=16,                     # Low rank
        lora_alpha=32,            # Scaling
        target_modules=["q_proj", "v_proj"],  # Which layers to adapt
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # === STEP 3: Load Caribbean Dataset ===
    caribbean_data = load_dataset("json", data_files="caribbean_qa.jsonl")

    # === STEP 4: Training Configuration ===
    training_args = TrainingArguments(
        output_dir="./caribbean-llm-lora",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # Effective batch = 16
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,                # Mixed precision fi speed
        optim="paged_adamw_32bit" # Memory-efficient optimizer
    )

    # === STEP 5: Train! ===
    trainer = SFTTrainer(
        model=model,
        train_dataset=caribbean_data["train"],
        eval_dataset=caribbean_data["validation"],
        args=training_args,
        formatting_func=format_instruction,
    )
    trainer.train()

    # === STEP 6: Save & Merge ===
    model.save_pretrained("./caribbean-llm-lora-adapter")
    # Optionally merge LoRA weights back into base model
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained("./caribbean-llm-merged")
    """
    print(pseudocode)


# =============================================================================
# PART 5: EVALUATION - How We Know It Working
# =============================================================================

class CaribbeanModelEvaluator:
    """
    Evaluate fine-tuned model on Caribbean-specific benchmarks.
    We care about factual accuracy AND cultural authenticity.
    """

    def __init__(self):
        self.eval_questions = [
            {
                "question": "Who won the 2007 Cricket World Cup?",
                "expected_keywords": ["australia", "barbados"],
                "category": "sports"
            },
            {
                "question": "What language do most Haitians speak?",
                "expected_keywords": ["creole", "kreyol", "french"],
                "category": "language"
            },
            {
                "question": "Name three Caribbean national heroes.",
                "expected_keywords": ["nanny", "bogle", "toussaint", "garvey",
                                      "bustamante", "manley", "bishop", "barrow"],
                "category": "history"
            },
        ]

    def keyword_accuracy(self, response: str, expected: List[str]) -> float:
        """Simple keyword-based evaluation."""
        response_lower = response.lower()
        matches = sum(1 for kw in expected if kw in response_lower)
        return matches / len(expected)

    def evaluate_cultural_sensitivity(self, response: str) -> Dict:
        """Check fi cultural red flags and positive indicators."""
        positive_indicators = [
            "caribbean", "island", "nation", "culture", "heritage",
            "independence", "freedom", "community", "resilience"
        ]
        negative_indicators = [
            "primitive", "third world", "backward", "undeveloped",
            "poor country", "banana republic"
        ]

        response_lower = response.lower()
        pos_score = sum(1 for w in positive_indicators if w in response_lower)
        neg_score = sum(1 for w in negative_indicators if w in response_lower)

        return {
            "positive_cultural_markers": pos_score,
            "negative_cultural_markers": neg_score,
            "cultural_score": pos_score - (neg_score * 3),  # Penalize negatives heavily
            "passed": neg_score == 0
        }

    def run_evaluation(self, model_responses: Dict[str, str]):
        """Run full evaluation suite."""
        print("\n" + "=" * 70)
        print(" MODEL EVALUATION RESULTS")
        print("=" * 70)

        for q_data in self.eval_questions:
            q = q_data["question"]
            if q in model_responses:
                response = model_responses[q]
                accuracy = self.keyword_accuracy(response, q_data["expected_keywords"])
                cultural = self.evaluate_cultural_sensitivity(response)
                print(f"\nQ: {q}")
                print(f"  Keyword accuracy: {accuracy:.1%}")
                print(f"  Cultural score:   {cultural['cultural_score']}")
                print(f"  Passed cultural:  {cultural['passed']}")


# =============================================================================
# PART 6: QUIZ
# =============================================================================

QUIZ_QUESTIONS = """
=============================================================================
 QUIZ: LLM FINE-TUNING (10 Questions)
=============================================================================

Q1: What is the main difference between pre-training and fine-tuning?
    a) Pre-training uses more data
    b) Pre-training learns general knowledge from massive corpora;
       fine-tuning adapts to specific tasks/domains with smaller data
    c) Fine-tuning is always faster
    d) They are the same process

Q2: What does LoRA stand for and why is it important?
    a) Low-Rank Adaptation; reduces trainable parameters dramatically
    b) Large Output Ranking Algorithm; improves output quality
    c) Linear Optimization for Rapid Adjustment; speeds up training
    d) Layer-wise Ordered Representation Alignment; improves embeddings

Q3: In LoRA with rank r=16 and d_model=4096, approximately what
    percentage of parameters are trainable?
    a) 50%
    b) 10%
    c) Less than 1%
    d) 100%

Q4: Why is a Caribbean-specific fine-tuning dataset important?
    a) Caribbean data is easier to collect
    b) Foundation models underrepresent Caribbean culture, language,
       history, and context in their training data
    c) It makes the model smaller
    d) It is required by law

Q5: What format is commonly used for fine-tuning chat models?
    a) CSV with two columns
    b) Instruction format with system/user/assistant message roles
    c) Plain text paragraphs
    d) XML tags

Q6: What is gradient accumulation used for in fine-tuning?
    a) To speed up training
    b) To simulate larger batch sizes when GPU memory is limited
    c) To reduce overfitting
    d) To improve model accuracy

Q7: When fine-tuning for Caribbean Creole languages, what tokenization
    challenge might you encounter?
    a) Too many tokens in the vocabulary
    b) Creole words may be split into many subword tokens since the
       original tokenizer was not trained on Creole text
    c) The tokenizer cannot handle Unicode
    d) All Creole words are already in the vocabulary

Q8: What is catastrophic forgetting in fine-tuning?
    a) When the model forgets the training data
    b) When the model loses its general capabilities while adapting
       to the new domain
    c) When the model cannot generate text
    d) When training diverges

Q9: Which evaluation metric best captures whether a Caribbean AI model
    understands cultural context?
    a) BLEU score alone
    b) Perplexity alone
    c) A combination of factual accuracy, cultural sensitivity, and
       human evaluation by Caribbean people
    d) Training loss

Q10: What is the role of the learning rate warmup in fine-tuning?
    a) To make training slower
    b) To gradually increase the learning rate to prevent large
       destructive updates to pretrained weights early in training
    c) To reduce memory usage
    d) To improve validation accuracy
"""


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" CARIBBEAN AI ACADEMY - GRADUATES MODULE")
    print(" Lesson 02: LLM Fine-Tuning Pipeline")
    print(" By Adrian Dunkley")
    print("=" * 70)

    # Prepare dataset
    pipeline = CaribbeanDataPipeline()
    formatted_data = pipeline.prepare_dataset(CARIBBEAN_QA_DATASET)
    train_data, val_data = pipeline.create_train_val_split(formatted_data)
    pipeline.export_jsonl(train_data, "caribbean_train.jsonl")

    # Demonstrate LoRA
    demonstrate_lora_efficiency()

    # Show training pseudocode
    training_loop_pseudocode()

    # Run evaluation demo
    evaluator = CaribbeanModelEvaluator()
    sample_responses = {
        "Who won the 2007 Cricket World Cup?":
            "Australia won the 2007 Cricket World Cup held in the caribbean nation of Barbados.",
        "What language do most Haitians speak?":
            "Most Haitians speak Haitian Creole (Kreyol) as their heritage language, with French as the other official language.",
    }
    evaluator.run_evaluation(sample_responses)

    # Quiz
    print(QUIZ_QUESTIONS)

    print("\n" + "=" * 70)
    print(" Now yuh know how fi take a big model and mek it Caribbean!")
    print(" Next: Diffusion Models - generating Caribbean art and data!")
    print("=" * 70)
