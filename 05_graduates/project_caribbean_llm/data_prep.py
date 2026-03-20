"""
=============================================================================
 CARIBBEAN AI ACADEMY - GRADUATES CAPSTONE PROJECT
 Caribbean LLM - Data Preparation
 By Adrian Dunkley
=============================================================================

 Data is di foundation of every good model. Caribbean data is
 special - we have multiple languages, dialects, cultural contexts.
 Dis script handles preparation of Caribbean text corpora fi fine-tuning.

 Usage:
   python data_prep.py --output_dir ./data --val_ratio 0.15
=============================================================================
"""

import json
import re
import os
import argparse
from typing import List, Dict, Tuple
from collections import Counter

# =============================================================================
# CARIBBEAN QA CORPUS
# =============================================================================

# Comprehensive QA pairs covering all CARICOM nations
CARIBBEAN_QA_CORPUS = [
    # --- JAMAICA ---
    {"question": "What are Jamaica's national symbols?",
     "answer": "Jamaica's national symbols include di Doctor Bird (hummingbird), di Blue Mahoe tree, di Lignum Vitae flower, Ackee fruit, and di national motto 'Out of Many, One People'.",
     "category": "culture", "country": "Jamaica"},
    {"question": "Explain the significance of Marcus Garvey to the Caribbean.",
     "answer": "Marcus Mosiah Garvey from St. Ann's Bay, Jamaica was a pioneer of Pan-Africanism and Black nationalism. Him founded UNIA in 1914 and inspired movements across Africa and di Caribbean. Him is a National Hero of Jamaica.",
     "category": "history", "country": "Jamaica"},
    {"question": "What is the Jamaica Stock Exchange known for?",
     "answer": "Di Jamaica Stock Exchange (JSE) was di best performing stock exchange in di world in 2015 and 2018, outperforming all major global markets in terms of returns.",
     "category": "economics", "country": "Jamaica"},

    # --- TRINIDAD AND TOBAGO ---
    {"question": "How was the steelpan invented?",
     "answer": "Di steelpan was invented in Trinidad in di 1930s-40s when African-Trinidadians, banned from using drums, discovered dat di dented surfaces of oil drums could produce musical notes. It is di only acoustic instrument invented in di 20th century.",
     "category": "culture", "country": "Trinidad and Tobago"},
    {"question": "What is Pitch Lake in Trinidad?",
     "answer": "Pitch Lake in La Brea, Trinidad is di largest natural deposit of asphalt in di world, covering about 100 acres. It has been a source of asphalt fi paving roads worldwide since di colonial era.",
     "category": "geography", "country": "Trinidad and Tobago"},

    # --- BARBADOS ---
    {"question": "Why is Barbados called Little England?",
     "answer": "Barbados was called 'Little England' due to its strong British colonial influence. Unlike most Caribbean islands, it was continuously under British rule from 1627 until independence in 1966. Barbados became a Republic in November 2021.",
     "category": "history", "country": "Barbados"},
    {"question": "Who is Sir Garfield Sobers?",
     "answer": "Sir Garfield Sobers from Barbados is widely regarded as di greatest all-round cricketer ever. Him scored over 8,000 Test runs, took 235 wickets, and famously hit six sixes in one over against Glamorgan in 1968.",
     "category": "sports", "country": "Barbados"},

    # --- GUYANA ---
    {"question": "What is Kaieteur Falls?",
     "answer": "Kaieteur Falls in Guyana is one of di most powerful waterfalls in di world, about five times di height of Niagara Falls. Located in di Potaro-Siparuni region, it drops 226 meters in a single cascade.",
     "category": "geography", "country": "Guyana"},

    # --- THE BAHAMAS ---
    {"question": "What is Junkanoo?",
     "answer": "Junkanoo is di Bahamas' premier cultural festival, held on Boxing Day (December 26) and New Year's Day. It features elaborate costumes, goatskin drums, cowbells, and brass instruments in a spectacular street parade.",
     "category": "culture", "country": "Bahamas"},

    # --- HAITI ---
    {"question": "What is Haitian Vodou?",
     "answer": "Haitian Vodou is a syncretic religion combining West African Vodun traditions wid Roman Catholic elements, developed by enslaved Africans in Haiti. It was officially recognized as a religion in Haiti in 2003.",
     "category": "culture", "country": "Haiti"},

    # --- DOMINICA ---
    {"question": "Why is Dominica called the Nature Island?",
     "answer": "Dominica is called di Nature Island because of its lush volcanic terrain, boiling lake (second largest in di world), 365 rivers, and largely unspoiled rainforest. It is home to di Kalinago (Carib) indigenous people.",
     "category": "geography", "country": "Dominica"},

    # --- GRENADA ---
    {"question": "Why is Grenada called the Spice Isle?",
     "answer": "Grenada is called di Spice Isle because it is one of di world's largest producers of nutmeg and mace. Di country also grows cinnamon, cloves, ginger, and cocoa.",
     "category": "agriculture", "country": "Grenada"},

    # --- ST. KITTS AND NEVIS ---
    {"question": "What is the significance of Brimstone Hill Fortress?",
     "answer": "Brimstone Hill Fortress in St. Kitts is a UNESCO World Heritage Site, one of di best-preserved historical fortifications in di Americas. Built by enslaved Africans fi di British between 1690 and 1790.",
     "category": "history", "country": "St. Kitts and Nevis"},

    # --- ANTIGUA AND BARBUDA ---
    {"question": "Who is Sir Vivian Richards?",
     "answer": "Sir Isaac Vivian Alexander Richards from Antigua is one of di greatest batsmen in cricket history. Him scored 8,540 Test runs at an average of 50.23 and captained di West Indies. Him never wore a helmet.",
     "category": "sports", "country": "Antigua and Barbuda"},

    # --- ST. LUCIA ---
    {"question": "How many Nobel Laureates has St. Lucia produced?",
     "answer": "St. Lucia, wid a population of about 180,000, has produced TWO Nobel Laureates: Sir Arthur Lewis (Economics, 1979) and Derek Walcott (Literature, 1992). Dat is di highest per capita in di world.",
     "category": "education", "country": "St. Lucia"},

    # --- BELIZE ---
    {"question": "What is the Great Blue Hole of Belize?",
     "answer": "Di Great Blue Hole is a giant marine sinkhole off di coast of Belize, over 300 meters across and 125 meters deep. It is a UNESCO World Heritage Site and one of di top scuba diving destinations in di world.",
     "category": "geography", "country": "Belize"},

    # --- SURINAME ---
    {"question": "What makes Suriname unique in South America?",
     "answer": "Suriname is di smallest country in South America and di only one where Dutch is di official language. It has incredible ethnic diversity - Hindustani, Creole, Javanese, Maroon, Indigenous, and Chinese communities.",
     "category": "culture", "country": "Suriname"},

    # --- ST. VINCENT AND THE GRENADINES ---
    {"question": "What is the significance of La Soufriere volcano?",
     "answer": "La Soufriere is an active volcano in St. Vincent dat last erupted explosively in April 2021, displacing about 20,000 people. Previous major eruptions occurred in 1902 and 1979.",
     "category": "geography", "country": "St. Vincent and the Grenadines"},

    # --- REGIONAL SPORTS ---
    {"question": "Tell me about Caribbean dominance in sprinting.",
     "answer": "Di Caribbean, especially Jamaica, dominates world sprinting. Usain Bolt holds both 100m (9.58s) and 200m (19.19s) world records. Shelly-Ann Fraser-Pryce, Elaine Thompson-Herah, and Yohan Blake continue di tradition. Trinidad's Hasely Crawford won Olympic 100m gold in 1976.",
     "category": "sports", "country": "Regional"},
]


# =============================================================================
# DATA CLEANING AND PROCESSING
# =============================================================================

def clean_text(text: str) -> str:
    """Clean text while preserving Caribbean dialect features."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Normalize quotes
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    return text


def validate_qa_pair(pair: Dict) -> bool:
    """Validate a QA pair meets quality standards."""
    if len(pair.get("question", "")) < 10:
        return False
    if len(pair.get("answer", "")) < 20:
        return False
    if "category" not in pair:
        return False
    return True


def prepare_dataset(qa_pairs: List[Dict], val_ratio: float = 0.15,
                    output_dir: str = "./data") -> Tuple[List, List]:
    """
    Prepare Caribbean dataset fi fine-tuning.
    Clean, validate, split, and export.
    """
    print("=" * 60)
    print(" CARIBBEAN DATA PREPARATION")
    print("=" * 60)

    # Clean and validate
    cleaned = []
    for pair in qa_pairs:
        pair["question"] = clean_text(pair["question"])
        pair["answer"] = clean_text(pair["answer"])
        if validate_qa_pair(pair):
            cleaned.append(pair)

    print(f"\nTotal pairs: {len(qa_pairs)}")
    print(f"Valid pairs: {len(cleaned)}")

    # Statistics
    categories = Counter(p["category"] for p in cleaned)
    countries = Counter(p["country"] for p in cleaned)
    print(f"\nCategories: {dict(categories)}")
    print(f"Countries:  {dict(countries)}")

    # Split
    import random
    random.seed(42)
    random.shuffle(cleaned)
    val_size = int(len(cleaned) * val_ratio)
    val_data = cleaned[:val_size]
    train_data = cleaned[val_size:]

    print(f"\nTrain: {len(train_data)} | Validation: {len(val_data)}")

    # Export
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "caribbean_train.jsonl")
    val_path = os.path.join(output_dir, "caribbean_val.jsonl")

    for path, data in [(train_path, train_data), (val_path, val_data)]:
        with open(path, 'w') as f:
            for entry in data:
                f.write(json.dumps(entry) + '\n')
        print(f"Exported: {path} ({len(data)} examples)")

    return train_data, val_data


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Caribbean LLM Data Preparation")
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    args = parser.parse_args()

    print("=" * 60)
    print(" CARIBBEAN AI ACADEMY - DATA PREPARATION")
    print(" By Adrian Dunkley")
    print("=" * 60)

    train_data, val_data = prepare_dataset(
        CARIBBEAN_QA_CORPUS,
        val_ratio=args.val_ratio,
        output_dir=args.output_dir
    )

    print("\n" + "=" * 60)
    print(" Data preparation complete!")
    print(" Next step: python main.py --epochs 3")
    print("=" * 60)
