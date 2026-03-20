"""
=============================================================================
 CARIBBEAN AI ACADEMY - GRADUATES CAPSTONE PROJECT
 Caribbean LLM Fine-Tuning Pipeline
 By Adrian Dunkley
=============================================================================

 Dis is di main fine-tuning script fi di Caribbean LLM project.
 It takes a pre-trained model and adapts it fi Caribbean knowledge.

 Usage:
   python main.py --epochs 3 --lr 2e-4 --rank 16

 Requirements:
   pip install transformers datasets peft accelerate torch
=============================================================================
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

# NOTE: In production, uncomment these imports:
# import torch
# from transformers import (
#     AutoModelForCausalLM, AutoTokenizer,
#     TrainingArguments, DataCollatorForLanguageModeling
# )
# from peft import LoraConfig, get_peft_model, TaskType
# from datasets import load_dataset, Dataset
# from trl import SFTTrainer

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    "base_model": "gpt2",  # Start small; upgrade to Llama-3-8B when ready
    "output_dir": "./output/caribbean-llm-lora",
    "num_epochs": 3,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 100,
    "max_seq_length": 512,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["c_attn", "c_proj"],  # GPT-2 specific
    "fp16": True,
    "logging_steps": 10,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "data_path": "./data/caribbean_train.jsonl",
    "val_data_path": "./data/caribbean_val.jsonl",
}


# =============================================================================
# DATA FORMATTING
# =============================================================================

def format_instruction(example: Dict) -> str:
    """
    Format a single training example into instruction format.

    Template:
    ### System: You are a Caribbean AI assistant...
    ### Human: {question}
    ### Assistant: {answer}
    """
    system_msg = (
        "You are a knowledgeable Caribbean AI assistant. "
        "You understand Caribbean history, culture, sports, economics, "
        "and languages. You speak with authentic Caribbean voice and "
        "provide accurate, culturally sensitive responses."
    )

    # TODO: Customize template for your specific base model's chat format
    formatted = (
        f"### System:\n{system_msg}\n\n"
        f"### Human:\n{example['question']}\n\n"
        f"### Assistant:\n{example['answer']}\n"
    )
    return formatted


def format_chat_messages(example: Dict) -> List[Dict]:
    """Format as chat messages for models that support chat templates."""
    return [
        {
            "role": "system",
            "content": "You are a knowledgeable Caribbean AI assistant."
        },
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]}
    ]


# =============================================================================
# MODEL SETUP
# =============================================================================

def setup_model_and_tokenizer(config: Dict):
    """
    Load base model and apply LoRA configuration.

    TODO: Uncomment and run when yuh have PyTorch installed.
    """
    print(f"Loading base model: {config['base_model']}")

    # --- UNCOMMENT FOR ACTUAL TRAINING ---
    # tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    # tokenizer.pad_token = tokenizer.eos_token
    #
    # model = AutoModelForCausalLM.from_pretrained(
    #     config["base_model"],
    #     torch_dtype=torch.float16 if config["fp16"] else torch.float32,
    #     device_map="auto"
    # )
    #
    # # Configure LoRA
    # lora_config = LoraConfig(
    #     r=config["lora_rank"],
    #     lora_alpha=config["lora_alpha"],
    #     target_modules=config["lora_target_modules"],
    #     lora_dropout=config["lora_dropout"],
    #     bias="none",
    #     task_type=TaskType.CAUSAL_LM,
    # )
    #
    # model = get_peft_model(model, lora_config)
    # model.print_trainable_parameters()
    #
    # return model, tokenizer

    print("[SIMULATION] Model and LoRA configured")
    print(f"  LoRA rank: {config['lora_rank']}")
    print(f"  LoRA alpha: {config['lora_alpha']}")
    print(f"  Target modules: {config['lora_target_modules']}")
    return None, None


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_caribbean_dataset(config: Dict):
    """
    Load and prepare Caribbean training dataset.

    TODO: Replace with your actual data loading.
    """
    print(f"Loading dataset from: {config['data_path']}")

    # --- UNCOMMENT FOR ACTUAL TRAINING ---
    # train_dataset = load_dataset("json", data_files=config["data_path"])["train"]
    # val_dataset = load_dataset("json", data_files=config["val_data_path"])["train"]
    #
    # # Format examples
    # def preprocess(example):
    #     example["text"] = format_instruction(example)
    #     return example
    #
    # train_dataset = train_dataset.map(preprocess)
    # val_dataset = val_dataset.map(preprocess)
    #
    # return train_dataset, val_dataset

    # Simulation with sample data
    sample_data = [
        {"question": "Who is Usain Bolt?",
         "answer": "Usain Bolt from Trelawny, Jamaica is di fastest man in history."},
        {"question": "What is CARICOM?",
         "answer": "Di Caribbean Community established in 1973 fi regional integration."},
        {"question": "Tell me about Brian Lara.",
         "answer": "Brian Lara from Trinidad scored 400 not out, di highest Test innings."},
    ]
    print(f"[SIMULATION] Loaded {len(sample_data)} training examples")
    return sample_data, sample_data[:1]


# =============================================================================
# TRAINING
# =============================================================================

def train(config: Dict):
    """
    Main training loop using SFTTrainer from trl library.
    """
    print("\n" + "=" * 60)
    print(" CARIBBEAN LLM FINE-TUNING")
    print("=" * 60)

    # Setup
    model, tokenizer = setup_model_and_tokenizer(config)
    train_data, val_data = load_caribbean_dataset(config)

    # --- UNCOMMENT FOR ACTUAL TRAINING ---
    # training_args = TrainingArguments(
    #     output_dir=config["output_dir"],
    #     num_train_epochs=config["num_epochs"],
    #     per_device_train_batch_size=config["batch_size"],
    #     gradient_accumulation_steps=config["gradient_accumulation_steps"],
    #     learning_rate=config["learning_rate"],
    #     warmup_steps=config["warmup_steps"],
    #     logging_steps=config["logging_steps"],
    #     save_strategy=config["save_strategy"],
    #     evaluation_strategy=config["eval_strategy"],
    #     fp16=config["fp16"],
    #     optim="paged_adamw_32bit",
    #     report_to="none",  # Set to "wandb" for experiment tracking
    # )
    #
    # trainer = SFTTrainer(
    #     model=model,
    #     train_dataset=train_data,
    #     eval_dataset=val_data,
    #     args=training_args,
    #     formatting_func=format_instruction,
    #     max_seq_length=config["max_seq_length"],
    #     tokenizer=tokenizer,
    # )
    #
    # print("\nStarting training...")
    # trainer.train()
    #
    # # Save adapter
    # model.save_pretrained(config["output_dir"])
    # tokenizer.save_pretrained(config["output_dir"])
    # print(f"\nModel saved to: {config['output_dir']}")

    # Simulation
    print("\n[SIMULATION] Training loop:")
    for epoch in range(config["num_epochs"]):
        train_loss = 3.5 - epoch * 0.8 + (0.1 * (epoch + 1))
        val_loss = 3.7 - epoch * 0.7
        print(f"  Epoch {epoch + 1}/{config['num_epochs']}: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    print(f"\n[SIMULATION] Model would be saved to: {config['output_dir']}")


# =============================================================================
# INFERENCE
# =============================================================================

def generate_response(prompt: str, model_path: str = None):
    """
    Generate response using fine-tuned Caribbean LLM.

    TODO: Load your fine-tuned model and generate.
    """
    # --- UNCOMMENT FOR ACTUAL INFERENCE ---
    # from peft import PeftModel
    #
    # base_model = AutoModelForCausalLM.from_pretrained("gpt2")
    # model = PeftModel.from_pretrained(base_model, model_path)
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    #
    # formatted = format_instruction({"question": prompt, "answer": ""})
    # inputs = tokenizer(formatted, return_tensors="pt")
    #
    # with torch.no_grad():
    #     outputs = model.generate(
    #         **inputs,
    #         max_new_tokens=256,
    #         temperature=0.7,
    #         top_p=0.9,
    #         do_sample=True,
    #     )
    #
    # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # return response

    print(f"\n[SIMULATION] Generating response for: '{prompt}'")
    return f"[Caribbean LLM would respond to: {prompt}]"


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Caribbean LLM Fine-Tuning")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--base_model", type=str, default="gpt2", help="Base model name")
    parser.add_argument("--output_dir", type=str, default="./output/caribbean-llm-lora")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "generate"])
    parser.add_argument("--prompt", type=str, default="Who is Usain Bolt?")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = DEFAULT_CONFIG.copy()
    config["num_epochs"] = args.epochs
    config["learning_rate"] = args.lr
    config["lora_rank"] = args.rank
    config["batch_size"] = args.batch_size
    config["base_model"] = args.base_model
    config["output_dir"] = args.output_dir

    print("=" * 60)
    print(" CARIBBEAN AI ACADEMY - CAPSTONE PROJECT")
    print(" Caribbean LLM Fine-Tuning Pipeline")
    print(" By Adrian Dunkley")
    print("=" * 60)
    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    if args.mode == "train":
        train(config)
    elif args.mode == "generate":
        response = generate_response(args.prompt, args.output_dir)
        print(response)

    print("\n" + "=" * 60)
    print(" Caribbean LLM - Built by Caribbean people, fi Caribbean people!")
    print("=" * 60)
