"""
실전 LoRA/QLoRA Fine-tuning - HuggingFace 활용

목표:
1. HuggingFace 모델에 LoRA/QLoRA 적용
2. 재즈 음악 데이터로 fine-tuning
3. 실전 파이프라인 구축

필수 라이브러리:
pip install transformers peft bitsandbytes accelerate
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from datasets import load_dataset, Dataset
import os


# ============================================================================
# 1. LoRA Fine-tuning (Standard)
# ============================================================================

def finetune_with_lora(
    model_name: str = "gpt2",
    dataset_path: str = "./jazz_dataset",
    output_dir: str = "./lora-model",
    lora_r: int = 8,
    lora_alpha: int = 16,
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
    batch_size: int = 4
):
    """
    Standard LoRA fine-tuning

    메모리 요구사항:
    - GPT-2 small (124M): ~2 GB
    - GPT-2 medium (355M): ~4 GB
    - GPT-2 large (774M): ~8 GB
    """
    print("=" * 80)
    print(f"LoRA Fine-tuning: {model_name}")
    print("=" * 80)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print(f"\nModel loaded: {model_name}")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    # LoRA configuration
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["c_attn", "c_proj"],  # GPT-2 specific
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    if os.path.exists(dataset_path):
        dataset = load_dataset("text", data_files={"train": f"{dataset_path}/train.txt"})
    else:
        # Dummy dataset for demo
        print("\nWarning: Using dummy dataset")
        dataset = Dataset.from_dict({
            "text": [
                "This is a jazz improvisation in bebop style.",
                "The melody follows ii-V-I progression.",
            ] * 100
        })
        dataset = {"train": dataset}

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )

    tokenized_dataset = dataset["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        warmup_steps=100,
        report_to="none"  # Disable wandb
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    # Train!
    print("\nStarting training...")
    trainer.train()

    # Save LoRA adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nLoRA adapter saved to {output_dir}")
    print(f"Adapter size: ~{lora_r * 2 * 0.01:.1f} MB")


# ============================================================================
# 2. QLoRA Fine-tuning (4-bit)
# ============================================================================

def finetune_with_qlora(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    dataset_path: str = "./jazz_dataset",
    output_dir: str = "./qlora-model",
    lora_r: int = 64,
    lora_alpha: int = 128,
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
    batch_size: int = 4
):
    """
    QLoRA fine-tuning (4-bit quantization)

    메모리 요구사항:
    - LLaMA-7B: ~6 GB (vs 14 GB without QLoRA)
    - LLaMA-13B: ~10 GB (vs 26 GB)
    - LLaMA-70B: ~45 GB (vs 140 GB)
    """
    print("=" * 80)
    print(f"QLoRA Fine-tuning: {model_name}")
    print("=" * 80)

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"\nModel loaded with 4-bit quantization: {model_name}")

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # QLoRA configuration
    qlora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply QLoRA
    model = get_peft_model(model, qlora_config)
    model.print_trainable_parameters()

    # Load dataset
    if os.path.exists(dataset_path):
        dataset = load_dataset("text", data_files={"train": f"{dataset_path}/train.txt"})
    else:
        print("\nWarning: Using dummy dataset")
        dataset = Dataset.from_dict({
            "text": [
                "Jazz improvisation with complex chord progressions.",
                "Bebop style with fast tempos and intricate melodies.",
            ] * 100
        })
        dataset = {"train": dataset}

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,  # Longer context for music
            padding="max_length"
        )

    tokenized_dataset = dataset["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        fp16=False,  # QLoRA uses bf16
        bf16=True,
        optim="paged_adamw_8bit",  # Paged optimizer for QLoRA
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        warmup_ratio=0.1,
        report_to="none"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    # Train!
    print("\nStarting QLoRA training...")
    trainer.train()

    # Save QLoRA adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nQLoRA adapter saved to {output_dir}")
    print(f"Adapter size: ~{lora_r * 2 * 0.02:.1f} MB")


# ============================================================================
# 3. Inference with LoRA/QLoRA
# ============================================================================

def inference_with_lora(
    base_model_name: str,
    lora_adapter_path: str,
    prompt: str = "Generate a jazz solo:",
    max_length: int = 100,
    use_4bit: bool = False
):
    """
    Load base model + LoRA adapter and generate

    Args:
        base_model_name: Base model name
        lora_adapter_path: Path to LoRA adapter
        prompt: Input prompt
        max_length: Max generation length
        use_4bit: Whether base model is 4-bit quantized
    """
    print("=" * 80)
    print("Inference with LoRA")
    print("=" * 80)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load base model
    if use_4bit:
        # QLoRA: Load with 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        # LoRA: Load normally
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    print(f"Base model: {base_model_name}")
    print(f"LoRA adapter: {lora_adapter_path}")

    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\nPrompt: {prompt}")
    print(f"Generated:\n{generated_text}")


# ============================================================================
# 4. Merge LoRA for Fast Inference
# ============================================================================

def merge_lora_weights(
    base_model_name: str,
    lora_adapter_path: str,
    output_path: str = "./merged-model"
):
    """
    Merge LoRA weights into base model for faster inference

    장점: Inference 속도 손실 없음
    단점: 모델 크기 커짐 (FP16)
    """
    print("=" * 80)
    print("Merging LoRA weights into base model")
    print("=" * 80)

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    # Merge!
    merged_model = model.merge_and_unload()

    # Save merged model
    merged_model.save_pretrained(output_path)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(output_path)

    print(f"\nMerged model saved to {output_path}")
    print("This model can be used like a normal model (no LoRA overhead)")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LoRA/QLoRA Fine-tuning")
    parser.add_argument("--mode", type=str, required=True,
                      choices=["lora", "qlora", "inference", "merge"],
                      help="Operation mode")
    parser.add_argument("--model", type=str, default="gpt2",
                      help="Base model name")
    parser.add_argument("--dataset", type=str, default="./jazz_dataset",
                      help="Dataset path")
    parser.add_argument("--output", type=str, default="./output",
                      help="Output directory")
    parser.add_argument("--rank", type=int, default=8,
                      help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=16,
                      help="LoRA alpha")
    parser.add_argument("--lr", type=float, default=2e-4,
                      help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3,
                      help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                      help="Batch size")

    args = parser.parse_args()

    if args.mode == "lora":
        # LoRA fine-tuning
        finetune_with_lora(
            model_name=args.model,
            dataset_path=args.dataset,
            output_dir=args.output,
            lora_r=args.rank,
            lora_alpha=args.alpha,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            batch_size=args.batch_size
        )

    elif args.mode == "qlora":
        # QLoRA fine-tuning
        finetune_with_qlora(
            model_name=args.model,
            dataset_path=args.dataset,
            output_dir=args.output,
            lora_r=args.rank,
            lora_alpha=args.alpha,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            batch_size=args.batch_size
        )

    elif args.mode == "inference":
        # Inference with LoRA
        inference_with_lora(
            base_model_name=args.model,
            lora_adapter_path=args.output,
            prompt="Generate a bebop jazz solo:",
            use_4bit=False
        )

    elif args.mode == "merge":
        # Merge LoRA weights
        merge_lora_weights(
            base_model_name=args.model,
            lora_adapter_path=args.output,
            output_path="./merged-model"
        )

    print("\n" + "=" * 80)
    print("사용 예시:")
    print("=" * 80)
    print("""
# 1. LoRA fine-tuning (GPT-2)
python 04_practical_training.py --mode lora --model gpt2 --dataset ./jazz --output ./lora-gpt2

# 2. QLoRA fine-tuning (LLaMA-7B)
python 04_practical_training.py --mode qlora --model meta-llama/Llama-2-7b-hf --rank 64 --alpha 128

# 3. Inference
python 04_practical_training.py --mode inference --model gpt2 --output ./lora-gpt2

# 4. Merge LoRA weights
python 04_practical_training.py --mode merge --model gpt2 --output ./lora-gpt2
    """)
    print("=" * 80)
