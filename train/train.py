# train_fixed_v2.py
import torch
import os

# from dotenv import load_dotenv
from transformers import (
    Trainer,
    TrainingArguments,
    AutoConfig,
    AutoTokenizer,
    LlamaForCausalLM,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)

from jsonl_ds import JsonlDataset
from torch.utils.data import DataLoader, random_split
import torch.distributed as dist

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

# load_dotenv()

def setup_distributed():
    """Setup distributed training"""
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 2))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        print(f"Rank {rank}/{world_size} initialized on GPU {local_rank}")

    return rank, local_rank, world_size


def main():
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()

    if rank == 0:
        print("=" * 60)
        print("Minnow-Math-1.5B Training")
        print(f"World size: {world_size}")
        print("=" * 60)

    # Load config
    config_path = "./config.json"
    config = AutoConfig.from_pretrained(config_path)

    # Create model
    model = LlamaForCausalLM(config)
    model.cuda()

    if rank == 0:
        print(f"Model parameters: {model.num_parameters():,}")

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf",
            use_fast=True,
            padding_side="right",  # Important for training
            truncation_side="right")
    except:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create dataset - CORRECTED: Use all documents for rank 0, subset for others
    # if rank == 0:
    #     print(f"\nLoading full dataset...")
    max_samples = None  # Use all samples
    # else:
    #     max_samples = 1000  # Smaller subset for other ranks during testing

    train_dataset = JsonlDataset(
        jsonl_path="./dataset_train_val/train.jsonl",
        tokenizer=tokenizer,
        seq_length=768,
        max_samples=max_samples
    )

    val_dataset = JsonlDataset(
        jsonl_path="./dataset_train_val/val.jsonl",
        tokenizer=tokenizer,
        seq_length=768,
        max_samples=max_samples
    )

    if rank == 0:
        print(f"Dataset size for this rank: {len(train_dataset):,} documents")

    # Training arguments - FIXED API
    training_args = TrainingArguments(
        output_dir=f"./math-a1-checkpoints",
        deepspeed="./ds_config.json",
        per_device_train_batch_size=32,  # Start small
        gradient_accumulation_steps=32,
        num_train_epochs=5,  # Test with 1 epoch
        learning_rate=2e-4,
        warmup_steps=500,
        weight_decay=0.1,
        dataloader_num_workers=4,

        # Precision
        bf16=torch.cuda.is_bf16_supported(),
        # fp16=not torch.cuda.is_bf16_supported(),

        # Memory
        gradient_checkpointing=True,

        # Checkpointing
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=5,
        logging_steps=10,

        per_device_eval_batch_size=32,  # Much higher than training
        eval_accumulation_steps=10,
        eval_strategy="steps",
        eval_steps=1000, # Should match save steps

        # Distributed
        local_rank=local_rank,
        # DDP
        ddp_find_unused_parameters=False,
        ddp_timeout=1800,

        # Optimization
        optim="adamw_torch",
        max_grad_norm=1.0,

        # Data loading - IMPORTANT: We'll handle dataloader ourselves
        dataloader_drop_last=True,
        remove_unused_columns=False,
        report_to="wandb" if rank == 0 else "none",

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # FIXED: Custom trainer with correct API
    class CustomTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            # Remove tokenizer from kwargs if present (it's passed separately)
            if 'tokenizer' in kwargs:
                self._tokenizer = kwargs.pop('tokenizer')
            super().__init__(*args, **kwargs)

        def get_train_dataloader(self):
            """Create DataLoader for our custom dataset"""
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")

            # Create DataLoader with appropriate settings
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=True,  # Shuffle is fine even with distributed
                collate_fn=data_collator,
                pin_memory=True,
                drop_last=True,
            )

    # Initialize trainer - FIXED: pass tokenizer separately
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,  # This is now handled correctly
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3
            )
        ],
    )

    # Train
    if rank == 0:
        print("\nStarting training...")
        print(f"Batch size per GPU: {training_args.per_device_train_batch_size}")
        print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
        print(
            f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * world_size}")

    try:
        trainer.train()

        if rank == 0:
            # Save final model
            trainer.save_model("./kitefish-math-tokenizer")
            tokenizer.save_pretrained("./kitefish-math-tokenizer")
            print("\n✓ Training completed successfully!")

    except Exception as e:
        print(f"\n✗ Training failed on rank {rank}: {e}")
        import traceback
        traceback.print_exc()

    # Cleanup
    # if world_size > 1:
    #     dist.destroy_process_group()


if __name__ == "__main__":
    main()