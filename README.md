<h1 align="center">Minnow-Math-1.5B</h1>

<p align="center">
Scientific & Mathematical Language Model Trained from Raw arXiv LaTeX
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" />
  <img src="https://img.shields.io/badge/parameters-1.5B-green" />
  <img src="https://img.shields.io/badge/architecture-LLaMA--style-purple" />
</p>

**Minnow-Math-1.5B** is a 1.5B parameter decoder-only transformer trained from scratch on raw arXiv LaTeX sources spanning mathematics, computer science, and theoretical physics.

Paper (arXiv): https://arxiv.org/pdf/2602.17288  
Model (HuggingFace): https://huggingface.co/KiteFishAI/Minnow-Math-1.5B  

<p align="center">
  <a href="https://arxiv.org/abs/2602.17288">
    <img src="assets/paper_thumbnail.png" width="500">
  </a>
</p>

This repository documents the dataset construction pipeline, tokenizer design, and training process used to build a domain-specialized scientific language model under constrained compute (2× A100 80GB GPUs).

## Motivation

Most open models are trained on heterogeneous web corpora.  
KiteFish-A1 explores a different direction:

> What happens when a language model is trained purely on structured scientific LaTeX archives?

The goal is to study domain specialization, engineering trade-offs, and training dynamics under realistic compute budgets.

## Model Specifications

| Component | Value |
|-----------|--------|
| Parameters | ~1.5B |
| Architecture | LLaMA-style dense transformer |
| Layers | 24 |
| Hidden Size | 2048 |
| FFN Size | 5504 |
| Attention Heads | 16 |
| Vocabulary | 102,400 |
| Context Length | 4096 (trained at 768 tokens) |
| Precision | bfloat16 |
| Embeddings | Untied |

## Training Summary

Pretraining Tokens: 52.18B  
Post-training Tokens: 5B  
Processed Corpus Size: ~200GB  
Experimental Runs: 24  
Hardware: 2× NVIDIA A100 (80GB)  

Optimization stack:

- AdamW  
- ZeRO Stage 2  
- Gradient checkpointing  
- bf16 mixed precision  

Validation Perplexity: ~4.2 on held-out scientific corpus  

Training operated in a data-rich regime (~38 tokens per parameter), prioritizing domain robustness over benchmark optimization.

## Dataset Pipeline

Constructed directly from raw arXiv LaTeX archives:

1. Metadata filtering (subject selection, withdrawn removal)  
2. `.tar.gz` archive extraction  
3. Multi-file LaTeX resolution (`\input`, `\include`)  
4. Cleaning and normalization  
5. Deduplication  
6. Domain-aware tokenizer training (102k vocabulary)

Key engineering challenges included LaTeX extraction inconsistencies, formula-heavy language detection issues, symbol fragmentation during tokenization, and large-scale I/O bottlenecks.

## Performance Characteristics

Minnow-Math-1.5B is a base model.

It demonstrates:

- Strong familiarity with scientific writing style  
- Stable LaTeX structural modeling  
- Symbolic fluency  

It does not include:

- Instruction tuning  
- RLHF or preference alignment  
- Benchmark optimization  

Downstream benchmark accuracy is currently modest without supervised fine-tuning or LoRA adaptation.

This release is intended primarily for research and experimentation.

## Reproducing Training

Requirements:

- Python 3.10+
- PyTorch
- Transformers
- DeepSpeed
- 2× A100 GPUs recommended

Install dependencies:

```bash
pip install -r requirements.txt
````

Launch training:

```bash
deepspeed train.py
```

## Citation

If you use this work, please cite:

```bibtex
@article{kitefish2026,
  title={KiteFish-A1: Training a Scientific Language Model from Raw LaTeX Archives},
  author={Your Name},
  year={2026},
  eprint={2602.17288},
  archivePrefix={arXiv}
}
```

## License

MIT License © 2026 KiteFish