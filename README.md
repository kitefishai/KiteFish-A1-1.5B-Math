# KiteFish-A1-1.5B — Scientific Math Language Model

KiteFish-A1-1.5B is a ~1.5B parameter decoder-only transformer trained from scratch on raw arXiv LaTeX sources spanning mathematics, computer science, and theoretical physics.

This repository documents the full data pipeline, tokenizer design, and training process used to build a domain-specialized scientific language model under constrained compute (2×A100 80GB GPUs).

---

## Overview

KiteFish-A1-1.5B is an engineering-focused case study in training a scientific language model directly from raw arXiv archives.

Unlike curated or proprietary datasets, this model was built from:

- Raw arXiv metadata
- `.tar.gz` LaTeX source archives
- Custom filtering and normalization pipeline
- Domain-aware tokenization for symbolic text

The base model is not instruction-tuned. It is optimized for modeling mathematical and scientific text distributions.

---

## Model Specifications

| Component | Value |
|------------|--------|
| Parameters | ~1.5B |
| Architecture | LLaMA-style dense transformer |
| Layers | 24 |
| Hidden Size | 2048 |
| FFN Size | 5504 |
| Attention Heads | 16 |
| Vocabulary Size | 102,400 |
| Context Length | 4096 (trained at 768 tokens) |
| Precision | bfloat16 |
| Embeddings | Untied |

---

## Training Summary

- Pretraining Tokens: 52.18B  
- Post-training Tokens: 5B  
- Raw Processed Corpus: ~200GB  
- GPU Setup: 2× NVIDIA A100 (80GB)  
- Total Experimental Runs: 24  
- Final Validation Perplexity: ~4.2  

Optimization stack:

- AdamW
- ZeRO Stage 2
- Gradient checkpointing
- Mixed precision (bf16)
- Weights & Biases monitoring

---

## Dataset Construction Pipeline

The dataset was constructed directly from raw arXiv LaTeX archives.

Pipeline stages:

1. Metadata filtering (subject, year, withdrawn removal)
2. Archive validation
3. Multi-file LaTeX resolution (`\input`, `\include`)
4. Cleaning & normalization
5. Deduplication
6. Domain-aware tokenizer training (102k vocabulary)

Key engineering challenges:

- LaTeX extraction failures
- False negatives in language detection for formula-heavy documents
- Storage and I/O bottlenecks
- Mathematical symbol fragmentation during tokenization

---

## Training Dynamics

Training was iteratively refined across 24 runs.

Observed behavior:

- Small-data regime (20GB) resulted in unstable convergence.
- Full 200GB regime significantly improved gradient stability.
- No sustained divergence between train and validation loss.
- Gradient norms stabilized after early warmup.
- GPU utilization remained consistently above 95%.

The model was trained in a data-rich regime (~38 tokens per parameter), prioritizing domain robustness over strict compute-optimal scaling.

---

## Repository Structure

```

.
├── config.json
├── ds_config.json
├── train.py
├── data_prep/
│   ├── extraction.py
│   ├── filters.py
│   ├── latex_cleaner.py
│   └── prepare_dataset.py
├── dataset_train_val/
│   ├── train.jsonl
│   └── val.jsonl
└── paper/

````


## Reproducing Training

### Requirements

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

---

## Intended Use

Designed for:

* Scientific text modeling
* Mathematical generation
* Research experimentation in domain-specific LMs

Not optimized for:

* General conversational AI
* Instruction-following
* Chat-based deployment

---

## Limitations

* Not instruction-tuned
* Domain-restricted corpus
* Training sequence length: 768 tokens
* Significant storage requirements (~200GB processed)
* Dependent on LaTeX extraction quality

---

## Citation

If you use this work, please cite:

```bibtex
@misc{kitefish_a1_1_5b,
  title={ArXiv-to-Model: A Practical Study of Scientific LM Training},
  author={Your Name},
  year={2026}
}
```

---

## License

MIT License

Copyright (c) 2026 KiteFish

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.