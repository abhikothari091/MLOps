# LLM Data Pipeline Lab – GPT-2 Mini Fine-Tuning

This lab demonstrates how to build a **complete text-processing pipeline** and perform a small-scale **GPT-2 fine-tuning** experiment using Hugging Face `datasets`, `transformers`, and PyTorch.  
Compared to the starter code, this version introduces **significant improvements** that make the workflow realistic and production-like.

---

## 🔧 What We Built

1. **Loaded raw text** from the Wikitext-2 dataset.  
2. **Cleaned & filtered** noisy lines (empty text, headers, etc.).  
3. **Tokenized** using GPT-2 tokenizer with proper padding.  
4. **Concatenated and chunked** tokens into fixed-length sequences suitable for language modeling.  
5. Created **custom PyTorch DataLoaders** using a manual `collate_fn`.  
6. Implemented a **minimal fine-tuning loop** for GPT-2.  
7. Evaluated the model using **validation loss + perplexity**.  
8. Added an **optional text-generation cell** to verify model behavior.

---

## 🆕 Key Improvements Over the Original Lab

### 1. Real Data Cleaning  
- Removed empty lines and structural markup.  
- Ensured the dataset contained usable natural-language text.

### 2. Proper Tokenization Setup  
- Configured GPT-2 tokenizer with pad token assigned to EOS.  
- Returned attention masks for correct batching.

### 3. Language-Model Sequence Formatting  
Added a grouping step that:
- concatenates token sequences across lines,
- trims to a multiple of `block_size`,
- splits into equal chunks required for LLM training.

This is the core piece missing from the original lab.

### 4. Train/Validation Pipeline  
- Introduced a clean 95/5 split.  
- Created PyTorch DataLoaders with manual batch collation (input_ids, attention_mask, labels).

### 5. Actual Model Training  
- Loaded GPT-2 and ran a minimal fine-tuning loop with:
  - AdamW optimizer  
  - Linear warmup scheduler  
  - GPU support if available  

### 6. Evaluation Metrics  
- Computed mean validation loss  
- Calculated perplexity for interpretability  

### 7. Text Generation  
Added a final demonstration of the fine-tuned model.

---

## 🗂 Project Flow Summary

1. **Load + Clean Data**  
2. **Tokenize**  
3. **Group tokens into blocks**  
4. **DataLoader creation**  
5. **Fine-tuning loop**  
6. **Validation & perplexity**  
7. **Generation demo**

This transforms the lab from “tokenization only” into a **complete LLM preprocessing + training pipeline**.

---

## ▶️ How to Run

Install dependencies:

```bash
pip install datasets transformers torch
```

