import os
import matplotlib.pyplot as plt
import evaluate
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# Suppress TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# --- Model Paths ---
mbart_path = "models/facebook--mbart-large-50-many-to-many-mmt"
summarizer_path = "models/bart-large-cnn"

# --- Load Translation Model ---
mbart_tokenizer = MBart50TokenizerFast.from_pretrained(mbart_path, local_files_only=True)
mbart_model = MBartForConditionalGeneration.from_pretrained(mbart_path, local_files_only=True)

# --- Load Summarization Model ---
sum_tokenizer = AutoTokenizer.from_pretrained(summarizer_path, local_files_only=True)
sum_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_path, local_files_only=True)
summarizer = pipeline(
    "summarization",
    model=sum_model,
    tokenizer=sum_tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# --- Indian Language Codes for mBART-50 ---
INDIAN_LANG_CODES = {
    "Bengali": "bn_IN",
    "Gujarati": "gu_IN",
    "Hindi": "hi_IN",
    "Kannada": "kn_IN",
    "Malayalam": "ml_IN",
    "Marathi": "mr_IN",
    "Nepali": "ne_NP",
    "Odia": "or_IN",
    "Punjabi": "pa_IN",
    "Sinhala": "si_LK",
    "Tamil": "ta_IN",
    "Telugu": "te_IN",
    "Urdu": "ur_PK",
    "English": "en_XX"
}

print("Supported Indian languages and codes:")
for lang, code in INDIAN_LANG_CODES.items():
    print(f"{lang}: {code}")

# --- Load Metrics ---
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

# --- User Input ---
text = input("\nEnter the regional language text: ")
src_lang = input("Enter the source language code (e.g., ta_IN): ").strip()
tgt_lang = "en_XX"
final_lang = input("Enter the target language code for final output (e.g., hi_IN): ").strip()

# --- Step 1: Regional Language → English ---
mbart_tokenizer.src_lang = src_lang
encoded = mbart_tokenizer(text, return_tensors="pt")
translated_tokens = mbart_model.generate(
    **encoded,
    forced_bos_token_id=mbart_tokenizer.lang_code_to_id[tgt_lang]
)
translated_en = mbart_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
print("\n[Step 1] Translated to English:\n", translated_en)

# --- Step 2: English Summarization ---
summary = summarizer(
    translated_en,
    max_length=80,
    min_length=25,
    do_sample=False
)[0]['summary_text']
print("\n[Step 2] English Summary:\n", summary)

# --- Step 3: English Summary → Another Regional Language ---
mbart_tokenizer.src_lang = "en_XX"
encoded_summary = mbart_tokenizer(summary, return_tensors="pt")
final_tokens = mbart_model.generate(
    **encoded_summary,
    forced_bos_token_id=mbart_tokenizer.lang_code_to_id[final_lang]
)
final_translation = mbart_tokenizer.batch_decode(final_tokens, skip_special_tokens=True)[0]
print(f"\n[Step 3] Summary in target language ({final_lang}):\n", final_translation)

# --- Ask user for reference texts for evaluation ---
ref_translation = input("\nEnter the reference English translation text (for evaluation): ")
ref_summary = input("Enter the reference English summary text (for evaluation): ")

# --- Compute ROUGE for summarization ---
rouge_scores = rouge.compute(predictions=[summary], references=[ref_summary])
# --- Compute BLEU for translation ---
bleu_scores = bleu.compute(predictions=[translated_en], references=[[ref_translation]])

# --- Print evaluation metrics ---
print("\nEvaluation Metrics:")
print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
print(f"BLEU: {bleu_scores['bleu']:.4f}")

# --- Plot the metrics ---
metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU']
scores = [rouge_scores['rouge1'], rouge_scores['rouge2'], rouge_scores['rougeL'], bleu_scores['bleu']]

plt.figure(figsize=(8, 5))
plt.bar(metrics, scores, color=['skyblue', 'orange', 'green', 'red'])
plt.title('Model Performance Metrics Comparison')
plt.ylabel('Score')
plt.ylim(0, 1)
for i, v in enumerate(scores):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
plt.show()
