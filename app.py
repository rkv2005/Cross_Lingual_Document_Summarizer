import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import matplotlib.pyplot as plt
import evaluate
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# --- Model Paths ---
mbart_path = "models/mbartlarge50mmt"
summarizer_path = "models/bartlargecnn"

# --- Load Models and Tokenizers ---
@st.cache_resource
def load_mbart():
    tokenizer = MBart50TokenizerFast.from_pretrained(mbart_path, local_files_only=True)
    model = MBartForConditionalGeneration.from_pretrained(mbart_path, local_files_only=True)
    return tokenizer, model

@st.cache_resource
def load_summarizer():
    tokenizer = AutoTokenizer.from_pretrained(summarizer_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_path, local_files_only=True)
    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    return summarizer

mbart_tokenizer, mbart_model = load_mbart()
summarizer = load_summarizer()

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

st.title("Cross-Lingual Summarizer (Indian Languages)")

# --- User Inputs ---
text = st.text_area("Enter regional language text:", height=150)
src_lang = st.selectbox("Source language", list(INDIAN_LANG_CODES.keys()), index=0)
final_lang = st.selectbox("Target language for summary", list(INDIAN_LANG_CODES.keys()), index=1)

# Reference inputs for evaluation
st.markdown("#### (Optional) For Evaluation: Paste reference English translation and summary below")
ref_translation = st.text_area("Reference English Translation", height=80)
ref_summary = st.text_area("Reference English Summary", height=80)

if st.button("Run Pipeline"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        # Step 1: Regional Language → English
        mbart_tokenizer.src_lang = INDIAN_LANG_CODES[src_lang]
        encoded = mbart_tokenizer(text, return_tensors="pt")
        translated_tokens = mbart_model.generate(
            **encoded,
            forced_bos_token_id=mbart_tokenizer.lang_code_to_id["en_XX"]
        )
        translated_en = mbart_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        st.markdown("### Step 1: Translated to English")
        st.success(translated_en)

        # Step 2: English Summarization
        summary = summarizer(
            translated_en,
            max_length=80,
            min_length=25,
            do_sample=False
        )[0]['summary_text']
        st.markdown("### Step 2: English Summary")
        st.info(summary)

        # Step 3: English Summary → Another Regional Language
        mbart_tokenizer.src_lang = "en_XX"
        encoded_summary = mbart_tokenizer(summary, return_tensors="pt")
        final_tokens = mbart_model.generate(
            **encoded_summary,
            forced_bos_token_id=mbart_tokenizer.lang_code_to_id[INDIAN_LANG_CODES[final_lang]]
        )
        final_translation = mbart_tokenizer.batch_decode(final_tokens, skip_special_tokens=True)[0]
        st.markdown(f"### Step 3: Summary in {final_lang}")
        st.success(final_translation)

        # --- Metrics and Plot ---
        if ref_translation.strip() and ref_summary.strip():
            try:
                rouge = evaluate.load("rouge")
                bleu = evaluate.load("bleu")
                # Compute metrics
                rouge_scores = rouge.compute(predictions=[summary], references=[ref_summary])
                bleu_scores = bleu.compute(predictions=[translated_en], references=[[ref_translation]])

                st.markdown("### Evaluation Metrics")
                st.write(f"**ROUGE-1:** {rouge_scores['rouge1']:.4f}")
                st.write(f"**ROUGE-2:** {rouge_scores['rouge2']:.4f}")
                st.write(f"**ROUGE-L:** {rouge_scores['rougeL']:.4f}")
                st.write(f"**BLEU:** {bleu_scores['bleu']:.4f}")

                # Bar plot
                metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU']
                scores = [rouge_scores['rouge1'], rouge_scores['rouge2'], rouge_scores['rougeL'], bleu_scores['bleu']]

                fig, ax = plt.subplots(figsize=(7, 4))
                bars = ax.bar(metrics, scores, color=['skyblue', 'orange', 'green', 'red'])
                ax.set_ylim(0, 1)
                ax.set_ylabel('Score')
                ax.set_title('Model Performance Metrics')
                for bar, score in zip(bars, scores):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{score:.2f}", ha='center', fontweight='bold')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error computing metrics: {e}")
        else:
            st.info("To see evaluation metrics and plot, please provide both reference English translation and summary.")

st.markdown("---")
st.markdown(
    "Supported languages: Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Nepali, Odia, Punjabi, Sinhala, Tamil, Telugu, Urdu, English"
)
