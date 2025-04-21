from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# IndicBART for translation
translation_model_name = "ai4bharat/IndicBART"
translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_name)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_name)

# mBART50 for summarization
summarization_model_name = "facebook/mbart-large-50-many-to-many-mmt"
summarization_tokenizer = AutoTokenizer.from_pretrained(summarization_model_name)
summarization_model = AutoModelForSeq2SeqLM.from_pretrained(summarization_model_name)
