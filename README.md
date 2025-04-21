![Cross Lingual Summarizer Banner](https://i.imgur.com/7n6YkL5.png)
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-brightgreen?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Cross_Lingual_Document_Summarizer
A deep learning NLP project that summarizes documents inputted in any Indian regional language and outputs the summary in another language of your choice.

## 🚀 How To Run

1. **Download models:**  
   Run `download.py` to download and save all required models locally.
2. **Install dependencies:**  
   Run `pip install -r requirements.txt`
3. **Launch the app:**  
   Run `streamlit run app.py`
4. **Use the app:**  
   - Paste your text in any supported Indian language.
   - Select source and target languages.
   - Click "Run Pipeline" to get your summary in the desired language.

## ✨ Example

## 🏗️ Architecture

[Regional Language Text]
↓
[mBART-50 Translation]
↓
[English Text]
↓
[BART-CNN Summarization]
↓
[English Summary]
↓
[mBART-50 Translation]
↓
[Summary in Target Language]

## 📄 License
MIT License

## 🙏 Credits
- Raghav Kishore V(https://github.com/rkv2005)
- [AI4Bharat](https://ai4bharat.org/)
