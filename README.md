# Cross Lingual Document Summarizer
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
    **Input (Hindi) **
       21वीं सदी में प्रौद्योगिकी की तीव्र प्रगति ने दैनिक जीवन के लगभग हर पहलू को बदल दिया है। लोगों के संवाद करने और सूचना तक पहुँचने के तरीके से लेकर उनके काम करने और मनोरंजन करने के तरीके तक, डिजिटल नवाचार समाज में गहराई से समाहित हो गए         हैं। स्मार्टफोन, हाई-स्पीड इंटरनेट और सोशल मीडिया प्लेटफ़ॉर्म ने व्यक्तियों के लिए दुनिया भर में दूसरों से तुरंत जुड़ना संभव बना दिया है। साथ ही, इन परिवर्तनों ने नई चुनौतियाँ भी पेश की हैं, जैसे कि गोपनीयता के बारे में चिंताएँ, गलत सूचना का प्रसार और उन          लोगों के बीच डिजिटल विभाजन जिनके पास प्रौद्योगिकी तक पहुँच है और जिनके पास नहीं है। जैसे-जैसे प्रौद्योगिकी विकसित होती जा रही है, व्यक्तियों और समुदायों के लिए अनुकूलन करना महत्वपूर्ण है, यह सुनिश्चित करना कि नवाचार के लाभ व्यापक रूप से साझा किए        जाएँ और संभावित जोखिमों को जिम्मेदारी से प्रबंधित किया जाए।

    **English Translation**
       The rapid progress of technology in the 21st century has changed almost every aspect of daily life. From the way people communicate and access information to the way they work and enjoy themselves, digital        innovation has become deeply embedded in the society. Smartphones, high speed internet and social media platforms have made it possible for individuals to connect instantly with others around the world. At        the same time, these changes have also presented new challenges, such as privacy concerns, the spread of misinformation and the digital divide between those who have access to technology and those who do          not. As technology evolves, it is important to adapt for individuals and communities, to ensure that the benefits of innovation are widely shared and the potential risks are managed responsibly.

    **English Summary** 
       The rapid progress of technology in the 21st century has changed almost every aspect of daily life. Smartphones, high speed internet and social media platforms have made it possible for individuals to             connect instantly with others around the world.
       
    **Output (Tamil) **
       21ம் நூற்றாண்டில் தொழில்நுட்பத்தின் விரைவான முன்னேற்றம், வாழ்க்கையின் ஒவ்வொரு அம்சத்தையும் மாற்றிவிட்டது. ஸ்மார்ட்பேர்ட்கள், அதிவேகமான இணையம், சமூக ஊடகங்கள் ஆகியவை            உலகெங்கும் உள்ள மற்றவர்களுடன் உடனடியாக இணைவதற்கு தனிநபர்களுக்கு வழிவகுத்துள்ளன.

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
