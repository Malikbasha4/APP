import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from textstat import flesch_kincaid_grade, flesch_reading_ease
from langdetect import detect, DetectorFactory
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
from sentence_transformers import SentenceTransformer # New import for sentence similarity

# --- Set random seed for reproducibility of langdetect ---
DetectorFactory.seed = 0

# --- NLTK Downloads (Run unconditionally at the start) ---
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# --- Custom CSS for Enhanced UI/UX ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
        color: #262626; /* Dark grey for text */
    }

    /* Main container styling */
    .stApp {
        background: linear-gradient(to right, #e0f7fa, #e8f5e9); /* Soft gradient background */
        padding: 20px;
    }

    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        color: #004d40; /* Dark teal for headers */
        font-weight: 700; /* Bolder headers */
        margin-top: 1.5em;
        margin-bottom: 0.8em;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
    }

    /* Streamlit widgets styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 12px; /* More rounded */
        border: 1px solid #c8e6c9; /* Light green border */
        box-shadow: 3px 3px 12px rgba(0,0,0,0.1); /* More pronounced shadow */
        padding: 12px;
        transition: all 0.3s ease-in-out;
        background-color: white;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #00796b; /* Teal on focus */
        box-shadow: 4px 4px 15px rgba(0,121,107,0.3); /* Stronger shadow on focus */
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(to right, #00796b, #004d40); /* Teal gradient */
        color: white;
        border-radius: 10px; /* More rounded */
        border: none;
        padding: 12px 25px;
        font-weight: 700;
        box-shadow: 4px 4px 12px rgba(0,121,107,0.4);
        transition: all 0.3s ease-in-out;
        cursor: pointer;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        background: linear-gradient(to right, #004d40, #00796b); /* Reverse gradient on hover */
        box-shadow: 5px 5px 15px rgba(0,121,107,0.5);
        transform: translateY(-3px); /* Lift effect */
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #e0f2f1; /* Light teal for expander headers */
        border-radius: 10px; /* More rounded */
        padding: 12px 20px;
        margin-bottom: 12px;
        border: 1px solid #b2dfdb;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.08);
        transition: all 0.2s ease-in-out;
        font-weight: 600;
        color: #004d40;
    }
    .streamlit-expanderHeader:hover {
        background-color: #c8e6c9; /* Slightly different hover color */
        box-shadow: 3px 3px 10px rgba(0,0,0,0.1);
    }
    .streamlit-expanderContent {
        background-color: white; /* White background for content */
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #e0e0e0;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 25px;
    }

    /* Info, Success, Warning, Error messages */
    .stAlert {
        border-radius: 10px;
        padding: 12px 20px;
        margin-bottom: 20px;
        font-weight: 500;
    }
    .stAlert.info {
        background-color: #e3f2fd; /* Light blue */
        border-left: 6px solid #2196f3; /* Blue border */
    }
    .stAlert.success {
        background-color: #e8f5e9; /* Light green */
        border-left: 6px solid #4caf50; /* Green border */
    }
    .stAlert.warning {
        background-color: #fff3e0; /* Light orange */
        border-left: 6px solid #ff9800; /* Orange border */
    }
    .stAlert.error {
        background-color: #ffebee; /* Light red */
        border-left: 6px solid #f44336; /* Red border */
    }

    /* Table styling for NER */
    .dataframe {
        width: 100%;
        border-collapse: collapse;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .dataframe th {
        background: linear-gradient(to right, #00796b, #004d40); /* Teal gradient header */
        color: white;
        padding: 12px;
        text-align: left;
        font-weight: 600;
    }
    .dataframe td {
        background-color: white;
        padding: 12px;
        border-bottom: 1px solid #e0e0e0;
    }
    .dataframe tr:last-child td {
        border-bottom: none;
    }
    .dataframe tbody tr:nth-child(odd) {
        background-color: #f0fdfa; /* Lightest teal for odd rows */
    }

    /* Markdown elements */
    code {
        background-color: #e0f2f1;
        border-radius: 5px;
        padding: 3px 6px;
        font-family: 'Fira Code', monospace;
        color: #004d40;
    }
    mark {
        background-color: #b2dfdb; /* Light teal for highlights */
        padding: 0 5px;
        border-radius: 4px;
        font-weight: 600;
    }

    /* Custom spinner animation */
    .stSpinner > div > div {
        border-top-color: #00796b;
        border-left-color: #00796b;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --- Load Models (Cached for performance) ---
@st.cache_resource
def load_summarizer_model():
    """Loads the pre-trained summarization model."""
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@st.cache_resource
def load_sentiment_model():
    """Loads the pre-trained sentiment analysis model."""
    return pipeline("sentiment-analysis")

@st.cache_resource
def load_spacy_model():
    """Loads the spaCy English model for NER."""
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_text_generation_model():
    """Loads the pre-trained text generation model."""
    return pipeline("text-generation", model="gpt2")

@st.cache_resource
def load_translation_models():
    """Loads pre-trained translation models for multiple languages."""
    translation_pipelines = {}
    language_models = {
        "French": "Helsinki-NLP/opus-mt-en-fr",
        "Chinese": "Helsinki-NLP/opus-mt-en-zh",
        "German": "Helsinki-NLP/opus-mt-en-de",
        "Arabic": "Helsinki-NLP/opus-mt-en-ar",
        "Spanish": "Helsinki-NLP/opus-mt-en-es",
        "Portuguese": "Helsinki-NLP/opus-mt-en-pt",
        "Russian": "Helsinki-NLP/opus-mt-en-ru",
    }

    for lang_name, model_name in language_models.items():
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            translation_pipelines[lang_name] = pipeline("translation", model=model, tokenizer=tokenizer)
        except Exception as e:
            translation_pipelines[lang_name] = None
            
    return translation_pipelines

@st.cache_resource
def load_emotion_model():
    """Loads a pre-trained model for emotion detection."""
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

@st.cache_resource
def load_grammar_correction_model():
    """Loads a pre-trained model for grammar and spelling correction."""
    # This model is for text2text-generation, suitable for correction tasks
    return pipeline("text2text-generation", model="oliverguhr/detoxify-unbiased-xlm-r-multilingual")

@st.cache_resource
def load_sentence_transformer_model():
    """Loads a Sentence-Transformer model for generating sentence embeddings."""
    return SentenceTransformer('all-MiniLM-L6-v2')


# Initialize models
with st.spinner("🚀 Initializing powerful AI models... This might take a moment!"):
    summarizer = load_summarizer_model()
    sentiment_analyzer = load_sentiment_model()
    nlp_spacy = load_spacy_model()
    text_generator = load_text_generation_model()
    translators = load_translation_models()
    emotion_analyzer = load_emotion_model() # New model
    grammar_corrector = load_grammar_correction_model() # New model
    sentence_embedder = load_sentence_transformer_model() # New model
st.success("✅ Models initialized and ready to analyze!")

# --- Helper Functions ---

def extract_keywords(text, num_keywords=10):
    """
    Extracts keywords using NLTK tokenization and frequency counting.
    Filters out stopwords and punctuation.
    """
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    word_counts = Counter(filtered_words)
    return [word for word, count in word_counts.most_common(num_keywords)]

def perform_ner_and_highlight(text):
    """
    Performs Named Entity Recognition using spaCy and returns the text with highlighted entities.
    Returns a tuple: (list of entities, highlighted text HTML string).
    """
    doc = nlp_spacy(text)
    entities = []
    highlighted_text = text
    
    # Sort entities by start index in reverse to avoid issues with index changes during replacement
    sorted_ents = sorted(doc.ents, key=lambda ent: ent.start_char, reverse=True)

    for ent in sorted_ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "explanation": spacy.explain(ent.label_)
        })
        # Basic highlighting for display
        highlighted_text = (
            highlighted_text[:ent.start_char] +
            f"<mark><b>{ent.text}</b> ({ent.label_})</mark>" + # Using custom mark style from CSS
            highlighted_text[ent.end_char:]
        )
    return entities, highlighted_text

def generate_wordcloud(text, keywords):
    """Generates a word cloud image from the text, focusing on keywords."""
    if not keywords or not text.strip():
        return None

    word_freq = Counter(keywords)
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    for word in filtered_words:
        if word not in word_freq:
            word_freq[word] = 1

    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          max_words=50, collocations=False).generate_from_frequencies(word_freq)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def preprocess_text_for_similarity(text):
    """Basic preprocessing for text comparison."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation and numbers
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    return " ".join(filtered_words)

# --- Streamlit UI ---

st.set_page_config(layout="wide", page_title="Advanced Intelligent Text Analysis Workbench")

st.title("🧠 Advanced Intelligent Text Analysis Workbench")
st.markdown("""
    Welcome to your **Intelligent Text Analysis Workbench**! 🚀
    This powerful app helps you understand your text better by offering a suite of AI-powered tools.
    Paste your text, upload a file, and explore insights from summarization, sentiment, entity recognition,
    text generation, readability, language detection, and even text comparison!
""")

# --- Input Section ---
st.header("1. Input Your Text 📝")

input_container = st.container()
with input_container:
    col1_input, col2_input = st.columns(2)

    with col1_input:
        input_text = st.text_area(
            "Paste your text here:",
            height=300,
            placeholder="Type or paste any text you want to analyze (e.g., an article, a speech, a review)...",
            key="main_input_text"
        )

    with col2_input:
        st.markdown("<br>", unsafe_allow_html=True) # Add some vertical space
        uploaded_file = st.file_uploader("Or upload a text file (.txt):", type=["txt"])
        if uploaded_file is not None:
            file_content = uploaded_file.read().decode("utf-8")
            st.session_state.main_input_text = file_content # Update text area with file content
            st.info("File uploaded and content loaded into the text area. ✨")

    st.write("") # Add a little space
    if st.button("🗑️ Clear All Text", help="Clears the input text area and any uploaded file content"):
        st.session_state.main_input_text = ""
        st.experimental_rerun()

if input_text:
    st.markdown("---")
    st.header("2. Choose Your Analysis Tools 🛠️")

    analysis_options = st.multiselect(
        "Select Analysis Features to Run:",
        ["Summarization", "Keyword Extraction", "Sentiment Analysis", "Emotion Detection", # Added Emotion Detection
         "Named Entity Recognition", "Text Generation", "Grammar & Style Correction", # Added Grammar Correction
         "Readability Score", "Language Detection", "Translation", "Word Cloud", 
         "Text Comparison", "Sentence Similarity"], # Added Sentence Similarity
        default=["Summarization", "Keyword Extraction", "Sentiment Analysis", "Named Entity Recognition"]
    )

    st.markdown("---")
    st.header("3. Analysis Results 📊")

    # Initialize variables for download section
    summary_for_download = "N/A"
    keywords_for_download = []
    label = "N/A"
    score = "N/A"
    emotion_results = "N/A" # New variable
    corrected_text = "N/A" # New variable
    entity_data = "N/A"
    highlighted_text_html = input_text # Default to original text if NER not run
    generated_text = "N/A"
    fk_grade = "N/A"
    flesch_ease = "N/A"
    detected_lang = "N/A"
    translation_result = "N/A"
    text_comparison_result = "N/A"
    sentence_similarity_result = "N/A" # New variable

    # --- Summarization ---
    if "Summarization" in analysis_options:
        with st.expander("📝 Text Summary", expanded=True):
            st.markdown("Adjust the desired length of the summary below.")
            summary_type = st.radio("Summary Length Control:", ["Auto", "Percentage", "Word Count"], key="summary_type")
            
            words_in_text = len(input_text.split())
            min_len_default = min(50, words_in_text // 4)
            max_len_default = min(200, words_in_text // 2)

            if summary_type == "Percentage":
                summary_percentage = st.slider("Summary Length (% of original text):", min_value=10, max_value=80, value=30, step=5, key="summary_percent")
                min_len = int(words_in_text * (summary_percentage / 100) * 0.8) # 80% of target percentage
                max_len = int(words_in_text * (summary_percentage / 100) * 1.2) # 120% of target percentage
                min_len = max(min_len, 20) # Ensure minimum length
                max_len = max(max_len, min_len + 10) # Ensure max is greater than min
            elif summary_type == "Word Count":
                target_words = st.slider("Target Summary Word Count:", min_value=20, max_value=300, value=100, step=10, key="summary_words")
                min_len = int(target_words * 0.8)
                max_len = int(target_words * 1.2)
                min_len = max(min_len, 20)
                max_len = max(max_len, min_len + 10)
            else: # Auto
                min_len = min_len_default
                max_len = max_len_default

            with st.spinner("Generating summary..."):
                try:
                    if words_in_text < 50:
                        st.warning("Text is too short for effective summarization. Minimum recommended words: 50. ⚠️")
                        summary_for_download = "Not enough text to generate a meaningful summary."
                    else:
                        summary_for_download = summarizer(input_text, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']
                    st.success("Summary Generated: ✨")
                    st.write(summary_for_download)
                except Exception as e:
                    st.error(f"Error generating summary: {e} ❌")
                    st.warning("Summarization might fail for very short texts or due to model limitations. 🚧")
                    summary_for_download = ""

    # --- Keyword Extraction ---
    if "Keyword Extraction" in analysis_options:
        with st.expander("🔑 Keywords/Key Phrases", expanded=True):
            with st.spinner("Extracting keywords..."):
                try:
                    keywords_for_download = extract_keywords(input_text)
                    if keywords_for_download:
                        st.success("Extracted Keywords: 💡")
                        st.markdown(f"**`{', '.join(keywords_for_download)}`**")
                    else:
                        st.info("No significant keywords found or text is too short. ℹ️")
                except Exception as e:
                    st.error(f"Error extracting keywords: {e} ❌")

    # --- Word Cloud ---
    if "Word Cloud" in analysis_options:
        with st.expander("☁️ Word Cloud Visualization", expanded=True):
            st.info("This feature visualizes the most frequent words in your text or summary. It does not generate logos or graphic designs. ℹ️")
            with st.spinner("Generating word cloud..."):
                try:
                    wordcloud_source_text = summary_for_download if summary_for_download and summary_for_download != "Not enough text to generate a meaningful summary." else input_text
                    
                    if not keywords_for_download and wordcloud_source_text.strip():
                        keywords_for_download = extract_keywords(wordcloud_source_text, num_keywords=50)
                        
                    wordcloud_fig = generate_wordcloud(wordcloud_source_text, keywords_for_download)
                    if wordcloud_fig:
                        st.pyplot(wordcloud_fig)
                    else:
                        st.info("Not enough keywords or suitable text to generate a meaningful word cloud. ℹ️")
                except Exception as e:
                    st.error(f"Error generating word cloud: {e} ❌")

    # --- Sentiment Analysis ---
    if "Sentiment Analysis" in analysis_options:
        with st.expander("😊 Sentiment Analysis", expanded=True):
            with st.spinner("Analyzing sentiment..."):
                try:
                    sentiment_result = sentiment_analyzer(input_text, truncation=True, max_length=512)[0]
                    label = sentiment_result['label']
                    score = sentiment_result['score']
                    st.success("Sentiment Detected: 🎉")
                    st.markdown(f"**Label:** `{label}` (Confidence: `{score:.2f}`)")
                    if label == "POSITIVE":
                        st.balloons()
                    elif label == "NEGATIVE":
                        st.snow()
                except Exception as e:
                    st.error(f"Error analyzing sentiment: {e} ❌")
                    st.warning("Sentiment analysis might be less accurate for very short or ambiguous texts. 🚧")

    # --- Emotion Detection (NEW FEATURE) ---
    if "Emotion Detection" in analysis_options:
        with st.expander("🎭 Emotion Detection", expanded=False):
            st.markdown("Identify specific emotions expressed in the text (e.g., joy, sadness, anger).")
            with st.spinner("Detecting emotions..."):
                try:
                    emotion_results_raw = emotion_analyzer(input_text, truncation=True, max_length=512)
                    if emotion_results_raw:
                        st.success("Emotions Detected: 🌈")
                        emotion_data = []
                        for emotion_item in emotion_results_raw[0]: # emotion_results_raw is usually a list of lists
                            emotion_data.append({
                                "Emotion": emotion_item['label'].capitalize(),
                                "Score": f"{emotion_item['score']:.2f}"
                            })
                        st.table(emotion_data)
                        emotion_results = emotion_data # For download
                    else:
                        st.info("No specific emotions detected or text is too short. ℹ️")
                except Exception as e:
                    st.error(f"Error detecting emotions: {e} ❌")
                    st.warning("Emotion detection might be less accurate for very short or ambiguous texts. �")


    # --- Named Entity Recognition ---
    if "Named Entity Recognition" in analysis_options:
        with st.expander("👤🏢📍 Named Entity Recognition & Highlighting", expanded=True):
            with st.spinner("Identifying and highlighting entities..."):
                try:
                    entities, highlighted_text_html = perform_ner_and_highlight(input_text)
                    if entities:
                        st.success("Identified Entities: 🎯")
                        entity_data = [{"Entity": ent["text"], "Type": ent["label"], "Description": ent["explanation"]} for ent in entities]
                        st.table(entity_data)
                        st.markdown("---")
                        st.subheader("Text with Highlighted Entities:")
                        st.markdown(highlighted_text_html, unsafe_allow_html=True)
                    else:
                        st.info("No named entities found in the text. ℹ️")
                except Exception as e:
                    st.error(f"Error performing NER: {e} ❌")
                    st.warning("NER might not identify all entities accurately, especially in informal text. 🚧")

    # --- Text Generation ---
    if "Text Generation" in analysis_options:
        with st.expander("✍️ Text Generation/Completion", expanded=False):
            st.markdown("Enter a prompt and the model will try to complete it. 🤖")
            generation_prompt = st.text_area("Enter your prompt for text generation:", value=input_text[:100], height=100, key="gen_prompt")
            max_gen_length = st.slider("Max generated text length:", min_value=50, max_value=500, value=200, step=10)
            if st.button("Generate Text"):
                with st.spinner("Generating text..."):
                    try:
                        generated_text = text_generator(
                            generation_prompt,
                            max_length=max_gen_length,
                            num_return_sequences=1,
                            do_sample=True,
                            top_k=50,
                            top_p=0.95,
                            temperature=0.7,
                            no_repeat_ngram_size=2
                        )[0]['generated_text']
                        st.success("Generated Text: 📝")
                        st.write(generated_text)
                    except Exception as e:
                        st.error(f"Error generating text: {e} ❌")
                        st.warning("Text generation might produce nonsensical or repetitive results. 🚧")

    # --- Grammar & Style Correction (NEW FEATURE) ---
    if "Grammar & Style Correction" in analysis_options:
        with st.expander("✍️ Grammar & Style Correction", expanded=False):
            st.markdown("Corrects grammatical errors and suggests style improvements.")
            correction_text_input = st.text_area("Enter text for correction:", value=input_text, height=150, key="correction_input")
            if st.button("Correct Text"):
                with st.spinner("Correcting text..."):
                    try:
                        # Correction models often take input as a list and return a list
                        correction_output = grammar_corrector(correction_text_input, max_length=len(correction_text_input.split()) + 50, truncation=True)
                        if correction_output and correction_output[0]['generated_text']:
                            corrected_text = correction_output[0]['generated_text']
                            st.success("Corrected Text: ✅")
                            st.write(corrected_text)
                        else:
                            st.info("No significant corrections suggested or text is too short. ℹ️")
                    except Exception as e:
                        st.error(f"Error correcting text: {e} ❌")
                        st.warning("Correction might fail for very long texts or complex issues. 🚧")


    # --- Readability Score ---
    if "Readability Score" in analysis_options:
        with st.expander("📖 Readability Score", expanded=False):
            with st.spinner("Calculating readability..."):
                try:
                    fk_grade = flesch_kincaid_grade(input_text)
                    flesch_ease = flesch_reading_ease(input_text)
                    st.success("Readability Scores: ✅")
                    st.markdown(f"**Flesch-Kincaid Grade Level:** `{fk_grade:.2f}` (Approximate grade level needed to understand the text)")
                    st.markdown(f"**Flesch Reading Ease Score:** `{flesch_ease:.2f}` (Higher score means easier to read)")
                    st.info("""
                        Think of these scores like a report card for your text!
                        * **Flesch-Kincaid Grade Level:** This number tells you what school grade level (like 5th grade or 10th grade) someone would generally need to be in to easily understand your text.
                        * **Flesch Reading Ease Score:** This number tells you how "easy" your text is to read. Higher numbers mean it's super easy (like a storybook!), and lower numbers mean it's a bit trickier (like a science book).
                        These scores are just a guess, but they help you see if your writing is easy for others to understand! 📚
                    """)
                except Exception as e:
                    st.error(f"Error calculating readability: {e} ❌")
                    st.warning("Readability scores require sufficient text length to be accurate. 🚧")

    # --- Language Detection ---
    if "Language Detection" in analysis_options:
        with st.expander("🌐 Language Detection", expanded=False):
            with st.spinner("Detecting language..."):
                try:
                    detected_lang = detect(input_text)
                    st.success("Language Detected: 🗣️")
                    st.markdown(f"**Detected Language:** `{detected_lang.upper()}`")
                except Exception as e:
                    st.error(f"Error detecting language: {e} ❌")
                    st.warning("Language detection might fail for very short texts or mixed languages. 🚧")

    # --- Text Translation ---
    if "Translation" in analysis_options:
        with st.expander("🌍 Text Translation", expanded=False):
            st.markdown("Translate your text from English to another language. 💬")
            st.info("Please note: The translation model can process approximately **380 words** at a time. Longer texts will be truncated. ⚠️")
            
            available_languages = sorted([lang for lang, pipe in translators.items() if pipe is not None])
            if not available_languages:
                st.warning("No translation models were loaded successfully. Translation feature is unavailable. 🚫")
            else:
                target_lang = st.selectbox("Translate to:", available_languages, key="target_lang")
                
                if st.button("Translate Text"):
                    if translators[target_lang] is not None:
                        with st.spinner(f"Translating to {target_lang}..."):
                            try:
                                translation_result = translators[target_lang](
                                    input_text,
                                    max_length=512,
                                    truncation=True,
                                    do_sample=True,
                                    top_k=50,
                                    top_p=0.95,
                                    temperature=0.7,
                                    no_repeat_ngram_size=2
                                )[0]['translation_text']
                                st.success(f"Translated Text (to {target_lang}): ✨")
                                st.write(translation_result)
                            except Exception as e:
                                st.error(f"Error translating text to {target_lang}: {e} ❌")
                                st.warning("Translation might be limited by model's language pair and text length. 🚧")
                    else:
                        st.error(f"Translation model for {target_lang} was not loaded successfully. Cannot translate. 🚫")

    # --- Text Comparison ---
    if "Text Comparison" in analysis_options:
        with st.expander("🔍 Text Comparison (Similarity Score)", expanded=False):
            st.markdown("Compare the similarity between your main text and another piece of text. 🤝")
            text_to_compare = st.text_area("Paste text to compare with:", height=200, key="compare_text")
            
            if st.button("Compare Texts"):
                if input_text.strip() and text_to_compare.strip():
                    with st.spinner("Calculating similarity..."):
                        try:
                            # Preprocess texts
                            processed_text1 = preprocess_text_for_similarity(input_text)
                            processed_text2 = preprocess_text_for_similarity(text_to_compare)

                            if not processed_text1 or not processed_text2:
                                st.warning("Cannot compare: One or both texts are too short or contain no meaningful words after cleaning. 🚧")
                            else:
                                # Create TF-IDF vectors
                                vectorizer = TfidfVectorizer().fit([processed_text1, processed_text2])
                                text1_vector = vectorizer.transform([processed_text1])
                                text2_vector = vectorizer.transform([processed_text2])

                                # Calculate cosine similarity
                                similarity_score = cosine_similarity(text1_vector, text2_vector)[0][0]
                                text_comparison_result = f"Similarity Score: `{similarity_score:.2f}` (0 = no similarity, 1 = identical)"
                                st.success("Comparison Result: ✅")
                                st.markdown(text_comparison_result)
                                if similarity_score > 0.8:
                                    st.info("The texts are highly similar! ✨")
                                elif similarity_score > 0.5:
                                    st.info("The texts share a moderate level of similarity. 👍")
                                else:
                                    st.info("The texts have low similarity. 👎")
                        except Exception as e:
                            st.error(f"Error comparing texts: {e} ❌")
                            st.warning("Ensure both texts are long enough for meaningful comparison. 🚧")
                else:
                    st.warning("Please enter text in both boxes to perform comparison. ℹ️")

    # --- Sentence Similarity (NEW FEATURE) ---
    if "Sentence Similarity" in analysis_options:
        with st.expander("↔️ Sentence Similarity", expanded=False):
            st.markdown("Compare the semantic similarity between two individual sentences.")
            sentence1 = st.text_input("Enter Sentence 1:", key="sentence1_input")
            sentence2 = st.text_input("Enter Sentence 2:", key="sentence2_input")

            if st.button("Compare Sentences"):
                if sentence1.strip() and sentence2.strip():
                    with st.spinner("Calculating sentence similarity..."):
                        try:
                            # Generate embeddings for both sentences
                            embeddings1 = sentence_embedder.encode(sentence1, convert_to_tensor=True)
                            embeddings2 = sentence_embedder.encode(sentence2, convert_to_tensor=True)

                            # Calculate cosine similarity between embeddings
                            sentence_sim_score = cosine_similarity(embeddings1.reshape(1, -1), embeddings2.reshape(1, -1))[0][0]
                            sentence_similarity_result = f"Sentence Similarity Score: `{sentence_sim_score:.2f}` (0 = no similarity, 1 = identical meaning)"
                            st.success("Sentence Comparison Result: ✅")
                            st.markdown(sentence_similarity_result)
                            if sentence_sim_score > 0.8:
                                st.info("The sentences have very similar meanings! 🎯")
                            elif sentence_sim_score > 0.5:
                                st.info("The sentences share some semantic similarity. 🤔")
                            else:
                                st.info("The sentences have low semantic similarity. 🤷")
                        except Exception as e:
                            st.error(f"Error comparing sentences: {e} ❌")
                            st.warning("Ensure both sentences are valid. 🚧")
                else:
                    st.warning("Please enter both sentences to compare. ℹ️")


    # --- Download Results ---
    st.markdown("---")
    st.header("4. Download All Results 📥")
    if input_text:
        # Initialize variables for download section
        # (These are already initialized at the top of section 3, but re-assign for clarity if needed)
        
        all_results = {
            "Original Text": input_text,
            "Summary": summary_for_download,
            "Keywords": ", ".join(keywords_for_download) if isinstance(keywords_for_download, list) else keywords_for_download,
            "Sentiment": f"{label} (Confidence: {score:.2f})" if isinstance(score, float) else label,
            "Emotions": emotion_results, # New
            "Corrected Text": corrected_text, # New
            "Entities (Table)": entity_data,
            "Entities (Highlighted Text)": highlighted_text_html,
            "Generated Text": generated_text,
            "Flesch-Kincaid Grade Level": fk_grade,
            "Flesch Reading Ease Score": flesch_ease,
            "Detected Language": detected_lang.upper() if isinstance(detected_lang, str) else detected_lang,
            "Translated Text": translation_result,
            "Text Comparison Result": text_comparison_result,
            "Sentence Similarity Result": sentence_similarity_result, # New
        }

        download_string = ""
        for key, value in all_results.items():
            download_string += f"--- {key} ---\n"
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        download_string += f"{item}\n"
                    else:
                        download_string += f"{item}\n"
            else:
                download_string += f"{value}\n"
            download_string += "\n"

        st.download_button(
            label="Download Analysis Results (TXT)",
            data=download_string,
            file_name="text_analysis_results.txt",
            mime="text/plain"
        )
    else:
        st.info("Enter text to enable result download. ℹ️")

st.markdown("---")
st.caption("Developed with Streamlit, Hugging Face Transformers, NLTK, spaCy, scikit-learn, textstat, langdetect, wordcloud, and sentence-transformers.")
