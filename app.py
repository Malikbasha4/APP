import streamlit as st

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from collections import Counter

import spacy

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import NMF # Non-negative Matrix Factorization for topic modeling

from textstat import flesch_kincaid_grade, flesch_reading_ease # For readability

from langdetect import detect, DetectorFactory # For language detection

from wordcloud import WordCloud # For word cloud visualization

import matplotlib.pyplot as plt

import io # For handling byte streams for downloads



# --- Set random seed for reproducibility of langdetect ---

# This makes langdetect results consistent across runs

DetectorFactory.seed = 0



# --- NLTK Downloads (Run unconditionally at the start) ---

# This ensures the necessary NLTK data is downloaded before any other NLTK operations.

# Removed st.info here as the downloads are quiet and don't need to show on UI

nltk.download('stopwords', quiet=True) # Use quiet=True to suppress verbose output if not needed

nltk.download('punkt', quiet=True)

nltk.download('punkt_tab', quiet=True) # Added download for 'punkt_tab'



# --- Load Models (Cached for performance) ---

# This decorator ensures the function runs only once and caches the result

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

    # Removed st.info and st.write calls from here to keep the UI clean during loading

    translation_pipelines = {}

   

    # Define language pairs and their corresponding Hugging Face model names

    # Using Helsinki-NLP/opus-mt for various language pairs

    # Removed Hindi, Japanese, Korean, Tamil, and Bengali models due to loading issues or user request.

    language_models = {

        "French": "Helsinki-NLP/opus-mt-en-fr",

        "Chinese": "Helsinki-NLP/opus-mt-en-zh",

        "German": "Helsinki-NLP/opus-mt-en-de",

        "Arabic": "Helsinki-NLP/opus-mt-en-ar",

        "Spanish": "Helsinki-NLP/opus-mt-en-es",

        "Portuguese": "Helsinki-NLP/opus-mt-en-pt", # Re-added Portuguese

        "Russian": "Helsinki-NLP/opus-mt-en-ru",

    }



    for lang_name, model_name in language_models.items():

        try:

            tokenizer = AutoTokenizer.from_pretrained(model_name)

            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            translation_pipelines[lang_name] = pipeline("translation", model=model, tokenizer=tokenizer)

            # st.write(f"Loaded English to {lang_name} model.") # Removed this line

        except Exception as e:

            # st.error(f"Could not load translation model for {lang_name} ({model_name}): {e}") # Removed this line

            # st.warning(f"Please ensure 'sentencepiece' is installed and check model availability for {lang_name}.") # Removed this line

            translation_pipelines[lang_name] = None # Mark as None if loading fails

           

    return translation_pipelines



# Initialize models

with st.spinner("Initializing models... This might take a moment."):

    summarizer = load_summarizer_model()

    sentiment_analyzer = load_sentiment_model()

    nlp_spacy = load_spacy_model()

    text_generator = load_text_generation_model()

    translators = load_translation_models() # Now loads multiple translators

# Removed st.success("Models initialized!") to keep the UI cleaner after loading

# The spinner disappearing is enough indication of completion.



# --- Helper Functions ---



def extract_keywords(text, num_keywords=10):

    """

    Extracts keywords using NLTK tokenization and frequency counting.

    Filters out stopwords and punctuation.

    """

    words = word_tokenize(text.lower())

    stop_words = set(stopwords.words('english'))

    # Filter out stopwords and non-alphabetic tokens

    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]

    word_counts = Counter(filtered_words)

    return [word for word, count in word_counts.most_common(num_keywords)]



def perform_ner(text):

    """

    Performs Named Entity Recognition using spaCy.

    Returns a list of dictionaries with entity text, label, and explanation.

    """

    doc = nlp_spacy(text)

    entities = []

    for ent in doc.ents:

        entities.append({

            "text": ent.text,

            "label": ent.label_,

            "explanation": spacy.explain(ent.label_)

        })

    return entities



def perform_topic_modeling(text, num_topics=3, num_words=5):

    """

    Performs topic modeling using NMF.

    Returns a list of dominant topics with their keywords.

    """

    if not text.strip():

        return []



    # Ensure text is long enough for meaningful topic modeling

    if len(text.split()) < 50: # Arbitrary threshold for meaningful topic modeling

        return []



    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')

    try:

        dtm = vectorizer.fit_transform([text])

    except ValueError: # Handle cases where text is too short after stopword removal

        return []



    # Check if dtm has enough features for NMF

    if dtm.shape[1] < num_topics:

        return []



    nmf_model = NMF(n_components=num_topics, random_state=1, alpha_W=0.01, alpha_H=0.01)

    nmf_model.fit(dtm)



    feature_names = vectorizer.get_feature_names_out()

    topics = []

    for topic_idx, topic in enumerate(nmf_model.components_):

        top_words = [feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]

        topics.append(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

    return topics



def generate_wordcloud(text, keywords):

    """Generates a word cloud image from the text, focusing on keywords."""

    if not keywords:

        return None



    # Create a frequency dictionary from keywords to emphasize them

    word_freq = Counter(keywords)

    # Add other words from the text but with lower weight

    words = word_tokenize(text.lower())

    stop_words = set(stopwords.words('english'))

    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]

    for word in filtered_words:

        if word not in word_freq:

            word_freq[word] = 1 # Give a base frequency if not a top keyword



    wordcloud = WordCloud(width=800, height=400, background_color='white',

                          max_words=50, collocations=False).generate_from_frequencies(word_freq)



    fig, ax = plt.subplots(figsize=(10, 5))

    ax.imshow(wordcloud, interpolation='bilinear')

    ax.axis('off')

    return fig



# --- Streamlit UI ---



st.set_page_config(layout="wide", page_title="Advanced Intelligent Text Analysis Workbench")



st.title("🧠 Advanced Intelligent Text Analysis Workbench")

st.markdown("""

    This powerful app leverages advanced Machine Learning and Deep Learning models

    to provide comprehensive insights into your text.

    It offers summarization, keyword extraction, sentiment analysis, named entity recognition,

    text generation, topic modeling, readability scores, language detection, and translation.

""")



# Sidebar for additional features and settings

st.sidebar.header("⚙️ Settings & Advanced Features")

analysis_options = st.sidebar.multiselect(

    "Select Analysis Features:",

    ["Summarization", "Keyword Extraction", "Sentiment Analysis", "Named Entity Recognition",

     "Text Generation", "Topic Modeling", "Readability Score", "Language Detection", "Translation", "Word Cloud"],

    default=["Summarization", "Keyword Extraction", "Sentiment Analysis", "Named Entity Recognition"]

)



# Text Input Area

st.header("1. Enter Your Text")

input_text = st.text_area(

    "Paste your text here:",

    height=300,

    placeholder="Type or paste any text you want to analyze (e.g., an article, a speech, a review)...",

    key="main_input_text" # Added key for better state management

)



# Clear Text Button

if st.button("Clear Text", help="Clears the input text area"):

    st.session_state.main_input_text = "" # Clear the text area

    st.experimental_rerun() # Rerun to reflect the cleared text



if input_text:

    st.markdown("---")

    st.header("2. Analysis Results")



    # --- Summarization ---

    if "Summarization" in analysis_options:

        with st.expander("📝 Text Summary", expanded=True): # Removed (Deep Learning)

            with st.spinner("Generating summary..."):

                try:

                    # Min/Max length for summarization, adjusted dynamically

                    words_in_text = len(input_text.split())

                    min_len = min(50, words_in_text // 4)

                    max_len = min(200, words_in_text // 2)

                    if words_in_text < 50: # Summarization models struggle with very short texts

                        st.warning("Text is too short for effective summarization. Minimum recommended words: 50.")

                        summary = "Not enough text to generate a meaningful summary."

                    else:

                        summary = summarizer(input_text, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']

                    st.success("Summary Generated:")

                    st.write(summary)

                except Exception as e:

                    st.error(f"Error generating summary: {e}")

                    st.warning("Summarization might fail for very short texts or due to model limitations.")



    # --- Keyword Extraction ---

    if "Keyword Extraction" in analysis_options:

        with st.expander("🔑 Keywords/Key Phrases", expanded=True): # Removed (NLP)

            with st.spinner("Extracting keywords..."):

                try:

                    keywords = extract_keywords(input_text)

                    if keywords:

                        st.success("Extracted Keywords:")

                        st.markdown(f"**`{', '.join(keywords)}`**")

                    else:

                        st.info("No significant keywords found or text is too short.")

                except Exception as e:

                    st.error(f"Error extracting keywords: {e}")



    # --- Word Cloud ---

    if "Word Cloud" in analysis_options and "Keyword Extraction" in analysis_options:

        with st.expander("☁️ Word Cloud Visualization", expanded=True):

            with st.spinner("Generating word cloud..."):

                try:

                    keywords_for_cloud = extract_keywords(input_text, num_keywords=50) # Get more keywords for cloud

                    wordcloud_fig = generate_wordcloud(input_text, keywords_for_cloud)

                    if wordcloud_fig:

                        st.pyplot(wordcloud_fig)

                    else:

                        st.info("Not enough keywords to generate a meaningful word cloud.")

                except Exception as e:

                    st.error(f"Error generating word cloud: {e}")



    # --- Sentiment Analysis ---

    if "Sentiment Analysis" in analysis_options:

        with st.expander("😊 Sentiment Analysis", expanded=True): # Removed (ML/NLP)

            with st.spinner("Analyzing sentiment..."):

                try:

                    # Added truncation and max_length to handle long inputs

                    sentiment_result = sentiment_analyzer(input_text, truncation=True, max_length=512)[0]

                    label = sentiment_result['label']

                    score = sentiment_result['score']

                    st.success("Sentiment Detected:")

                    st.markdown(f"**Label:** `{label}` (Confidence: `{score:.2f}`)")

                    if label == "POSITIVE":

                        st.balloons()

                    elif label == "NEGATIVE":

                        st.snow()

                except Exception as e:

                    st.error(f"Error analyzing sentiment: {e}")

                    st.warning("Sentiment analysis might be less accurate for very short or ambiguous texts.")



    # --- Named Entity Recognition ---

    if "Named Entity Recognition" in analysis_options:

        with st.expander("👤🏢📍 Named Entity Recognition", expanded=True): # Removed (Advanced NLP)

            with st.spinner("Identifying entities..."):

                try:

                    entities = perform_ner(input_text)

                    if entities:

                        st.success("Identified Entities:")

                        entity_data = [{"Entity": ent["text"], "Type": ent["label"], "Description": ent["explanation"]} for ent in entities]

                        st.table(entity_data)

                    else:

                        st.info("No named entities found in the text.")

                except Exception as e:

                    st.error(f"Error performing NER: {e}")

                    st.warning("NER might not identify all entities accurately, especially in informal text.")



    # --- Text Generation ---

    if "Text Generation" in analysis_options:

        with st.expander("✍️ Text Generation/Completion", expanded=False): # Removed (Deep Learning)

            st.markdown("Enter a prompt and the model will try to complete it.")

            generation_prompt = st.text_area("Enter your prompt for text generation:", value=input_text[:100], height=100, key="gen_prompt")

            max_gen_length = st.slider("Max generated text length:", min_value=50, max_value=500, value=200, step=10)

            if st.button("Generate Text"):

                with st.spinner("Generating text..."):

                    try:

                        # Modified text generation parameters for more diverse output

                        generated_text = text_generator(

                            generation_prompt,

                            max_length=max_gen_length,

                            num_return_sequences=1,

                            do_sample=True,  # Enable sampling for more varied output

                            top_k=50,        # Consider only the top 50 most likely words

                            top_p=0.95,      # Sample from the smallest set of words whose cumulative probability exceeds 0.95

                            temperature=0.7, # Control randomness (lower = more predictable, higher = more creative)

                            no_repeat_ngram_size=2 # Prevent repeating 2-word sequences

                        )[0]['generated_text']

                        st.success("Generated Text:")

                        st.write(generated_text)

                    except Exception as e:

                        st.error(f"Error generating text: {e}")

                        st.warning("Text generation might produce nonsensical or repetitive results.")



    # --- Topic Modeling ---

    if "Topic Modeling" in analysis_options:

        with st.expander("📊 Topic Modeling", expanded=False): # Removed (Machine Learning)

            num_topics = st.slider("Number of topics to find:", min_value=1, max_value=10, value=3)

            num_topic_words = st.slider("Number of keywords per topic:", min_value=3, max_value=10, value=5)

            with st.spinner("Performing topic modeling..."):

                try:

                    topics = perform_topic_modeling(input_text, num_topics, num_topic_words)

                    if topics:

                        st.success("Identified Topics:")

                        for topic in topics:

                            st.write(f"- {topic}")

                    else:

                        st.info("Not enough text or too few unique words to perform meaningful topic modeling.")

                except Exception as e:

                    st.error(f"Error performing topic modeling: {e}")

                    st.warning("Topic modeling works best on longer, coherent texts.")



    # --- Readability Score ---

    if "Readability Score" in analysis_options:

        with st.expander("📖 Readability Score", expanded=False): # Removed (NLP)

            with st.spinner("Calculating readability..."):

                try:

                    fk_grade = flesch_kincaid_grade(input_text)

                    flesch_ease = flesch_reading_ease(input_text)

                    st.success("Readability Scores:")

                    st.markdown(f"**Flesch-Kincaid Grade Level:** `{fk_grade:.2f}` (Approximate grade level needed to understand the text)")

                    st.markdown(f"**Flesch Reading Ease Score:** `{flesch_ease:.2f}` (Higher score means easier to read)")

                    # Simplified explanation for a 10-year-old

                    st.info("""

                        Think of these scores like a report card for your text!

                        * **Flesch-Kincaid Grade Level:** This number tells you what school grade level (like 5th grade or 10th grade) someone would generally need to be in to easily understand your text.

                        * **Flesch Reading Ease Score:** This number tells you how "easy" your text is to read. Higher numbers mean it's super easy (like a storybook!), and lower numbers mean it's a bit trickier (like a science book).

                        These scores are just a guess, but they help you see if your writing is easy for others to understand!

                    """)

                except Exception as e:

                    st.error(f"Error calculating readability: {e}")

                    st.warning("Readability scores require sufficient text length to be accurate.")



    # --- Language Detection ---

    if "Language Detection" in analysis_options:

        with st.expander("🌐 Language Detection", expanded=False): # Removed (NLP)

            with st.spinner("Detecting language..."):

                try:

                    detected_lang = detect(input_text)

                    st.success("Language Detected:")

                    st.markdown(f"**Detected Language:** `{detected_lang.upper()}`")

                except Exception as e:

                    st.error(f"Error detecting language: {e}")

                    st.warning("Language detection might fail for very short texts or mixed languages.")



    # --- Text Translation ---

    if "Translation" in analysis_options:

        with st.expander("🌍 Text Translation", expanded=False): # Removed (Deep Learning)

            st.markdown("Translate your text from English to another language.")

           

            # Get available languages from the loaded models

            available_languages = sorted([lang for lang, pipe in translators.items() if pipe is not None])

            if not available_languages:

                st.warning("No translation models were loaded successfully. Translation feature is unavailable.")

            else:

                target_lang = st.selectbox("Translate to:", available_languages, key="target_lang")

               

                if st.button("Translate Text"):

                    if translators[target_lang] is not None:

                        with st.spinner(f"Translating to {target_lang}..."):

                            try:

                                # Use the specific translator pipeline for the selected language

                                # Added truncation=True to handle long inputs

                                translation_result = translators[target_lang](

                                    input_text,

                                    max_length=512,

                                    truncation=True,

                                    do_sample=True, # Enable sampling for more varied output

                                    top_k=50,       # Consider only the top 50 most likely words

                                    top_p=0.95,     # Sample from the smallest set of words whose cumulative probability exceeds 0.95

                                    temperature=0.7,# Control randomness (lower = more predictable, higher = more creative)

                                    no_repeat_ngram_size=2 # Prevent repeating 2-word sequences

                                )[0]['translation_text']

                                st.success(f"Translated Text (to {target_lang}):")

                                st.write(translation_result)

                            except Exception as e:

                                st.error(f"Error translating text to {target_lang}: {e}")

                                st.warning("Translation might be limited by model's language pair and text length.")

                    else:

                        st.error(f"Translation model for {target_lang} was not loaded successfully. Cannot translate.")





    # --- Download Results ---

    st.markdown("---")

    st.header("3. Download Results")

    if input_text:

        # Initialize variables that might not be set if their features are not selected

        summary = "N/A"

        keywords = []

        label = "N/A"

        score = "N/A"

        entity_data = "N/A"

        generated_text = "N/A"

        topics = "N/A"

        fk_grade = "N/A"

        flesch_ease = "N/A"

        detected_lang = "N/A"

        translation_result = "N/A"



        # Re-assign if features were selected and results were generated

        if "Summarization" in analysis_options and 'summary' in locals():

            summary = locals()['summary']

        if "Keyword Extraction" in analysis_options and 'keywords' in locals():

            keywords = locals()['keywords']

        if "Sentiment Analysis" in analysis_options and 'label' in locals():

            label = locals()['label']

            score = locals()['score']

        if "Named Entity Recognition" in analysis_options and 'entity_data' in locals():

            entity_data = locals()['entity_data']

        if "Text Generation" in analysis_options and 'generated_text' in locals():

            generated_text = locals()['generated_text']

        if "Topic Modeling" in analysis_options and 'topics' in locals():

            topics = locals()['topics']

        if "Readability Score" in analysis_options and 'fk_grade' in locals():

            fk_grade = locals()['fk_grade']

            flesch_ease = locals()['flesch_ease']

        if "Language Detection" in analysis_options and 'detected_lang' in locals():

            detected_lang = locals()['detected_lang']

        if "Translation" in analysis_options and 'translation_result' in locals():

            translation_result = locals()['translation_result']





        all_results = {

            "Original Text": input_text,

            "Summary": summary,

            "Keywords": ", ".join(keywords) if isinstance(keywords, list) else keywords,

            "Sentiment": f"{label} (Confidence: {score:.2f})" if isinstance(score, float) else label,

            "Entities": entity_data,

            "Generated Text": generated_text,

            "Topics": topics,

            "Flesch-Kincaid Grade Level": fk_grade,

            "Flesch Reading Ease Score": flesch_ease,

            "Detected Language": detected_lang.upper() if isinstance(detected_lang, str) else detected_lang,

            "Translated Text": translation_result,

        }



        # Format results for download

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

        st.info("Enter text to enable result download.")



else:

    st.info("Please enter some text above to start the analysis.")



st.markdown("---")

st.caption("Developed with Streamlit, Hugging Face Transformers, NLTK, spaCy, scikit-learn, textstat, langdetect, and wordcloud.")
