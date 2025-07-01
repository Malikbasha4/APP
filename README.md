# APP
Advanced Intelligent Text Analysis Workbench
This Streamlit application is a powerful tool for comprehensive text analysis, leveraging both traditional Machine Learning (ML) and state-of-the-art Deep Learning (DL) models. It allows users to gain deep insights into textual data by performing various Natural Language Processing (NLP) tasks.

🌟 What the App Does
The "Advanced Intelligent Text Analysis Workbench" provides the following functionalities:

Text Summarization (Deep Learning): Generates concise summaries of longer texts, powered by a transformer-based deep learning model.

Keyword/Key Phrase Extraction (NLP): Identifies the most relevant keywords and phrases from the input text using frequency analysis and stopword filtering.

Sentiment Analysis (ML/NLP): Determines the emotional tone of the text (positive, negative, or neutral).

Named Entity Recognition (NER) (Advanced NLP): Automatically identifies and categorizes named entities such as people, organizations, locations, dates, and more.

Text Generation/Completion (Deep Learning): Given a prompt, the app can generate coherent and contextually relevant text to complete the input.

Topic Modeling (Machine Learning): Discovers the underlying abstract "topics" that occur in a collection of documents (or a single long document), represented by a cluster of keywords.

Readability Score (NLP): Calculates metrics like Flesch-Kincaid Grade Level and Flesch Reading Ease to assess how easy the text is to read and understand.

Language Detection (NLP): Automatically identifies the language of the input text.

Text Translation (Deep Learning): Translates the input text from English to French (can be extended to other languages).

Word Cloud Visualization: Generates a visual representation of the most frequent and important words in the text, emphasizing keywords.

Download Results: Allows users to download all generated analysis results as a plain text file.

🚀 Setup and Installation
To run this application, you need to have Python installed on your system.

Save the App Code:
Save the provided Python code into a file named app.py (or any other .py file).

Create requirements.txt:
In the same directory as your app.py file, create a file named requirements.txt and add the following content:

streamlit
transformers
nltk
spacy
scikit-learn
textstat
langdetect
wordcloud
matplotlib
torch
sentencepiece

Install Dependencies:
Open your terminal or command prompt, navigate to the directory where you saved app.py and requirements.txt, and run the following command to install all required Python libraries:

pip install -r requirements.txt

Download SpaCy Language Model:
The spaCy library requires a specific English language model. After installing the requirements, run this command:

python -m spacy download en_core_web_sm

🏃 How to Run the App
Once all dependencies are installed, you can launch the Streamlit application:

Open your terminal or command prompt.

Navigate to the directory where your app.py file is located.

Execute the following command:

streamlit run app.py

Your default web browser should automatically open to the Streamlit application (usually at http://localhost:8501). If it doesn't, copy the URL displayed in your terminal and paste it into your browser.

💡 How to Use the App
The application is designed for intuitive use:

Enter Your Text:

On the main page, you will see a large text area labeled "Paste your text here:".

Type or paste any text you wish to analyze into this area (e.g., an article, a speech, a product review, etc.). The app will automatically start processing once text is entered.

Select Analysis Features (Sidebar):

On the left sidebar, you'll find a section titled "⚙️ Settings & Advanced Features".

Use the "Select Analysis Features:" multi-select box to choose which specific analyses you want to perform (e.g., Summarization, Sentiment Analysis, Topic Modeling). By default, some common features are pre-selected.

View Analysis Results:

After you enter text, the "2. Analysis Results" section will appear below the input area.

Each selected analysis feature will have its own expandable section (e.g., "📝 Text Summary (Deep Learning)"). Click on the header of each section to expand or collapse the results.

You will see loading spinners (st.spinner) while the models are processing, indicating that the analysis is in progress.

Results will be displayed within their respective sections. For example, text summaries, keyword lists, sentiment labels, and entity tables.

Interact with Specific Features:

Text Generation: If selected, you'll find a separate text area for a "prompt" and a slider to control the maximum length of the generated text. Click "Generate Text" to see the output.

Topic Modeling: Adjust the "Number of topics to find" and "Number of keywords per topic" sliders to refine the topic extraction.

Translation: Select the target language (currently French) and click "Translate Text".

Clear Text:

Below the main text input area, there's a "Clear Text" button. Click this to clear the current input and start a new analysis.

Download Results:

At the bottom of the app, there's a "3. Download Results" section.

Click the "Download Analysis Results (TXT)" button to save all the analysis outputs into a single text file on your computer.

Enjoy exploring the capabilities of your Advanced Intelligent Text Analysis Workbench!