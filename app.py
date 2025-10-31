import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gensim
import string
import nltk
import torch
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from textattack.augmentation import Augmenter
from textattack.transformations.word_swaps import WordSwapEmbedding, WordSwapWordNet
from textattack.transformations.word_swaps import WordSwapRandomCharacterDeletion # 3. THIS IS THE FIX
from textattack.transformations.word_swaps.word_swap_qwerty import WordSwapQWERTY
import lime
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 0. PAGE CONFIG & HELPER FUNCTIONS
# ----------------------------------------------------------------------

# Set Streamlit page config
st.set_page_config(
    page_title="The Model Interrogator",
    page_icon="🔬",
    layout="wide",
)

# Download NLTK data (if not already downloaded)
# We do this once at the top
@st.cache_data
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    return True

download_nltk_data()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Applies all preprocessing steps to a single string of text
    for Models 1, 2, 3, 4.
    """
    text = text.lower()
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# ----------------------------------------------------------------------
# 1. LOAD ALL 5 MODELS (using st.cache_resource)
# ----------------------------------------------------------------------

# --- Model 1: Naive Bayes ---
@st.cache_resource
def load_model_1():
    print("Loading Model 1 (NB)...")
    model = joblib.load('model_1_nb.pkl')
    return model

# --- Model 2: Logistic Regression ---
@st.cache_resource
def load_model_2():
    print("Loading Model 2 (LogReg)...")
    model = joblib.load('model_2_lr.pkl')
    return model

# --- Model 3: Word2Vec + LogReg ---
@st.cache_resource
def load_model_3():
    print("Loading Model 3 (W2V + LogReg)...")
    w2v_model = gensim.models.Word2Vec.load('model_3_w2v.model')
    lr_model = joblib.load('model_3_lr.pkl')
    return w2v_model, lr_model

def vectorize_complaints(text, model):
    """Helper function for Model 3. Averages word vectors for a single complaint."""
    tokens = clean_text(text).split()
    word_vectors = [model.wv[word] for word in tokens if word in model.wv]
    
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0).reshape(1, -1)
    else:
        return np.zeros((1, model.vector_size))

# --- Model 4: LSTM (Keras) ---
@st.cache_resource
def load_model_4():
    print("Loading Model 4 (LSTM)...")
    model = tf.keras.models.load_model('model_4_lstm.h5')
    tokenizer = joblib.load('model_4_keras_tokenizer.pkl')
    encoder = joblib.load('model_4_label_encoder.pkl')
    return model, tokenizer, encoder

# --- Model 5: DistilBERT (Transformer) ---
@st.cache_resource
def load_model_5():
    print("Loading Model 5 (DistilBERT)...")
    model_dir = 'model_5_distilbert_final'
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    # --- ADDED: Move model to GPU if available, once ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

# --- Load all models into memory ---
model_1 = load_model_1()
model_2 = load_model_2()
w2v_model, model_3_lr = load_model_3()
model_4_lstm, model_4_tokenizer, model_4_encoder = load_model_4()
model_5_bert, model_5_tokenizer, bert_device = load_model_5() # Get device

# Get class names from one of the models (e.g., Model 1)
CLASS_NAMES = list(model_1.classes_)
BERT_CLASS_NAMES = [model_5_bert.config.id2label[i] for i in range(len(model_5_bert.config.id2label))]

# ----------------------------------------------------------------------
# 2. PREDICTION HELPER FUNCTIONS
# ----------------------------------------------------------------------

def get_model_1_prediction(text):
    """Returns (class_name, confidence_score) for Model 1"""
    probabilities = model_1.predict_proba([text])[0]
    top_class_index = np.argmax(probabilities)
    return model_1.classes_[top_class_index], float(probabilities[top_class_index])

def get_model_2_prediction(text):
    """Returns (class_name, confidence_score) for Model 2"""
    probabilities = model_2.predict_proba([text])[0]
    top_class_index = np.argmax(probabilities)
    return model_2.classes_[top_class_index], float(probabilities[top_class_index])

def get_model_3_prediction(text):
    """Returns (class_name, confidence_score) for Model 3"""
    vectorized_text = vectorize_complaints(text, w2v_model)
    probabilities = model_3_lr.predict_proba(vectorized_text)[0]
    top_class_index = np.argmax(probabilities)
    return model_3_lr.classes_[top_class_index], float(probabilities[top_class_index])

def get_model_4_prediction(text):
    """Returns (class_name, confidence_score) for Model 4"""
    cleaned = clean_text(text)
    seq = model_4_tokenizer.texts_to_sequences([cleaned])
    padded_seq = pad_sequences(seq, maxlen=150) # 150 was our MAX_SEQ_LENGTH
    probabilities = model_4_lstm.predict_on_batch(padded_seq)[0]
    top_class_index = np.argmax(probabilities)
    return model_4_encoder.classes_[top_class_index], float(probabilities[top_class_index])

def get_model_5_prediction(text):
    """Returns (class_name, confidence_score) for Model 5"""
    inputs = model_5_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(bert_device)
    with torch.no_grad():
        logits = model_5_bert(**inputs).logits
    
    probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
    top_class_index = torch.argmax(probabilities).item()
    # Move probs to CPU for numpy/float conversion
    probabilities = probabilities.cpu() 
    return model_5_bert.config.id2label[top_class_index], probabilities[top_class_index].item()

# ----------------------------------------------------------------------
# 3. LIME EXPLANATION FUNCTION (ONLY FOR MODEL 2 NOW)
# ----------------------------------------------------------------------

def sklearn_predictor(texts):
    """Wrapper for LIME to use with sklearn pipelines (Model 2)"""
    cleaned_texts = [clean_text(text) for text in texts]
    # Return probabilities specifically for model 2
    return model_2.predict_proba(cleaned_texts)

@st.cache_data(max_entries=10) # Reduced cache size
def generate_lime_explanation(text):
    """Generates a LIME explanation HTML string for Model 2"""
    try:
        explainer = LimeTextExplainer(class_names=CLASS_NAMES)
        explanation = explainer.explain_instance(
            text, 
            sklearn_predictor, 
            num_features=10, 
            top_labels=1
        )
        return explanation.as_html()
    except Exception as e:
        print(f"LIME Error (Model 2): {e}")
        return "<p>Could not generate LIME explanation for Model 2.</p>"

# --- REMOVED LIME FUNCTIONS FOR BERT (MODEL 5) ---

# ----------------------------------------------------------------------
# 4. STREAMLIT UI LAYOUT
# ----------------------------------------------------------------------

st.title("🔬 The Model Interrogator")
st.subheader("A Comparative Study of NLP Robustness for Public Grievance Classification")

st.markdown("""
Welcome! This app stress-tests 5 different NLP models. 
1.  **Choose an ambiguous complaint** (or write your own).
2.  **Select the 'Ground Truth'** (it's set automatically for examples).
3.  **Select an attack** and hit **"Run Interrogation!"** to see which models are fooled.
""")

# --- Input Area ---
COMPLAINT_EXAMPLES = {
    "Debt vs. Credit": {
        "text": "I am getting harassing calls about a bill that I already paid off. This is wrong. It is hurting my credit score and I need this removed from my file immediately.",
        "truth": "Debt collection"
    },
    "Mortgage vs. Bank": {
        "text": "I tried to make my mortgage payment, but the money was taken from my savings account twice! The bank is now charging me late fees and it's a mess.",
        "truth": "Mortgage"
    },
    "Credit vs. Debt vs. Bank": {
        "text": "My bank account was charged for a debt I don't recognize. This overdraft is now on my credit report, and the bank won't help me.",
        "truth": "Credit Reporting" # Tricky, but 'credit report' is the final problem
    },
    "Mortgage vs. Credit vs. Debt": {
        "text": "I was denied a mortgage application because of a 'serious delinquency' on my credit file. This is an old medical bill that was sent to collections by mistake. I need this fixed.",
        "truth": "Credit Reporting"
    }
}

selected_complaint_key = st.selectbox(
    "Choose an ambiguous complaint example (or write your own below):",
    options=list(COMPLAINT_EXAMPLES.keys())
)

example_text = COMPLAINT_EXAMPLES[selected_complaint_key]["text"]
truth_label = COMPLAINT_EXAMPLES[selected_complaint_key]["truth"]

all_labels = ["None"] + CLASS_NAMES
default_index = all_labels.index(truth_label) if truth_label in all_labels else 0

selected_truth = st.selectbox(
    "Select Ground Truth (if known):",
    options=all_labels,
    index=default_index,
    help="This is the 'correct' answer. It is set automatically for the examples."
)

user_input = st.text_area(
    "Enter a citizen complaint:", 
    example_text, 
    height=150
)

attack_type = st.selectbox(
    "Select an Attack Method:",
    ["WordSwap (WordNet)", "WordSwap (Synonym)", "WordSwap (QWERTY)", "CharDelete (Typo)"],
    help="WordNet is a stronger synonym attack. QWERTY simulates keyboard mistakes. CharDelete is a typo."
)

if st.button("Run Interrogation!", type="primary"):
    
    # --- 1. Run the Attack ---
    with st.spinner("Attacking text..."):
        if attack_type == "WordSwap (Synonym)":
            transformation = WordSwapEmbedding()
        elif attack_type == "WordSwap (WordNet)":
            transformation = WordSwapWordNet()
        elif attack_type == "WordSwap (QWERTY)":
            transformation = WordSwapQWERTY()
        else: # CharDelete
            transformation = WordSwapRandomCharacterDeletion(random_one=True)
            
        augmenter = Augmenter(transformation=transformation)
            
        attacked_text = augmenter.augment(user_input)
        attacked_text = attacked_text[0] if isinstance(attacked_text, list) and len(attacked_text) > 0 else user_input

    # --- 2. Get All Predictions (Original & Attacked) ---
    with st.spinner("Running all 5 models... This may take a moment."):
        # Original Predictions
        pred_1, conf_1 = get_model_1_prediction(user_input)
        pred_2, conf_2 = get_model_2_prediction(user_input)
        pred_3, conf_3 = get_model_3_prediction(user_input)
        pred_4, conf_4 = get_model_4_prediction(user_input)
        pred_5, conf_5 = get_model_5_prediction(user_input)
        
        # Attacked Predictions
        atk_pred_1, atk_conf_1 = get_model_1_prediction(attacked_text)
        atk_pred_2, atk_conf_2 = get_model_2_prediction(attacked_text)
        atk_pred_3, atk_conf_3 = get_model_3_prediction(attacked_text)
        atk_pred_4, atk_conf_4 = get_model_4_prediction(attacked_text)
        atk_pred_5, atk_conf_5 = get_model_5_prediction(attacked_text)
        
        # --- EDITED: Only generate LIME for Model 2 ---
        lime_html_model_2 = generate_lime_explanation(user_input)
        # --- REMOVED BERT LIME GENERATION ---

    st.success("Interrogation Complete!")

    # --- 3. Show Attacked Text ---
    st.subheader("Attacked Text:")
    st.info(attacked_text)

    # --- 4. Display the "Aww-Factor" Grid ---
    st.subheader("Model Gauntlet Results:")
    
    def get_color(pred, truth_label):
        """Compares prediction to the ground truth and returns a color."""
        if truth_label == "None":
            return "grey" 
        return "green" if pred == truth_label else "red"

    # --- ROW 1: Naive Bayes vs. LogReg ---
    col1, col2 = st.columns(2)
    with col1:
        st.header("Model 1: Naive Bayes")
        st.markdown(f"**:color[{get_color(pred_1, selected_truth)}][Original:]** `{pred_1}`")
        st.progress(conf_1)
        st.markdown(f"**:color[{get_color(atk_pred_1, selected_truth)}][Attacked:]** `{atk_pred_1}`")
        st.progress(atk_conf_1)

    with col2:
        st.header("Model 2: LogReg + TF-IDF")
        st.markdown(f"**:color[{get_color(pred_2, selected_truth)}][Original:]** `{pred_2}`")
        st.progress(conf_2)
        st.markdown(f"**:color[{get_color(atk_pred_2, selected_truth)}][Attacked:]** `{atk_pred_2}`")
        st.progress(atk_conf_2)

    st.divider()

    # --- ROW 2: W2V vs. LSTM ---
    col3, col4 = st.columns(2)
    with col3:
        st.header("Model 3: LogReg + W2V (Avg)")
        st.markdown(f"**:color[{get_color(pred_3, selected_truth)}][Original:]** `{pred_3}`")
        st.progress(conf_3)
        st.markdown(f"**:color[{get_color(atk_pred_3, selected_truth)}][Attacked:]** `{atk_pred_3}`")
        st.progress(atk_conf_3)

    with col4:
        st.header("Model 4: LSTM + W2V")
        st.markdown(f"**:color[{get_color(pred_4, selected_truth)}][Original:]** `{pred_4}`")
        st.progress(conf_4)
        st.markdown(f"**:color[{get_color(atk_pred_4, selected_truth)}][Attacked:]** `{atk_pred_4}`")
        st.progress(atk_conf_4)
        
    st.divider()

    # --- ROW 3: DistilBERT vs. LIME (for Model 2 only) ---
    col5, col6 = st.columns(2)
    with col5:
        st.header("Model 5: DistilBERT 🏆")
        st.markdown(f"**:color[{get_color(pred_5, selected_truth)}][Original:]** `{pred_5}`")
        st.progress(conf_5)
        st.markdown(f"**:color[{get_color(atk_pred_5, selected_truth)}][Attacked:]** `{atk_pred_5}`")
        st.progress(atk_conf_5)

    # --- EDITED: Only show Model 2 LIME ---
    with col6:
        st.header("Model 2 (Classic) Explanation")
        st.markdown("**(LIME)**")
        st.components.v1.html(lime_html_model_2, height=350, scrolling=True)
        # --- REMOVED BERT LIME EXPANDER ---

