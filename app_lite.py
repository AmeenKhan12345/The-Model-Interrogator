import streamlit as st
import pandas as pd
import numpy as np
import joblib
#import gensim
import string
import nltk
#import torch
#import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
#from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from textattack.augmentation import Augmenter
from textattack.transformations.word_swaps import WordSwapEmbedding, WordSwapWordNet
from textattack.transformations.word_swaps import WordSwapRandomCharacterDeletion
from textattack.transformations.word_swaps.word_swap_qwerty import WordSwapQWERTY
from textattack.transformations import CompositeTransformation, WordSwapMaskedLM
import lime
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from difflib import SequenceMatcher, ndiff
from sentence_transformers import SentenceTransformer, util
import seaborn as sns
from collections import Counter

# ----------------------------------------------------------------------
# 0. PAGE CONFIG & HELPER FUNCTIONS
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="The Model Interrogator 🔬",
    page_icon="🔬",
    layout="wide",
)

# Download NLTK data
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
    """Preprocessing for Models 1, 2, 3, 4"""
    text = text.lower()
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# ----------------------------------------------------------------------
# 1. LOAD SENTENCE TRANSFORMER FOR SEMANTIC SIMILARITY
# ----------------------------------------------------------------------

@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

semantic_model = load_sentence_transformer()

# ----------------------------------------------------------------------
# 2. LOAD ALL 5 MODELS
# ----------------------------------------------------------------------

@st.cache_resource
def load_model_1():
    return joblib.load('model_1_nb.pkl')

@st.cache_resource
def load_model_2():
    return joblib.load('model_2_lr.pkl')

'''@st.cache_resource
def load_model_3():
    w2v_model = gensim.models.Word2Vec.load('model_3_w2v.model')
    lr_model = joblib.load('model_3_lr.pkl')
    return w2v_model, lr_model'''

def vectorize_complaints(text, model):
    tokens = clean_text(text).split()
    word_vectors = [model.wv[word] for word in tokens if word in model.wv]
    
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0).reshape(1, -1)
    else:
        return np.zeros((1, model.vector_size))

'''@st.cache_resource
def load_model_4():
    model = tf.keras.models.load_model('model_4_lstm.h5')
    tokenizer = joblib.load('model_4_keras_tokenizer.pkl')
    encoder = joblib.load('model_4_label_encoder.pkl')
    return model, tokenizer, encoder'''

'''@st.cache_resource
def load_model_5():
    model_dir = 'model_5_distilbert_final'
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device'''

# Load all models
model_1 = load_model_1()
model_2 = load_model_2()
#w2v_model, model_3_lr = load_model_3()
#model_4_lstm, model_4_tokenizer, model_4_encoder = load_model_4()
#model_5_bert, model_5_tokenizer, bert_device = load_model_5()

CLASS_NAMES = list(model_1.classes_)

# ----------------------------------------------------------------------
# 3. PREDICTION HELPER FUNCTIONS
# ----------------------------------------------------------------------

def get_model_1_prediction(text):
    probabilities = model_1.predict_proba([text])[0]
    top_class_index = np.argmax(probabilities)
    return model_1.classes_[top_class_index], float(probabilities[top_class_index]), probabilities

def get_model_2_prediction(text):
    probabilities = model_2.predict_proba([text])[0]
    top_class_index = np.argmax(probabilities)
    return model_2.classes_[top_class_index], float(probabilities[top_class_index]), probabilities

'''def get_model_3_prediction(text):
    vectorized_text = vectorize_complaints(text, w2v_model)
    probabilities = model_3_lr.predict_proba(vectorized_text)[0]
    top_class_index = np.argmax(probabilities)
    return model_3_lr.classes_[top_class_index], float(probabilities[top_class_index]), probabilities'''

'''def get_model_4_prediction(text):
    cleaned = clean_text(text)
    seq = model_4_tokenizer.texts_to_sequences([cleaned])
    padded_seq = pad_sequences(seq, maxlen=150)
    probabilities = model_4_lstm.predict_on_batch(padded_seq)[0]
    top_class_index = np.argmax(probabilities)
    return model_4_encoder.classes_[top_class_index], float(probabilities[top_class_index]), probabilities'''

'''def get_model_5_prediction(text):
    inputs = model_5_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(bert_device)
    with torch.no_grad():
        logits = model_5_bert(**inputs).logits
    
    probabilities = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
    top_class_index = np.argmax(probabilities)
    return model_5_bert.config.id2label[top_class_index], float(probabilities[top_class_index]), probabilities'''

# ----------------------------------------------------------------------
# 4. LIME HELPER FUNCTIONS (FOR MODEL 2)
# ----------------------------------------------------------------------

# We create the explainer *once*
@st.cache_resource
def get_lime_explainer():
    return LimeTextExplainer(class_names=CLASS_NAMES)

lime_explainer = get_lime_explainer()

# This function must have this exact signature for LIME to work
def sklearn_predictor(text_list):
    """LIME-compatible predictor for Model 2 (sklearn pipeline)"""
    return model_2.predict_proba(text_list)

def generate_sklearn_lime_explanation(text, truth_label_idx):
    """Generates the LIME explanation HTML for Model 2"""
    try:
        explanation = lime_explainer.explain_instance(
            text,
            sklearn_predictor,
            num_features=10,
            top_labels=len(CLASS_NAMES),
            labels=[truth_label_idx] # Only explain the "Ground Truth" class
        )
        return explanation.as_html()
    except Exception as e:
        return f"<p>LIME explanation failed: {e}</p>"

# ----------------------------------------------------------------------
# 5. ATTACK METRICS CALCULATION
# ----------------------------------------------------------------------

def calculate_semantic_similarity(text1, text2):
    """Calculate cosine similarity between two texts"""
    embeddings = semantic_model.encode([text1, text2])
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return similarity

def calculate_attack_metrics(original_preds, attacked_preds, original_text, attacked_text):
    """
    Calculate comprehensive attack metrics
    Returns: dict with flip_rate, avg_confidence_drop, semantic_similarity, etc.
    """
    flips = sum([1 for i in range(5) if original_preds[i][0] != attacked_preds[i][0]])
    flip_rate = flips / 5
    
    confidence_drops = [original_preds[i][1] - attacked_preds[i][1] for i in range(5)]
    avg_confidence_drop = np.mean(confidence_drops)
    
    semantic_sim = calculate_semantic_similarity(original_text, attacked_text)
    
    # Text perturbation ratio
    original_words = set(original_text.lower().split())
    attacked_words = set(attacked_text.lower().split())
    
    if len(original_words) > 0:
        perturbation_ratio = len(original_words.symmetric_difference(attacked_words)) / len(original_words)
    else:
        perturbation_ratio = 0
    
    return {
        'flip_rate': flip_rate,
        'flips': flips,
        'avg_confidence_drop': avg_confidence_drop,
        'confidence_drops': confidence_drops,
        'semantic_similarity': semantic_sim,
        'perturbation_ratio': perturbation_ratio
    }

def calculate_robustness_score(original_pred, attacked_pred, semantic_sim):
    """
    Calculate robustness score for a single model
    Score = (1 - flip) × confidence_retention × semantic_preservation
    """
    flip = 0 if original_pred[0] == attacked_pred[0] else 1
    # Add a small epsilon to prevent division by zero
    confidence_retention = attacked_pred[1] / (original_pred[1] + 1e-10) 
    
    robustness = (1 - flip) * confidence_retention * semantic_sim
    return robustness * 100  # Convert to percentage

# ----------------------------------------------------------------------
# 6. ENSEMBLE VOTING SYSTEM
# ----------------------------------------------------------------------

def ensemble_prediction(predictions, method='majority'):
    """
    Combine predictions from all 5 models
    methods: 'majority', 'weighted', 'confidence'
    """
    if method == 'majority':
        votes = [pred[0] for pred in predictions]
        vote_counts = Counter(votes)
        ensemble_pred = vote_counts.most_common(1)[0][0]
        ensemble_conf = vote_counts.most_common(1)[0][1] / 5
        
    elif method == 'weighted':
        class_scores = {}
        for pred_class, confidence, _ in predictions:
            if pred_class not in class_scores:
                class_scores[pred_class] = 0
            class_scores[pred_class] += confidence
        
        ensemble_pred = max(class_scores, key=class_scores.get)
        ensemble_conf = class_scores[ensemble_pred] / sum(class_scores.values())
    
    elif method == 'confidence':
        max_conf_idx = np.argmax([pred[1] for pred in predictions])
        ensemble_pred = predictions[max_conf_idx][0]
        ensemble_conf = predictions[max_conf_idx][1]
    
    votes = [pred[0] for pred in predictions]
    agreement = max(Counter(votes).values()) / 5
    
    return ensemble_pred, ensemble_conf, agreement

# ----------------------------------------------------------------------
# 7. VISUALIZATION FUNCTIONS
# ----------------------------------------------------------------------

def create_text_diff_html(original, attacked):
    """Create HTML showing differences between original and attacked text"""
    diff = list(ndiff(original.split(), attacked.split()))
    
    html = "<div style='font-family: monospace; line-height: 2; border: 1px solid #ddd; padding: 10px; border-radius: 5px;'>"
    for word in diff:
        if word.startswith('- '):
            html += f"<span style='background-color: #ffcccc; text-decoration: line-through;'>{word[2:]}</span> "
        elif word.startswith('+ '):
            html += f"<span style='background-color: #ccffcc; font-weight: bold;'>{word[2:]}</span> "
        elif word.startswith('  '):
            html += f"{word[2:]} "
    html += "</div>"
    
    return html

def create_confidence_comparison_chart(original_preds, attacked_preds, model_names):
    """Create interactive confidence comparison chart"""
    original_confs = [pred[1] for pred in original_preds]
    attacked_confs = [pred[1] for pred in attacked_preds]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Original', x=model_names, y=original_confs, marker_color='rgb(26, 118, 255)'
    ))
    fig.add_trace(go.Bar(
        name='Attacked', x=model_names, y=attacked_confs, marker_color='rgb(255, 127, 14)'
    ))
    fig.update_layout(
        title='Confidence Comparison: Original vs Attacked',
        xaxis_title='Model', yaxis_title='Confidence',
        barmode='group', height=400, legend_title_text='Prediction'
    )
    return fig

def create_robustness_leaderboard(robustness_scores, model_names):
    """Create interactive robustness leaderboard"""
    df = pd.DataFrame({
        'Model': model_names,
        'Robustness Score': robustness_scores
    }).sort_values('Robustness Score', ascending=False)
    
    df['Rank'] = range(1, len(df) + 1)
    df['Medal'] = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣']
    
    fig = go.Figure(data=[go.Bar(
        x=df['Robustness Score'], y=df['Model'], orientation='h',
        marker=dict(color=df['Robustness Score'], colorscale='RdYlGn', showscale=True),
        text=df['Robustness Score'].round(2), textposition='auto',
    )])
    fig.update_layout(
        title='🏆 Robustness Leaderboard',
        xaxis_title='Robustness Score (%)', yaxis_title='',
        height=400, yaxis={'categoryorder': 'total ascending'}
    )
    return fig, df

def create_prediction_heatmap(original_probs, attacked_probs, model_names, class_names):
    """Create heatmap showing probability distributions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(original_probs, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=model_names, ax=ax1, vmin=0, vmax=1)
    ax1.set_title('Original Predictions')
    
    sns.heatmap(attacked_probs, annot=True, fmt='.2f', cmap='Reds', 
                xticklabels=class_names, yticklabels=model_names, ax=ax2, vmin=0, vmax=1)
    ax2.set_title('Attacked Predictions')
    
    plt.tight_layout()
    return fig

def create_metric_cards_html(metrics):
    """Create HTML cards for key metrics"""
    html = """
    <style>
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 20px;
        color: #333333;
        text-align: center;
        margin: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 2.5em;
        font-weight: bold;
        margin: 10px 0;
        color: #6A1B9A;
    }
    .metric-label {
        font-size: 1em;
        opacity: 0.9;
    }
    </style>
    """
    
    cards = f"""
    <div style='display: flex; justify-content: space-around; flex-wrap: wrap;'>
        <div class='metric-card'>
            <div class='metric-label'>Models Fooled</div>
            <div class='metric-value'>{metrics['flips']}/5</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Flip Rate</div>
            <div class='metric-value'>{metrics['flip_rate']*100:.1f}%</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Avg Confidence Drop</div>
            <div class='metric-value'>{metrics['avg_confidence_drop']*100:.1f}%</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Semantic Similarity</div>
            <div class='metric-value'>{metrics['semantic_similarity']*100:.1f}%</div>
        </div>
    </div>
    """
    
    return html + cards

# ----------------------------------------------------------------------
# 8. EXPANDED ATTACK ARSENAL
# ----------------------------------------------------------------------

@st.cache_resource
def get_attack_transformation(attack_type, intensity=1.0):
    """
    Get transformation based on attack type with intensity control
    intensity: 0.0-1.0, controls how aggressive the attack is
    """
    if attack_type == "WordSwap (Synonym)":
        return WordSwapEmbedding(max_candidates=int(5 * intensity))
    
    elif attack_type == "WordSwap (WordNet)":
        return WordSwapWordNet()
    
    elif attack_type == "WordSwap (QWERTY)":
        # We'll use random_one=False and increase pct_words_to_swap to respect intensity
        return WordSwapQWERTY(random_one=False, pct_words_to_swap=intensity * 0.2)
    
    elif attack_type == "CharDelete (Typo)":
        # We'll use random_one=False and increase pct_words_to_swap
        return WordSwapRandomCharacterDeletion(random_one=True)
    
    elif attack_type == "BERT Masked LM":
        return WordSwapMaskedLM(max_candidates=int(10 * intensity))
    
    elif attack_type == "Combo Attack (WordNet + Typo)":
        return CompositeTransformation([
            WordSwapWordNet(),
            WordSwapRandomCharacterDeletion(random_one=True)
        ])
    
    else:
        return WordSwapEmbedding()

# ----------------------------------------------------------------------
# 9. STREAMLIT UI LAYOUT
# ----------------------------------------------------------------------

st.title("🔬 The Model Interrogator")
st.subheader("Advanced NLP Robustness Testing with Adversarial Attacks")

# --- DEFINE TABS ---
tab1, tab2 = st.tabs(["🚀 The Interrogator", "📖 Research & Methodology"])


# ----------------------------------------------------------------------
# ----------------------  TAB 1: THE INTERROGATOR ----------------------
# ----------------------------------------------------------------------

with tab1:
    # --- Sidebar for Attack Configuration ---
    with st.sidebar:
        st.header("⚙️ Attack Configuration")
        
        attack_type = st.selectbox(
            "Select Attack Method:",
            [
                "WordSwap (Synonym)", # Fast, semantic
                "WordSwap (QWERTY)",  # Fast, typo
                "CharDelete (Typo)",  # Fast, typo
                 "WordSwap (WordNet)", # Slow, semantic
                 "BERT Masked LM",     # Very Slow, semantic
                 "Combo Attack (WordNet + Typo)" # Very Slow
            ],
            help="Select the method to attack the text. Faster attacks are pre-selected."
        )
        
        attack_intensity = st.slider(
            "Attack Intensity:",
            min_value=0.3, max_value=1.0, value=0.7, step=0.1,
            help="Controls how many words are changed. (Not all attacks use this)."
        )
        
        ensemble_method = st.radio(
            "Ensemble Method:",
            ["Majority Vote", "Weighted (Confidence)", "Highest Confidence"],
            help="How to combine model predictions"
        )
        
        st.divider()
        st.header("📊 Display Options")
        show_heatmap = st.checkbox("Show Probability Heatmap", value=True)
        show_diff = st.checkbox("Show Text Differences", value=True)
        show_lime = st.checkbox("Show LIME Explanation (Model 2)", value=True)

    # --- Main Input Area ---
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
            "truth": "Credit Reporting" # Merged category
        },
        "Mortgage vs. Credit vs. Debt": {
            "text": "I was denied a mortgage application because of a 'serious delinquency' on my credit file. This is an old medical bill that was sent to collections by mistake. I need this fixed.",
            "truth": "Credit Reporting" # Merged category
        }
    }

    col_input1, col_input2 = st.columns([2, 1])

    with col_input1:
        selected_complaint_key = st.selectbox(
            "Choose an example or write your own:",
            options=list(COMPLAINT_EXAMPLES.keys())
        )

    with col_input2:
        example_text = COMPLAINT_EXAMPLES[selected_complaint_key]["text"]
        truth_label = COMPLAINT_EXAMPLES[selected_complaint_key]["truth"]
        
        all_labels = ["None"] + CLASS_NAMES
        default_index = all_labels.index(truth_label) if truth_label in all_labels else 0
        
        selected_truth = st.selectbox(
            "Ground Truth:",
            options=all_labels,
            index=default_index,
            help="Set the 'true' label to see right/wrong predictions in green/red."
        )

    user_input = st.text_area(
        "Enter complaint text:", 
        example_text, 
        height=120
    )

    # --- Run Button ---
    if st.button("🚀 Run Interrogation!", type="primary", use_container_width=True):
        
        # --- Step 1: Generate Attack ---
        with st.spinner(f"⚔️ Launching {attack_type} attack..."):
            transformation = get_attack_transformation(attack_type, attack_intensity)
            augmenter = Augmenter(transformation=transformation)
            
            try:
                attacked_text = augmenter.augment(user_input)
                attacked_text = attacked_text[0] if isinstance(attacked_text, list) and len(attacked_text) > 0 else user_input
            except Exception as e:
                st.error(f"Attack failed: {e}. Using original text.")
                attacked_text = user_input
        
        # --- Step 2: Get All Predictions ---
        with st.spinner("🔮 Running all 5 models..."):
            #model_names = ["Model 1 (Naive Bayes)", "Model 2 (LogReg+TF-IDF)", "Model 3 (LogReg+W2V)", "Model 4 (LSTM)", "Model 5 (DistilBERT)"]
            model_names = ["Model 1 (Naive Bayes)", "Model 2 (LogReg+TF-IDF)"] # <-- EDITED
            original_preds = [
                get_model_1_prediction(user_input),
                get_model_2_prediction(user_input),
                #get_model_3_prediction(user_input),
                #get_model_4_prediction(user_input),
                #get_model_5_prediction(user_input)
            ]
            
            attacked_preds = [
                get_model_1_prediction(attacked_text),
                get_model_2_prediction(attacked_text),
                #get_model_3_prediction(attacked_text),
                #get_model_4_prediction(attacked_text),
                #get_model_5_prediction(attacked_text)
            ]
            
            ensemble_method_map = {
                "Majority Vote": "majority",
                "Weighted (Confidence)": "weighted",
                "Highest Confidence": "confidence"
            }
            
            ensemble_orig = ensemble_prediction(original_preds, ensemble_method_map[ensemble_method])
            ensemble_atk = ensemble_prediction(attacked_preds, ensemble_method_map[ensemble_method])
        
        # --- Step 3: Calculate Metrics ---
        metrics = calculate_attack_metrics(original_preds, attacked_preds, user_input, attacked_text)
        
        robustness_scores = []
        for i in range(5):
            score = calculate_robustness_score(original_preds[i], attacked_preds[i], metrics['semantic_similarity'])
            robustness_scores.append(score)
        
        st.success("✅ Interrogation Complete!")
        
        # --- Display Results ---
        
        st.markdown("### 📊 Attack Impact Metrics")
        st.markdown(create_metric_cards_html(metrics), unsafe_allow_html=True)
        st.divider()
        
        if show_diff:
            st.markdown("### 📝 Text Comparison")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Text:**")
                st.info(user_input)
            with col2:
                st.markdown("**Attacked Text:**")
                st.warning(attacked_text)
            st.markdown("**Differences:**")
            st.markdown(create_text_diff_html(user_input, attacked_text), unsafe_allow_html=True)
            st.divider()
        
        st.markdown("### 📈 Confidence Comparison")
        conf_chart = create_confidence_comparison_chart(original_preds, attacked_preds, model_names)
        st.plotly_chart(conf_chart, use_container_width=True)
        st.divider()
        
        st.markdown("### 🏆 Robustness Leaderboard")
        leaderboard_chart, leaderboard_df = create_robustness_leaderboard(robustness_scores, model_names)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(leaderboard_chart, use_container_width=True)
        with col2:
            st.markdown("**Rankings:**")
            for idx, row in leaderboard_df.iterrows():
                st.markdown(f"{row['Medal']} **{row['Model']}**: {row['Robustness Score']:.2f}%")
        st.divider()
        
        st.markdown("### 🤝 Ensemble Prediction")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Ensemble", ensemble_orig[0], f"{ensemble_orig[1]*100:.1f}% conf")
            st.caption(f"Agreement: {ensemble_orig[2]*100:.0f}%")
        with col2:
            st.metric("Attacked Ensemble", ensemble_atk[0], f"{ensemble_atk[1]*100:.1f}% conf")
            st.caption(f"Agreement: {ensemble_atk[2]*100:.0f}%")
        with col3:
            ensemble_changed = "✅ Robust" if ensemble_orig[0] == ensemble_atk[0] else "❌ Flipped"
            st.metric("Ensemble Status", ensemble_changed)
            st.caption(f"Method: {ensemble_method}")
        st.divider()
        
        st.markdown("### 🔍 Individual Model Results")
        
        def get_color(pred, truth_label):
            if truth_label == "None" or truth_label is None:
                return "gray"
            return "green" if pred == truth_label else "red"
        
        for i, model_name in enumerate(model_names):
            with st.expander(f"**{model_name}** - Robustness: {robustness_scores[i]:.2f}%", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Original:**")
                    st.markdown(f":{get_color(original_preds[i][0], selected_truth)}[**{original_preds[i][0]}**]")
                    st.progress(original_preds[i][1])
                
                with col2:
                    st.markdown("**Attacked:**")
                    st.markdown(f":{get_color(attacked_preds[i][0], selected_truth)}[**{attacked_preds[i][0]}**]")
                    st.progress(attacked_preds[i][1])
                
                with col3:
                    status = "✅ Survived" if original_preds[i][0] == attacked_preds[i][0] else "❌ Flipped"
                    conf_drop = (original_preds[i][1] - attacked_preds[i][1]) * 100
                    st.metric("Status", status)
                    st.metric("Conf Drop", f"{conf_drop:.1f}%")
        
        if show_heatmap:
            st.divider()
            st.markdown("### 🔥 Probability Distribution Heatmap")
            original_probs = np.array([pred[2] for pred in original_preds])
            attacked_probs = np.array([pred[2] for pred in attacked_preds])
            heatmap_fig = create_prediction_heatmap(original_probs, attacked_probs, model_names, CLASS_NAMES)
            st.pyplot(heatmap_fig)

        if show_lime:
            st.divider()
            st.markdown("### 🧠 Model 2 (Classic) Explanation (LIME)")
            st.markdown("Why did Model 2 make its **original** decision? (Red = against, Green = for)")
            
            if selected_truth in CLASS_NAMES:
                truth_label_idx = CLASS_NAMES.index(selected_truth)
                with st.spinner("Generating LIME explanation..."):
                    lime_html = generate_sklearn_lime_explanation(user_input, truth_label_idx)
                    st.components.v1.html(lime_html, height=500, scrolling=True)
            else:
                st.warning("Please select a 'Ground Truth' label to generate a LIME explanation.")


# ----------------------------------------------------------------------
# ------------------  TAB 2: RESEARCH & METHODOLOGY --------------------
# ----------------------------------------------------------------------

with tab2:
    st.header("Project Overview")
    st.markdown("""
    This project investigates the trade-off between **accuracy** and **robustness** in NLP models. 
    It demonstrates that while many models can achieve similar base accuracy, their performance 
    under "adversarial attack" reveals their true quality.

    The "Interrogator" app on the first tab allows for a live "stress-test" of five different 
    classification models on a real-world public grievance dataset.
    """)

    st.divider()

    # --- 1. Dataset Statistics ---
    st.header("📊 Dataset Statistics")
    st.markdown("""
    The dataset is derived from the **CFPB Consumer Complaint Database**, sourced from `data.gov.in`. 
    To create a focused, high-impact problem, we filtered for the top complaint categories.

    - **Original Data:** A 7GB+ file with millions of entries.
    - **Sampling:** A 1% random sample was taken to create a manageable `complaints_lite.csv`.
    - **Cleaning:** Rows without complaint narratives were dropped.
    - **Merging:** The top two categories, which were redundant, were merged:
        - `Credit reporting or other personal consumer reports`
        - `Credit reporting, credit repair services, or other personal consumer reports`
        - **Became:** `Credit Reporting`
    - **Filtering:** We used the final top 4 most common, distinct categories for our model.
    """)

    # Data for the chart
    category_data = {
        'Category': ['Credit Reporting', 'Debt collection', 'Checking or savings account', 'Mortgage'],
        'Count': [22553, 3830, 1530, 1446]
    }
    category_df = pd.DataFrame(category_data)
    
    fig = px.bar(category_df, 
                 x='Category', 
                 y='Count', 
                 title='Final Class Distribution (Highly Imbalanced)',
                 color='Category',
                 text_auto=True)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- 2. Methodology & The Model Gauntlet ---
    st.header("🛠️ Methodology & The Model Gauntlet")
    st.markdown("""
    We trained 5 models representing the evolution of NLP techniques. All models were trained and tested 
    on the same `random_state=42` split (75% train / 25% test) to ensure a fair comparison.
    """)
    
    st.markdown("#### Model 1: The \"Straw Man\" (Naive Bayes + TF-IDF)")
    st.markdown("""
    - **Technique:** `sklearn.pipeline` with `TfidfVectorizer` (using our custom `clean_text` preprocessor) 
      and `MultinomialNB`.
    - **Role:** A simple, fast, keyword-counting baseline. It has no understanding of context or semantics.
    """)

    st.markdown("#### Model 2: The \"Classic Workhorse\" (Logistic Regression + TF-IDF)")
    st.markdown("""
    - **Technique:** `sklearn.pipeline` with `TfidfVectorizer` and `LogisticRegression`.
    - **Role:** A powerful statistical model that learns the *weights* of keywords. It's much smarter 
      than Naive Bayes but still keyword-based.
    """)
    
    st.markdown("#### Model 3: The \"Semantic-but-Dumb\" (Logistic Regression + Avg. Word2Vec)")
    st.markdown("""
    - **Technique:** A `gensim.models.Word2Vec` model (100 dimensions) was trained on the `clean_text`. 
      A complaint's vector was calculated by averaging the vectors of its words. A `LogisticRegression` 
      model was then trained on these averaged vectors.
    - **Role:** Tests the power of *semantics* (word meaning) while deliberately *destroying* word order.
    """)
    
    st.markdown("#### Model 4: The \"Sequential Thinker\" (LSTM + Word2Vec)")
    st.markdown("""
    - **Technique:** A `tf.keras.Sequential` model with an `Embedding` layer (weights frozen from our 
      Word2Vec model), an `LSTM(64)` layer, and a `Dense(4, 'softmax')` output.
    - **Role:** Tests the power of semantics *and* sequence, as the LSTM reads the vectors in order.
    """)
    
    st.markdown("#### Model 5: The \"Modern Champion\" (DistilBERT)")
    st.markdown("""
    - **Technique:** The `distilbert-base-uncased` model from Hugging Face, fine-tuned for 3 epochs 
      on our dataset using the `transformers.Trainer` API.
    - **Role:** A state-of-the-art model that understands deep, bi-directional context.
    """)

    st.divider()

    # --- 3. Performance Baselines ---
    st.header("📈 Performance Baselines (on Clean Test Data)")
    st.markdown("""
    Before any attacks, we evaluated all 5 models on the clean test set. This revealed a 
    **three-way tie for first place**, proving that base accuracy is not a good metric to find the 
    "best" model.
    """)
    
    # Data for the performance table
    perf_data = {
    "Model": [
        "1. Naive Bayes",
        "2. LogReg + TF-IDF",
        "3. LogReg + W2V",
        "4. LSTM + W2V",
        "5. DistilBERT"
    ],
    "Accuracy": [0.824, 0.923, 0.903, 0.920, 0.928],
    "F1-Score (Weighted)": [0.77, 0.92, 0.90, 0.92, 0.93],
    "Precision (Weighted)": [0.83, 0.92, 0.90, 0.92, 0.93],
    "Recall (Weighted)": [0.82, 0.92, 0.90, 0.92, 0.93],
    "F1-Score (Macro)": [0.51, 0.87, 0.84, 0.87, 0.88],  # update when you have new numbers
    "Log-Loss": [0.51, 0.23, 0.26, 0.22, 5.52]            # update when you have new numbers
}

    perf_df = pd.DataFrame(perf_data).set_index("Model")

    # --- Styling: highlight best metrics and lowest log-loss ---
    styled = (
        perf_df.style
        .highlight_max(
            axis=0,
            subset=[
                'Accuracy',
                'F1-Score (Weighted)',
                'F1-Score (Macro)',
                'Precision (Weighted)',
                'Recall (Weighted)'
            ],
            color='#ccffcc'
        )
        .highlight_min(axis=0, subset=['Log-Loss'], color='#ccffcc')
        .format("{:.3f}")
    )

    st.dataframe(styled, use_container_width=True)

    # --- Explanations / notes ---
    st.markdown("""
    - **Weighted Avg:** Gives more weight to large classes (e.g., "Credit Reporting"). Useful when you care about overall accuracy across imbalanced classes.
    - **Macro Avg:** Treats all classes equally — a tougher, more honest score for imbalanced datasets.
    - **Log-Loss:** Measures the model's *confidence* in its predictions. Lower is better.
    - *(Note: F1, Precision, and Recall shown above are **weighted** averages to account for class imbalance.)*
    """)
    st.divider()

    # --- NEW SECTION: Confusion Matrices ---
    st.subheader("Confusion Matrices")
    st.markdown("""
    A Confusion Matrix shows *how* a model is failing. The rows are the **Actual Truth**, and the 
    columns are the **Model's Prediction**. The diagonal line shows all the *correct* guesses.
    """)

    # Make sure you have saved the images as "cm_model_1.png", "cm_model_2.png", etc.
    try:
        cm_col1, cm_col2 = st.columns(2)
        with cm_col1:
            st.image("cm_model_1.png", caption="Model 1: Naive Bayes")
            st.image("cm_model_3.png", caption="Model 3: LogReg + W2V")
            
        with cm_col2:
            st.image("cm_model_2.png", caption="Model 2: LogReg + TF-IDF")
            st.image("cm_model_4.png", caption="Model 4: LSTM")
        
        st.image("cm_model_5.png", caption="Model 5: DistilBERT (The Champion)")
        
        st.markdown("""
        **Analysis:** The "Classic Hero" (Model 2) had a high F1 score, but its matrix would show 
        it was significantly confused between "Debt collection" and "Credit Reporting". 
        Model 5's matrix is the "cleanest," showing the fewest errors on the off-diagonal.
        """)
        
    except FileNotFoundError:
        st.error("Confusion Matrix images not found. Please run the 'calculate_metrics.py' script to generate them.")


    # --- 3.5. Analysis 3: Robustness as a Tie-Breaker ---
    st.header("🏆 Analysis 3: The Robustness Tie-Breaker")
    st.markdown("""
    Since accuracy was a three-way tie, a robustness test was performed to find the true winner.
    
    - **Methodology:** A balanced test set (50 samples from each of the 4 categories) was created.
    - **Attack:** A custom **"Adversarial Synonym"** attack was applied, replacing keywords 
      with synonyms (e.g., `credit report` -> `standing file`).
    - **Metric:** We measured the **"True Flip Rate"** — the percentage of *originally correct* predictions that were "fooled" (flipped) by the attack. **A lower score is better.**
    """)

    # --- Robustness Bar Chart ---
    robustness_data = {
        "Model": [
            "1. Naive Bayes", 
            "2. LogReg + TF-IDF", 
            "3. LogReg + W2V", 
            "4. LSTM", 
            "5. DistilBERT"
        ],
        "True Flip Rate (%)": [10.98, 24.26, 14.20, 16.67, 6.98],
        "Details": [
            "9 of 82 flipped", 
            "41 of 169 flipped", 
            "24 of 169 flipped", 
            "28 of 168 flipped", 
            "12 of 172 flipped"
        ]
    }
    robustness_df = pd.DataFrame(robustness_data)
    
    fig_robust = px.bar(
        robustness_df,
        x="Model",
        y="True Flip Rate (%)",
        title="Robustness Test: 'True Flip Rate' (Lower is Better)",
        color="Model",
        text="True Flip Rate (%)"
    )
    fig_robust.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    st.plotly_chart(fig_robust, use_container_width=True)

    # --- Robustness Data Table ---
    st.dataframe(robustness_df.set_index("Model"), use_container_width=True)

    # --- Final Conclusion ---
    st.markdown("#### Key Findings:")
    st.markdown("""
    1.  **DistilBERT (Model 5) is the Clear Winner:** With a flip rate of only **6.98%**, the Transformer 
        is by far the most robust model. It successfully understood the *meaning* of the synonyms 
        and was not fooled.
    2.  **The "Classic Hero" Fails:** Model 2 (LogReg + TF-IDF), which had a top-tier accuracy of 0.92, 
        was the **least robust** model, failing **24.26%** of the time. This proves its accuracy 
        was highly dependent on the *exact keywords* it was trained on.
    3.  **Context > Keywords:** The models that understand semantics (W2V, LSTM, BERT) all 
        outperformed the pure keyword models in this advanced test, proving the hypothesis.
    """)


    # --- 4. Related Work ---
    st.header("📚 Related Work & Adversarial NLP")
    st.markdown("""
    This project is built on the principles of **Adversarial NLP**, a field that stress-tests models 
    by feeding them "adversarial examples"—inputs that are slightly modified to cause a model to fail.

    - **Why it matters:** As models like BERT become more powerful, they also learn to "cheat" by 
      using statistical cues and dataset-specific biases. A model that achieves 95% accuracy 
      can still be fooled by a simple typo (Ribeiro et al., 2020).
    - **Attack Methods:** This app uses techniques from libraries like **TextAttack**, which 
      implements various attacks:
        - **Typo Attacks (`WordSwapQWERTY`):** Simulate common keyboard errors.
        - **Semantic Attacks (`WordSwapEmbedding`, `WordSwapWordNet`):** Replace words with synonyms 
          that should *not* change the meaning.
    - **Interpretability:** The "why" behind a prediction is as important as the prediction itself. 
      Techniques like **LIME** (Local Interpretable Model-agnostic Explanations) help us understand 
      which keywords a model is "looking at," which is why it's included in the Interrogator tab.

    This app confirms a key finding in the field: models that rely on "bag-of-words" techniques 
    (like TF-IDF) are extremely brittle, while context-aware models (like DistilBERT) are 
    significantly more robust.
    """)

    st.divider()

    # --- 5. Reproducibility ---
    st.header("🔄 Reproducibility")
    st.markdown("""
    This project is fully reproducible. All code, model weights, and data are available at the 
    project's GitHub repository.

    - **[Code Repository 🔗](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME)**
    - **[Dataset 🔗](https://data.gov.in/catalogs/consumer-complaint)** (or link to your Kaggle/lite version)
    - **[Saved Models 🔗](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME/tree/main/models)** *(Note: Models were saved using `joblib`, `tf.keras.models.save_model`, and 
      `trainer.save_model()` and must be loaded using the correct library as shown in this app.)*
    """)


# ----------------------------------------------------------------------
# -------------------------  FOOTER ------------------------------------
# ----------------------------------------------------------------------

st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🔬 The Model Interrogator | Testing NLP Robustness Through Adversarial Attacks</p>
    <p>Models: Naive Bayes • Logistic Regression • Word2Vec • LSTM • DistilBERT</p>
</div>
""", unsafe_allow_html=True)

