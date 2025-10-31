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
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import warnings
import time

# ----------------------------------------------------------------------
# 0. SETUP
# ----------------------------------------------------------------------
print("Starting robustness evaluation...")
warnings.filterwarnings('ignore') # Suppress warnings for cleaner output

# --- Load NLTK data ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# --- MOVED FUNCTION HERE ---
# Define clean_text function BEFORE loading models
def clean_text(text):
    text = text.lower()
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# ----------------------------------------------------------------------
# 1. LOAD ALL 5 MODELS
# ----------------------------------------------------------------------
print("Loading all 5 models...")

# --- Load Class Names (from Model 1) ---
model_1 = joblib.load('model_1_nb.pkl')
CLASS_NAMES = list(model_1.classes_)
NUM_CATEGORIES = len(CLASS_NAMES)

# Model 2: Logistic Regression
model_2 = joblib.load('model_2_lr.pkl')

# Model 3: Word2Vec + LogReg
w2v_model = gensim.models.Word2Vec.load('model_3_w2v.model')
model_3_lr = joblib.load('model_3_lr.pkl')

# Model 4: LSTM
model_4_lstm = tf.keras.models.load_model('model_4_lstm.h5')
model_4_tokenizer = joblib.load('model_4_keras_tokenizer.pkl')
model_4_encoder = joblib.load('model_4_label_encoder.pkl')

# Model 5: DistilBERT
model_dir = 'model_5_distilbert_final'
model_5_tokenizer = AutoTokenizer.from_pretrained(model_dir)
model_5_bert = AutoModelForSequenceClassification.from_pretrained(model_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_5_bert.to(device)
print(f"Models loaded. Running on: {device}")

# ----------------------------------------------------------------------
# 2. DEFINE PREDICTION FUNCTIONS
# ----------------------------------------------------------------------

def predict_model_1(text_list):
    return model_1.predict(text_list)

def predict_model_2(text_list):
    return model_2.predict(text_list)

def vectorize_batch(text_list, model):
    vectors = []
    for text in text_list:
        tokens = clean_text(text).split()
        word_vectors = [model.wv[word] for word in tokens if word in model.wv]
        if len(word_vectors) > 0:
            vectors.append(np.mean(word_vectors, axis=0))
        else:
            vectors.append(np.zeros(model.vector_size))
    return np.array(vectors)

def predict_model_3(text_list):
    vecs = vectorize_batch(text_list, w2v_model)
    return model_3_lr.predict(vecs)

def predict_model_4(text_list):
    cleaned_texts = [clean_text(text) for text in text_list]
    seqs = model_4_tokenizer.texts_to_sequences(cleaned_texts)
    padded_seqs = pad_sequences(seqs, maxlen=150)
    probs = model_4_lstm.predict_on_batch(padded_seqs)
    preds_indices = np.argmax(probs, axis=1)
    return model_4_encoder.inverse_transform(preds_indices)

def predict_model_5(text_list):
    inputs = model_5_tokenizer(text_list, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = model_5_bert(**inputs).logits
    preds_indices = torch.argmax(logits, dim=1).cpu().numpy()
    return [model_5_bert.config.id2label[i] for i in preds_indices]

# ----------------------------------------------------------------------
# 3. LOAD DATA AND RUN EVALUATION
# ----------------------------------------------------------------------

# --- Load the TEST data ---
df = pd.read_csv('complaints_final_clean.csv')
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df['category'])

# --- Set parameters ---
NUM_SAMPLES_PER_CATEGORY = 50 # How many samples to test *per category*
TOTAL_SAMPLES = NUM_SAMPLES_PER_CATEGORY * NUM_CATEGORIES

# --- NEW: Create a BALANCED Test Set ---
print(f"\nCreating a BALANCED test set with {NUM_SAMPLES_PER_CATEGORY} samples per category...")
balanced_test_list = []
for category in CLASS_NAMES:
    category_samples = test_df[test_df['category'] == category].sample(NUM_SAMPLES_PER_CATEGORY, random_state=42)
    balanced_test_list.append(category_samples)

test_sample = pd.concat(balanced_test_list).sample(frac=1, random_state=42) # Concat and shuffle

X_test = test_sample['text'].tolist()
y_test = test_sample['category'].tolist()

print(f"Balanced test set created. Total samples: {len(X_test)}")


# --- Define "Adversarial Synonym" Attack ---
print(f"Initializing 'Adversarial Synonym' attack... Will test on {len(X_test)} samples.")
ADVERSARIAL_SYNONYMS = {
    # Keywords for "Credit Reporting"
    "credit": "standing",
    "report": "file",
    "reporting": "filing",
    "score": "rating",
    "experian": "e-company",
    "equifax": "q-company",
    "transunion": "t-company",
    # Keywords for "Debt collection"
    "debt": "obligation",
    "bill": "invoice",
    "collection": "pickup",
    "collect": "pickup",
    "harassing": "bothering",
    "calls": "contacts",
    # Keywords for "Mortgage"
    "mortgage": "home-loan",
    "foreclosure": "repossession",
    "escrow": "holding",
    "payment": "remittance",
    # Keywords for "Checking or savings account"
    "bank": "institution",
    "account": "profile",
    "checking": "reviewing",
    "savings": "reserves",
    "overdraft": "shortfall",
    "fees": "charges",
}

def attack_with_synonyms(text):
    text_lower = text.lower()
    for key, value in ADVERSARIAL_SYNONYMS.items():
        text_lower = text_lower.replace(key, value)
    return text_lower

# --- Attack the text (This will be FAST) ---
print("Augmenting all test samples... (This should be fast!)")
start_time = time.time()
X_attacked = [attack_with_synonyms(text) for text in X_test]
print(f"Augmentation complete. Time taken: {time.time() - start_time:.2f}s")

# --- Initialize counters for "flips" ---
models_to_run = [
    "Model 1 (Naive Bayes)",
    "Model 2 (LogReg + TF-IDF)",
    "Model 3 (LogReg + W2V)",
    "Model 4 (LSTM)",
    "Model 5 (DistilBERT)"
]
model_flips = {name: 0 for name in models_to_run}
model_correct_predictions = {name: 0 for name in models_to_run}

# --- Run the evaluation loop ---
print("Running predictions...")
start_time = time.time()

BATCH_SIZE = 32 
for i in tqdm(range(0, len(X_test), BATCH_SIZE), desc="Running Predictions"):
    X_batch = X_test[i:i+BATCH_SIZE]
    y_batch = y_test[i:i+BATCH_SIZE]
    X_atk_batch = X_attacked[i:i+BATCH_SIZE]
    
    # Get original and attacked predictions for all models
    preds = {
        "Model 1 (Naive Bayes)": (predict_model_1(X_batch), predict_model_1(X_atk_batch)),
        "Model 2 (LogReg + TF-IDF)": (predict_model_2(X_batch), predict_model_2(X_atk_batch)),
        "Model 3 (LogReg + W2V)": (predict_model_3(X_batch), predict_model_3(X_atk_batch)),
        "Model 4 (LSTM)": (predict_model_4(X_batch), predict_model_4(X_atk_batch)),
        "Model 5 (DistilBERT)": (predict_model_5(X_batch), predict_model_5(X_atk_batch)),
    }
    
    for j in range(len(y_batch)): # Iterate through each item in the batch
        true_label = y_batch[j]
        
        for model_name in models_to_run:
            original_pred = preds[model_name][0][j]
            attacked_pred = preds[model_name][1][j]
            
            was_correct = (original_pred == true_label)
            is_now_wrong = (attacked_pred != true_label)
            
            if was_correct:
                model_correct_predictions[model_name] += 1
                if is_now_wrong:
                    model_flips[model_name] += 1

print(f"Prediction loop complete. Time taken: {time.time() - start_time:.2f}s")

# ----------------------------------------------------------------------
# 4. PRINT FINAL RESULTS (WITH NEW METRIC)
# ----------------------------------------------------------------------
print("\n--- 🔬 ROBUSTNESS EVALUATION RESULTS ---")
print(f"Based on a BALANCED test set of {TOTAL_SAMPLES} samples ({NUM_SAMPLES_PER_CATEGORY} per category).")
print("Attack: 'Adversarial Synonym' (e.g., 'credit report' -> 'standing file')\n")
print("True Flip Rate = % of *originally correct* predictions that were fooled by the attack.\n")

print("Model\t\t\t\t| True Flip Rate")
print("-----------------------------------------------------------------")

for model_name in models_to_run:
    flip_count = model_flips[model_name]
    correct_count = model_correct_predictions[model_name]
    
    if correct_count > 0:
        true_flip_rate = (flip_count / correct_count) * 100
    else:
        true_flip_rate = 0.0 # Avoid divide-by-zero
        
    print(f"{model_name:<30}\t| {true_flip_rate:.2f}% ({flip_count} of {correct_count} flipped)")

print("\n--- Analysis ---")
print("A *lower* True Flip Rate is better (more robust).")
print("This chart is the key 'tie-breaker' for your project!")

