import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import os



def identity_tokenizer(x):
    return x

def identity_preprocessor(x):
    return x

st.set_page_config(
    page_title="Détecteur de Spam",
    page_icon="📧",
    layout="centered"
)

@st.cache_resource
def download_nltk_data():
    """Télécharge les données NLTK nécessaires"""
    nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)
    
    try:
        nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
        nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)
        nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
    except:
        pass

download_nltk_data()

@st.cache_resource
def load_model():
    """Charge le modèle SVM et le vectorizer TF-IDF"""
    try:
        model = joblib.load('svm_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError as e:
        st.error(f"⚠️ Erreur : Fichier introuvable - {e}")
        st.info("💡 Assurez-vous que 'svm_model.pkl' et 'tfidf_vectorizer.pkl' sont dans le même dossier que app.py")
        st.stop()

model, vectorizer = load_model()

def preprocess_text(text):
    """
    Applique EXACTEMENT le même prétraitement que votre code d'entraînement
    """
    
    text = text.lower()
    
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t.lower() not in stop_words]
    
    tokens = [t for t in tokens if re.match(r'^[A-Za-z0-9]+$', t)]
    
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]
    
    return tokens

def predict_spam(text):
    """
    Prédit si un email est spam (1) ou ham (0)
    """
    processed_tokens = preprocess_text(text)
    
    text_vectorized = vectorizer.transform([processed_tokens])
    
    prediction = model.predict(text_vectorized)[0]
    
    decision_score = model.decision_function(text_vectorized)[0]
    
    return prediction, decision_score

st.title("📧 Détecteur de Spam Email")
st.markdown("### 🤖 Propulsé par SVM (Support Vector Machine)")
st.markdown("---")

st.markdown("""
#### 🎯 Mode d'emploi :
1. Collez le texte de votre email dans la zone ci-dessous
2. Cliquez sur **"🔍 Analyser"**
3. Découvrez si c'est un **SPAM** ou un email **LÉGITIME** !
""")

st.markdown("---")

email_text = st.text_area(
    "📝 Entrez votre email ici :",
    height=200,
    placeholder="Exemple : Congratulations! You've won $1000! Click here now..."
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("🔍 Analyser l'email", type="primary", use_container_width=True)

if analyze_button:
    
    if email_text.strip() == "":
        st.warning("⚠️ Veuillez entrer un texte avant d'analyser.")
    
    else:
        with st.spinner("🔄 Analyse en cours..."):
            
            prediction, score = predict_spam(email_text)
            
        st.markdown("---")
        st.subheader("📊 Résultat de l'analyse")
        
        if prediction == 1:  # SPAM
            st.error("### 🚨 SPAM DÉTECTÉ !")
            st.markdown("⚠️ Cet email est probablement un **spam**. Soyez prudent !")
            
            confidence = abs(score)
            st.metric(
                label="Score de confiance",
                value=f"{confidence:.2f}",
                help="Plus le score est élevé, plus le modèle est certain"
            )
            
            st.progress(min(confidence / 5, 1.0))  # Normaliser entre 0 et 1
            
            st.markdown("""
            #### 💡 Recommandations :
            - ❌ Ne cliquez pas sur les liens
            - ❌ Ne partagez pas vos informations personnelles
            - 🗑️ Supprimez cet email
            """)
        
        else: 
            st.success("### ✅ EMAIL LÉGITIME")
            st.markdown("✨ Cet email semble être **légitime** et sûr.")
            
            confidence = abs(score)
            st.metric(
                label="Score de confiance",
                value=f"{confidence:.2f}",
                help="Plus le score est élevé, plus le modèle est certain"
            )
            
            st.progress(min(confidence / 5, 1.0))
            
            st.markdown("""
            #### 💡 Note :
            - ✅ Cet email semble sûr
            - 🔍 Restez tout de même vigilant avec les liens inconnus
            """)

st.markdown("---")
st.subheader("💡 Exemples d'emails à tester")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**📩 Exemple SPAM :**")
    example_spam = """CONGRATULATIONS! You have been selected to receive a FREE $1000 Walmart gift card! 
Click here NOW to claim your prize before it expires! 
This is a LIMITED TIME offer! Act fast! 
Call 1-800-FAKE-NUM or visit www.suspicious-link.com"""
    
    st.text_area("", value=example_spam, height=150, key="spam_example", disabled=True)

with col2:
    st.markdown("**📬 Exemple HAM (Légitime) :**")
    example_ham = """Hi Sarah,

I hope this email finds you well. I wanted to follow up on our meeting 
from yesterday regarding the Q4 project timeline. 

Could you please send me the documents we discussed when you have a chance?

Thanks,
John"""
    
    st.text_area("", value=example_ham, height=150, key="ham_example", disabled=True)

st.markdown("---")

with st.expander("🔬 Détails techniques du modèle"):
    st.markdown("""
    **Algorithme utilisé :** Support Vector Machine (SVM) avec noyau linéaire
    
    **Prétraitement appliqué :**
    1. Normalisation en minuscules
    2. Tokenisation (découpage en mots)
    3. Suppression des stopwords (mots vides)
    4. Suppression de la ponctuation
    5. Stemming (réduction à la racine)
    6. Vectorisation TF-IDF
    
    **Note :** Le modèle a été entraîné sur un dataset d'emails en anglais.
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>🤖 Application développée avec Streamlit</p>
    <p>📊 Machine Learning | 🔒 Protection contre le spam</p>
</div>
""", unsafe_allow_html=True)