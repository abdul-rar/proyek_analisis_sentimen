import pickle
from google_play_scraper import reviews
import pandas as pd

# Load model yang sudah dilatih
svm_tfidf = pickle.load(open("model_dan_vectorizer/svm_tfidf_model.pkl", "rb"))
svm_bow = pickle.load(open("model_dan_vectorizer/svm_bow_model.pkl", "rb"))
rf_bow = pickle.load(open("model_dan_vectorizer/rf_bow_model.pkl", "rb"))

# Load vectorizer yang sudah dilatih sebelumnya
tfidf_vectorizer = pickle.load(open("model_dan_vectorizer/tfidf_vectorizer.pkl", "rb"))
bow_vectorizer = pickle.load(open("model_dan_vectorizer/bow_vectorizer.pkl", "rb"))

print("Model dan vectorizer berhasil dimuat!\n")

# Ambil review aplikasi DANA
app_reviews, _ = reviews(
    'id.dana',  # ID aplikasi DANA di Play Store
    lang='id',  # Bahasa Indonesia
    country='id',  # Negara Indonesia
    count=10  # Ambil 10 review terbaru
)

# Simpan ke DataFrame
df = pd.DataFrame(app_reviews)[['content']]
df.rename(columns={'content': 'review'}, inplace=True)

# **Transform review baru menggunakan vectorizer yang sama**
X_tfidf = tfidf_vectorizer.transform(df['review'])  # Gunakan transform()
X_bow = bow_vectorizer.transform(df['review'])  # Gunakan transform()

# Prediksi sentimen
df['sentiment_svm_tfidf'] = svm_tfidf.predict(X_tfidf)
df['sentiment_svm_bow'] = svm_bow.predict(X_bow)
df['sentiment_rf_bow'] = rf_bow.predict(X_bow)

# Menampilkan hasil prediksi
print("\n===== HASIL PREDIKSI SENTIMEN =====")
print(df[['review', 'sentiment_svm_tfidf', 'sentiment_svm_bow', 'sentiment_rf_bow']])