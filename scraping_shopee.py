from google_play_scraper import reviews, Sort
import pandas as pd

# Scraping review dari aplikasi Shopee
result, _ = reviews(
    'com.shopee.id',  # Package name aplikasi Shopee Indonesia
    lang='id',        # Bahasa Indonesia
    country='id',     # Play Store Indonesia
    sort=Sort.NEWEST, # Urutkan dari yang terbaru
    count=10500       # Ambil 10500 review pertama
)

# Simpan ke DataFrame
df = pd.DataFrame(result)

# Simpan ke CSV
df.to_csv("shopee_reviews.csv", index=False)

print("Scraping selesai! Data disimpan dalam shopee_reviews.csv")
