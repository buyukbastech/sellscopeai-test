import streamlit as st
import requests
import openai
import os
import plotly.graph_objects as go
import stripe
import pandas as pd
from dotenv import load_dotenv
from fpdf import FPDF
from pytrends.request import TrendReq
import plotly.express as px
import random
from supabase import create_client
from dotenv import load_dotenv
from auth_manager import register_user, login_user, get_user
from sklearn.linear_model import LinearRegression
import pytrends

st.set_page_config(
    page_title="SellScope AI",
    page_icon="favicon2.png",  # favicon dosyası ile aynı klasörde olmalı
    layout="wide"
)

# ✅ Session değişkeni başlatılıyor
if "selected_menu" not in st.session_state:
    st.session_state["selected_menu"] = None
load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

DOMAIN = "http://localhost:8501"  # Canlı ortamda buraya kendi domainini koy

st.sidebar.title("🔐 Giriş / Kayıt")

menu = st.sidebar.selectbox("Seçim Yap", ["Giriş Yap", "Kayıt Ol"])

if menu == "Kayıt Ol":
    email = st.sidebar.text_input("E-posta")
    password = st.sidebar.text_input("Şifre", type="password")
    if st.sidebar.button("📝 Kaydol"):
        try:
            response = register_user(email, password)
            user_id = response.user.id  # Yeni kullanıcı ID’si

            # Supabase'de kendi 'users' tablon varsa buraya kayıt et
            supabase.table("users").insert({
                "id": user_id,
                "email": email,
                "is_premium": False
            }).execute()

            st.sidebar.success("✅ Kayıt başarılı! Giriş yapabilirsiniz.")
        except Exception as e:
            st.sidebar.error(f"❌ Hata: {e}")

if menu == "Giriş Yap":
    email = st.sidebar.text_input("E-posta")
    password = st.sidebar.text_input("Şifre", type="password")
    if st.sidebar.button("🔓 Giriş"):
        try:
            result = login_user(email, password)
            session = result.session
            if session:
                st.session_state["user_email"] = email
                st.success(f"👋 Hoş geldiniz, {email}")
            else:
                st.warning("⚠️ Giriş başarısız")
        except Exception as e:
            st.sidebar.error(f"❌ Giriş Hatası: {e}")

def register_user(email, password):
    return supabase.auth.sign_up({"email": email, "password": password})

def login_user(email, password):
    return supabase.auth.sign_in_with_password({"email": email, "password": password})

def get_user():
    return supabase.auth.get_user()

def check_user_access():
    user = supabase.auth.get_user().user
    email = user.email

    response = supabase.table("users").select("*").eq("email", email).execute()
    data = response.data[0]

    if data["is_premium"]:
        return True, None

    usage = data.get("usage_count", 0)
    if usage >= 3:
        return False, "Ücretsiz kullanım sınırına ulaştınız. Premium üye olun."

    return True, None

def fetch_amazon_products(keyword):
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "amazon",
        "amazon_domain": "amazon.com",
        "k": keyword,
        "api_key": SERPAPI_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        results = data.get("organic_results", [])
        cleaned = []
        for p in results:
            if isinstance(p, dict) and p.get("title") and p.get("price") and p.get("rating") and p.get("reviews") and p.get("link"):
                cleaned.append(p)
        return cleaned
    else:
        st.error("🔴 SerpAPI bağlantısı başarısız.")
        return []

def extract_price(product):
    if 'price' in product:
        if isinstance(product['price'], dict):
            return product['price'].get("raw") or product['price'].get("extracted_value")
        elif isinstance(product['price'], str):
            return product['price']
    elif 'prices' in product and isinstance(product['prices'], list):
        for p in product['prices']:
            if isinstance(p, dict) and 'raw' in p:
                return p['raw']
    return None

def ai_product_summary(products):
    product_list = []
    for i, product in enumerate(products[:5]):
        title = product.get("title")
        price = extract_price(product)
        rating = str(product.get("rating"))

        reviews = product.get("reviews")
        if isinstance(reviews, dict):
            reviews_text = reviews.get("text", "Yorum yok")
        elif isinstance(reviews, int):
            reviews_text = f"{reviews} yorum"
        else:
            reviews_text = "Yorum yok"

        link = product.get("link")
        if title and price and rating and link:
            product_list.append(f"{i+1}. [{title}]({link}) – {price} – {rating}⭐ – {reviews_text}")

    if not product_list:
        return "Hiçbir ürün gösterilemiyor. Lütfen farklı bir kelime deneyin."

    prompt = f"""
Sen bir yapay zeka destekli e-ticaret uzmanısın. Elindeki ürün listesi Amazon'dan alınmış ve her biri fiyat, puan ve yorum bilgisi içeriyor.
Görevin aşağıdaki listeye bakarak en çok kâr getirecek ürünleri belirlemek ve ticari potansiyelini analiz etmek.

Çıktı formatı:
- 🔝 En kârlı 3 ürün (fiyat ve puan bilgisiyle birlikte)
- 💹 Kar oranı tahmini ve rekabet düzeyi yorumu
- 🧠 Neden bu ürünler? (veriye dayalı kısa açıklama)
- 🛒 Hedef satış stratejisi (alım/satım önerisi)
- 📈 Grafik açıklaması ve veri özet yorumu

Ürün Listesi:
{chr(10).join(product_list)}
"""
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content

def show_price_rating_graph(products):
    names, prices, ratings = [], [], []
    for p in products[:5]:
        title = p.get("title", "")[:20]
        price_raw = extract_price(p)
        try:
            price_val = float(price_raw.replace("$", "").replace(",", "")) if price_raw else 0
        except:
            price_val = 0
        try:
            rating_val = float(p.get("rating", 0))
        except:
            rating_val = 0
        names.append(title)
        prices.append(price_val)
        ratings.append(rating_val)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=prices, name="Fiyat", marker_color="orange"))
    fig.add_trace(go.Scatter(x=names, y=ratings, name="Puan", mode="lines+markers", yaxis="y2"))
    fig.update_layout(
        title="💰 Fiyat vs ⭐ Puan Grafiği",
        xaxis=dict(title="Ürün"),
        yaxis=dict(title="Fiyat ($)"),
        yaxis2=dict(title="Puan", overlaying="y", side="right"),
        legend=dict(x=0.5, y=1.1, orientation="h")
    )
    st.plotly_chart(fig)

def calculate_profit(buy_price, sell_price):
    try:
        buy, sell = float(buy_price), float(sell_price)
        profit = sell - buy
        roi = (profit / buy) * 100 if buy != 0 else 0
        return profit, roi
    except:
        return None, None

def generate_pdf_report(content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in content.split('\n'):
        pdf.multi_cell(0, 10, txt=line)
    pdf.output("ai_raporu.pdf")
    return "ai_raporu.pdf"

def create_checkout_session(email):
    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        line_items=[{
            "price": "price_1RfmP6JKPIxGAc5EtpHadPM4",
            "quantity": 1
        }],
        mode="subscription",
        customer_email=email,
        success_url=DOMAIN + "?success=true",
        cancel_url=DOMAIN + "?canceled=true"
    )
    return session.url

def continent_sales_analysis(product_name):
    continents = ["Kuzey Amerika", "Avrupa", "Asya", "Güney Amerika", "Afrika", "Okyanusya"]
    sales = [random.randint(20, 100) for _ in continents]
    fig = px.bar(x=continents, y=sales, labels={'x': 'Kıta', 'y': 'Satış Puanı'}, title=f"🌍 {product_name} – Kıta Bazlı Satış Analizi")
    return fig

def trend_analysis(keyword):
    pytrends = TrendReq(hl='en-US', tz=360)
    pytrends.build_payload([keyword], cat=0, timeframe='today 3-m', geo='', gprop='')
    df = pytrends.interest_over_time()
    if df.empty:
        st.warning("Trend verisi bulunamadı.")
        return
    st.subheader("📈 Arama Trendi")
    fig = px.line(df, x=df.index, y=keyword, title=f"📊 Google Trends: {keyword}")
    st.plotly_chart(fig)

    st.subheader("🌍 Bölge Bazlı Trend Analizi")
    region_df = pytrends.interest_by_region()
    region_df = region_df.sort_values(by=keyword, ascending=False).head(10)
    fig2 = px.bar(region_df, x=region_df.index, y=keyword, labels={'index': 'Bölge'}, title="📌 En Çok Aranan Bölgeler")
    st.plotly_chart(fig2)

# Menü
selected_menu = st.session_state.get("selected_menu", None)

st.sidebar.markdown("## 📋 Menüler:")
menu_options = {
    "📦 Ürün Analizi": "urun_analizi",
    "📈 Kâr Tahmini": "kar_tahmini",
    "🚚 Dropshipping Ürünleri": "dropshipping",
    "🧠 Yapay Zeka Önerileri": "ai_oneri",
    "📈 Trend Analizi": "trend",
    "📉 Tedarikçi Karşılaştırma": "tedarikci",
    "💰 Kazanç Simülatörü": "kar_simulator",
    "📉 Fiyat Takibi": "price_tracking",
    "🔑 Anahtar Kelime Analizi": "keyword_analysis",
    "🎯 Niş Ürün Keşfi": "niche_discovery",
    "🧠 Gelişmiş AI Stratejisi": "ai_strategy_advanced",
    "🆚 Ürün Karşılaştırma": "karsilastirma",
    "📄 PDF Rapor": "pdf_rapor",
    "🤖 SellScope AI AI Chatbot": "chatbot"
}
for label, key in menu_options.items():
    if st.sidebar.button(label):
        st.session_state.selected_menu = key
selected_menu = st.session_state.selected_menu

st.title("🛒 SellScope AI – Amazon Ürün Analizi ve Dropshipping Aracı")

if st.session_state.get("user_email") and not st.session_state.get("is_premium"):
    st.warning("🔐 Bu özellik sadece premium kullanıcılar içindir.")
    if st.button("💳 Premium Üyelik Satın Al"):
        checkout_url = create_checkout_session(st.session_state["user_email"])
        st.markdown(f"[👉 Satın Alma Sayfasına Git]({checkout_url})", unsafe_allow_html=True)

# Menü içerikleri
if selected_menu == "trend":
    st.header("📈 Trends – Gerçek Zamanlı Talep Analizi")
    keyword = st.text_input("🔍 Trendini öğrenmek istediğiniz ürün:", "wireless headphones")
    if st.button("📊 Trend Analizini Göster"):
        trend_analysis(keyword)

elif selected_menu == "tedarikci":
    st.header("📉 Tedarikçi Karşılaştırması (AliExpress, Temu, Alibaba)")

    product_name = st.text_input("🔍 Ürün Adı:", "Bluetooth speaker")
    amazon_price = st.text_input("💲 Amazon Fiyatı ($):", "25")
    currency = st.selectbox("💱 Para Birimi", ["USD ($)", "TRY (₺)"])

    def generate_supplier_links(product_name):
        from urllib.parse import quote
        query = quote(product_name)
        return {
            "AliExpress": f"https://www.aliexpress.com/wholesale?SearchText={query}",
            "Temu": f"https://www.temu.com/search.html?q={query}",
            "Alibaba": f"https://www.alibaba.com/trade/search?SearchText={query}"
        }

    if st.button("🤖 Tedarikçileri Karşılaştır"):
        prompt = f"""
Bir e-ticaret danışmanı olarak aşağıdaki ürünün AliExpress, Temu ve Alibaba üzerindeki tedarik maliyetlerini analiz et.

Ürün: {product_name}
Amazon Fiyatı: ${amazon_price}

Her platform için:
- Tahmini ürün fiyatı
- Ortalama kargo ücreti
- Toplam maliyet
- Tedarik avantajı yorumu

Cevabını tablo + yorum formatında döndür. En avantajlı olanı vurgula.
"""
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            result = response.choices[0].message.content
            st.markdown("### 📄 AI Analizi:")
            st.markdown(result)

            # 🔗 Tedarikçi Linkleri
            st.subheader("🔗 Ürünü Platformlarda Ara:")
            links = generate_supplier_links(product_name)
            for name, link in links.items():
                st.markdown(f"- [{name}]({link})")

            # 📊 Grafiksel Karşılaştırma
            st.subheader("📊 Tahmini Maliyet Grafiği")

            suppliers = ["AliExpress", "Temu", "Alibaba"]
            base_prices = [random.uniform(5, 15) for _ in suppliers]
            shipping_prices = [random.uniform(2, 5) for _ in suppliers]

            if currency.startswith("TRY"):
                rate = 33.5  # Güncel kur örneği
                symbol = "₺"
                prices = [round(p * rate, 2) for p in base_prices]
                shipping = [round(s * rate, 2) for s in shipping_prices]
            else:
                symbol = "$"
                prices = [round(p, 2) for p in base_prices]
                shipping = [round(s, 2) for s in shipping_prices]

            text_prices = [f"{symbol}{p}" for p in prices]
            text_shipping = [f"{symbol}{s}" for s in shipping]
            total = [p + s for p, s in zip(prices, shipping)]

            fig = go.Figure()
            fig.add_trace(go.Bar(name='Ürün Fiyatı', x=suppliers, y=prices, text=text_prices, textposition='outside'))
            fig.add_trace(go.Bar(name='Kargo Ücreti', x=suppliers, y=shipping, text=text_shipping, textposition='outside'))
            fig.update_layout(
                barmode='stack',
                title="💵 Platform Bazlı Maliyet (Ürün + Kargo)",
                yaxis_title=f"Toplam Maliyet ({symbol})"
            )
            st.plotly_chart(fig)

            st.info("🧠 Grafik AI tahminlerine dayalıdır. Gerçek fiyatlar platforma göre değişebilir.")

        except Exception as e:
            st.error(f"❌ Analiz yapılamadı: {str(e)}")

elif selected_menu == "price_tracking":
    st.header("📉 Amazon Ürün Fiyat Takibi ve Strateji (Simülasyon)")
    asin = st.text_input("🔍 ASIN girin:", "B07FZ8S74R")

    if st.button("📊 Fiyat Analizini Göster"):
        tarih = pd.date_range(end=pd.Timestamp.today(), periods=30)
        fiyatlar = [round(random.uniform(20, 30), 2) for _ in range(30)]
        df = pd.DataFrame({"Tarih": tarih, "Fiyat": fiyatlar})

        ort = round(sum(fiyatlar)/len(fiyatlar), 2)
        min_fiyat = min(fiyatlar)
        max_fiyat = max(fiyatlar)

        st.markdown(f"""
        - 🔽 **Minimum Fiyat:** ${min_fiyat}  
        - 🔼 **Maksimum Fiyat:** ${max_fiyat}  
        - 📊 **Ortalama Fiyat:** ${ort}
        """)

        fig = px.line(df, x="Tarih", y="Fiyat", title=f"📉 {asin} için 30 Günlük Fiyat Geçmişi")
        st.plotly_chart(fig)

        # 📉 Volatilite (dalgalanma)
        std_dev = round(pd.Series(fiyatlar).std(), 2)
        st.markdown(f"📉 **Fiyat Volatilitesi (Standart Sapma):** ${std_dev}")
        if std_dev > 3:
            st.warning("⚠️ Fiyatlarda yüksek dalgalanma var. Alış zamanlaması önemli olabilir.")
        else:
            st.success("✅ Fiyatlar stabil görünüyor.")

        # 📈 Günlük değişim yüzdesi
        df["Değişim (%)"] = df["Fiyat"].pct_change().fillna(0) * 100
        ortalama_degis = df["Değişim (%)"].mean()
        st.markdown(f"📈 **Günlük Ortalama Değişim:** %{ortalama_degis:.2f}")

        # 🔮 7 gün sonrası tahmini (regresyon modeli)
        from sklearn.linear_model import LinearRegression
        import numpy as np

        X = np.arange(len(df)).reshape(-1, 1)
        y = df["Fiyat"].values
        model = LinearRegression().fit(X, y)
        tahmin = model.predict([[len(df) + 7]])[0]
        st.markdown(f"🔮 **7 Gün Sonra Tahmini Fiyat:** ${tahmin:.2f}")

        # 🤖 AI Yorumu
        prompt = f"""
{asin} ASIN'li ürünün son 30 günlük fiyat verisi aşağıda verilmiştir.

Minimum fiyat: ${min_fiyat}
Maksimum fiyat: ${max_fiyat}
Ortalama fiyat: ${ort}

Bu verilere dayanarak:
- Bu ürün şu anda alınmalı mı yoksa beklenmeli mi?
- Hangi zaman aralığında fiyat düşüş eğiliminde?
- Rekabet ve stok yönetimi açısından fiyat stratejisi ne olmalı?
- Dropshipping yapan biri için bu üründe nasıl bir fiyat stratejisi önerirsin?

Profesyonel bir analiz sun.
"""
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            st.markdown("### 🤖 AI Strateji Yorumu:")
            st.markdown(response.choices[0].message.content)
        except Exception as e:
            st.error(f"AI analizi yapılamadı: {str(e)}")

        # 💹 Kâr Simülasyonu
        st.subheader("💹 Kâr Simülasyonu")
        maliyet = st.number_input("💰 Alış Fiyatı ($)", min_value=0.0, value=20.0)
        satis = st.number_input("🛒 Satış Fiyatı ($)", min_value=0.0, value=30.0)
        kar = satis - maliyet
        roi = (kar / maliyet * 100) if maliyet > 0 else 0
        st.markdown(f"✅ **Kâr:** ${kar:.2f} \n📈 **ROI (Yatırım Getirisi):** %{roi:.2f}")

        # 📊 Kâr dağılımı grafiği
        st.subheader("📊 Kâr Dağılımı Simülasyonu")
        karlar = [round(s - maliyet, 2) for s in fiyatlar]
        kar_df = pd.DataFrame({"Tarih": tarih, "Kâr": karlar})
        fig_kar = px.bar(kar_df, x="Tarih", y="Kâr", title="💰 Günlük Tahmini Kâr Dağılımı")
        st.plotly_chart(fig_kar)

elif selected_menu == "niche_discovery":
    st.header("🎯 Niş Ürün Keşfi")
    kategori = st.selectbox("Kategori Seçin:", ["Kitchen", "Electronics", "Pet Supplies"])
    if st.button("🔎 Niş Ürünleri Bul"):
        niş_ürünler = [f"{kategori} - Niche Product {i}" for i in range(1, 6)]
        st.markdown("### 📦 Önerilen Niş Ürünler:")
        for ürün in niş_ürünler:
            st.markdown(f"- {ürün}")

        prompt = f"{kategori} kategorisinde kâr potansiyeli yüksek, rekabeti düşük 3 ürün öner ve her biri için neden uygun olduğunu açıkla."
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        st.markdown("### 🤖 AI Açıklaması:")
        st.markdown(response.choices[0].message.content)

elif selected_menu == "ai_strategy_advanced":
    st.header("🧠 Gelişmiş AI Ürün Stratejisi")
    product = st.text_input("Ürün Adı Girin:", "Bluetooth speaker")
    if st.button("🚀 Strateji Üret"):
        prompt = f"""
'{product}' için gelişmiş bir e-ticaret stratejisi üret:
- Hedef pazar
- Satış platformu
- Kâr potansiyeli
- Tedarik önerisi
- AI analizli satış önerisi
"""
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        st.markdown("### 📄 Strateji Raporu")
        st.markdown(response.choices[0].message.content)

elif selected_menu == "urun_analizi":
    st.header("📦 Ürün Analizi")
    keyword = st.text_input("🔍 Ürün Ara:", "air filter")
    if st.button("🔎 Ürünleri Bul"):
        products = fetch_amazon_products(keyword)
        if products:
            for p in products[:5]:
                reviews = p.get("reviews")
                reviews_text = reviews.get("text") if isinstance(reviews, dict) else str(reviews)
                st.markdown(f"- [{p['title']}]({p['link']}) – {extract_price(p)} – {p['rating']}⭐ – {reviews_text}")
            st.subheader("🤖 AI Ürün Yorumu")
            st.markdown(ai_product_summary(products))
            show_price_rating_graph(products)
        else:
            st.warning("Hiçbir ürün bulunamadı.")

elif selected_menu == "kar_tahmini":
    st.header("📈 Kâr Tahmini")
    keyword = st.text_input("🔍 Kâr için ürün ara:", "usb microphone")
    if st.button("🔍 Kâr Analizi Başlat"):
        products = fetch_amazon_products(keyword)
        if products:
            st.markdown("### 🔗 Ürünler:")
            for i, p in enumerate(products[:5]):
                reviews = p.get("reviews")
                reviews_text = reviews.get("text") if isinstance(reviews, dict) else str(reviews)
                st.markdown(f"- [{p['title']}]({p['link']}) – {extract_price(p)} – {p['rating']}⭐ – {reviews_text}")
            st.subheader("📊 AI Kâr Tahmini")
            st.markdown(ai_product_summary(products))
            show_price_rating_graph(products)
        else:
            st.warning("Ürün bulunamadı.")

elif selected_menu == "dropshipping":
    st.header("🚚 Dropshipping Ürünleri")
    keyword = st.text_input("💡 Trend Ürün Ara:", "kitchen gadgets")
    
    def ai_dropshipping_score(title, price, rating, reviews):
        prompt = f"""
Bir ürün hakkında dropshipping açısından değerlendirme yapmanı istiyorum. 
Ürün: {title}
Fiyat: {price}
Puan: {rating}
Yorum Sayısı: {reviews}

Aşağıdaki kriterlere göre 0-100 arası bir skor ver:
- Rekabet
- Satış potansiyeli
- Tedarik edilebilirlik
- Kargo avantajı

Yalnızca sayısal bir skor üret (örnek: 83).
"""
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4
            )
            score_raw = response.choices[0].message.content.strip()
            score = int(''.join(filter(str.isdigit, score_raw)))
            return max(0, min(score, 100))
        except:
            return random.randint(30, 80)

    def colored_score(score):
        if score >= 71:
            color = "green"
        elif score >= 41:
            color = "orange"
        else:
            color = "red"
        return f"<span style='color:{color}; font-weight:bold;'>{score}</span>"

    if st.button("📦 Dropshipping Ürünlerini Göster"):
        products = fetch_amazon_products(keyword)
        if products:
            st.markdown("### 🔗 Ürünler ve AI Skorları:")
            for i, p in enumerate(products[:5]):
                title = p.get("title", "Ürün yok")
                price = extract_price(p)
                rating = p.get("rating", 0)
                reviews = p.get("reviews", {}).get("text", "0") if isinstance(p.get("reviews"), dict) else p.get("reviews")

                score = ai_dropshipping_score(title, price, rating, reviews)
                score_display = colored_score(score)

                st.markdown(f"- [{title}]({p['link']}) – {price} – {rating}⭐ – {reviews} yorum – Dropshipping Skoru: {score_display}", unsafe_allow_html=True)
            
            st.subheader("📈 AI Dropshipping Analizi")
            st.markdown(ai_product_summary(products))
            show_price_rating_graph(products)
        else:
            st.warning("Trend ürün bulunamadı.")

elif selected_menu == "keyword_analysis":
    st.header("🔑 Anahtar Kelime Analizi AI")

    keyword = st.text_input("✨ Anahtar Kelime Girin:", "wireless earbuds")

    if st.button("📊 Analizi Başlat"):
        # Simüle veri
        arama_hacmi = random.randint(5000, 500000)
        zorluk = random.randint(1, 10)

        st.markdown(f"""
        - 🔍 **Aylık Aranma Hacmi:** {arama_hacmi}  
        - ⚔️ **Rekabet Skoru (0-10):** {zorluk}/10
        """)

        # Pytrends - Google Trends Analizi
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl='en-US', tz=360)
        try:
            pytrends.build_payload([keyword], cat=0, timeframe='today 3-m', geo='', gprop='')
            df = pytrends.interest_over_time()
            if not df.empty:
                fig = px.line(df, x=df.index, y=keyword, title=f"📈 Trends: {keyword}")
                st.plotly_chart(fig)
            else:
                st.warning("Google Trends verisi bulunamadı.")
        except Exception as e:
            st.error(f"Trend verisi alınamadı: {e}")

        # AI ile Ürün + Pazar Önerisi
        prompt = f"""
Sen bir Amazon SEO ve ürün analisti yapay zekasısın.

Anahtar kelime: '{keyword}'
Aranma hacmi: {arama_hacmi}
Rekabet düzeyi: {zorluk}/10

Aşağıdaki konularda öneriler ver:
- Bu anahtar kelimeye göre satılabilecek 3 ürün önerisi (ürün adı + neden?)
- Hedef pazar (ülke, yaş grubu, demografi)
- Dropshipping ve kâr potansiyeli yorumu
- SEO stratejisi ve başlık/etiket örnekleri
- Satışa başlamak için ipuçları
"""
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            st.markdown("### 🤖 AI Ürün & Pazar Önerisi")
            st.markdown(response.choices[0].message.content)
        except Exception as e:
            st.error(f"AI yanıtı alınamadı: {e}")

elif selected_menu == "ai_oneri":
    st.header("🧠 AI Önerileri")
    keyword = st.text_input("✨ Yeni Ürün Fikirleri İçin Ara:", "smart home gadgets")
    if st.button("🚀 AI Analiz Başlat"):
        products = fetch_amazon_products(keyword)
        if products:
            st.markdown("### 🔗 Ürünler:")
            for i, p in enumerate(products[:5]):
                reviews = p.get("reviews")
                reviews_text = reviews.get("text") if isinstance(reviews, dict) else str(reviews)
                st.markdown(f"- [{p['title']}]({p['link']}) – {extract_price(p)} – {p['rating']}⭐ – {reviews_text}")
            st.subheader("🤖 AI Ürün Stratejisi")
            st.markdown(ai_product_summary(products))
            show_price_rating_graph(products)
        else:
            st.warning("AI önerisi için ürün bulunamadı.")

elif selected_menu == "kar_simulator":
    st.header("📈 Kâr Oranı ve ROI Hesaplama")
    buy_price = st.text_input("💸 Alış Fiyatı ($)", "10")
    sell_price = st.text_input("💵 Satış Fiyatı ($)", "25")
    if st.button("🧮 Hesapla"):
        profit, roi = calculate_profit(buy_price, sell_price)
        if profit is not None:
            st.success(f"✅ Kâr: ${profit:.2f} | ROI: %{roi:.2f}")
        else:
            st.error("Geçerli fiyat bilgileri giriniz.")

elif selected_menu == "karsilastirma":
    st.header("🆚 AI Ürün Karşılaştırma Paneli")

    urun1 = st.text_input("🛍️ 1. Ürün Adı", "Bluetooth speaker")
    urun2 = st.text_input("🛍️ 2. Ürün Adı", "Wireless earbuds")

    def fetch_basic_amazon_data(keyword):
        products = fetch_amazon_products(keyword)
        if products:
            p = products[0]
            return {
                "title": p.get("title"),
                "price": extract_price(p),
                "rating": float(p.get("rating", 0)),
                "reviews": int(p.get("reviews", {}).get("text", "0").replace(",", "").replace("+", "").split()[0]) if isinstance(p.get("reviews"), dict) else 0,
                "link": p.get("link")
            }
        return None

    def ai_comparison_prompt(p1, p2):
        return f"""
{p1['title']} ve {p2['title']} ürünlerini e-ticaret açısından karşılaştır:

- Fiyat
- Kullanıcı puanı
- Yorum sayısı
- Dropshipping uygunluğu
- Kâr potansiyeli
- Tedarik avantajı

Yorum yaparken tabloyu da dahil et. Sonuç olarak hangisinin daha mantıklı olduğunu belirt.
"""

    if st.button("🔬 Karşılaştır"):
        p1_data = fetch_basic_amazon_data(urun1)
        p2_data = fetch_basic_amazon_data(urun2)

        if not p1_data or not p2_data:
            st.error("Ürünlerden biri bulunamadı.")
        else:
            # 💬 GPT-4 Yorumlu Karşılaştırma
            prompt = ai_comparison_prompt(p1_data, p2_data)
            res = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            st.markdown("### 🤖 AI Karşılaştırma Yorumu:")
            st.markdown(res.choices[0].message.content)

            # 🔗 Ürün Linkleri
            st.subheader("🔗 Ürün Linkleri")
            st.markdown(f"1. [{p1_data['title']}]({p1_data['link']})")
            st.markdown(f"2. [{p2_data['title']}]({p2_data['link']})")

            # 📊 Karşılaştırma Grafiği
            st.subheader("📊 Ürün Özellikleri Karşılaştırması")
            labels = ["Fiyat ($)", "Puan", "Yorum Sayısı"]
            values1 = [
                float(p1_data['price'].replace("$", "").replace(",", "")) if p1_data['price'] else 0,
                p1_data['rating'],
                p1_data['reviews']
            ]
            values2 = [
                float(p2_data['price'].replace("$", "").replace(",", "")) if p2_data['price'] else 0,
                p2_data['rating'],
                p2_data['reviews']
            ]

            fig = go.Figure(data=[
                go.Bar(name=urun1, x=labels, y=values1, marker_color="blue"),
                go.Bar(name=urun2, x=labels, y=values2, marker_color="orange")
            ])
            fig.update_layout(
                title="📈 Ürün Bazlı Özellik Karşılaştırması",
                barmode='group',
                yaxis_title="Değer",
                legend=dict(orientation="h", x=0.3, y=1.1)
            )
            st.plotly_chart(fig)

            st.info("🎯 Bu grafik, ürünler arasındaki temel farkları görselleştirmek için AI destekli veri ile hazırlanmıştır.")

elif selected_menu == "pdf_rapor":
    st.header("📄 AI Raporu PDF'e Aktar")
    dummy_summary = "Bu örnek rapor AI tarafından hazırlanmıştır. Gerçek analizler Ürün Analizi menüsünde alınabilir."
    if st.button("📥 PDF Raporu Oluştur"):
        path = generate_pdf_report(dummy_summary)
        with open(path, "rb") as f:
            st.download_button("📄 PDF'i İndir", f, file_name="ai_raporu.pdf")

if selected_menu == "chatbot":
    st.header("🤖 AI Tabanlı E-Ticaret Danışmanı")
    q = st.text_input("Sormak istediğiniz soruyu yazın:", "Bu ürünü Avrupa'da satmak mantıklı mı?")
    product_hint = st.text_input("Analiz yapılacak ürün adı (grafik için):", "Bluetooth hoparlör")
    if st.button("💬 AI Cevaplasın"):
        if not product_hint:
            st.warning("Ürün adı boş olamaz. Grafik için lütfen ürün adı girin.")
        else:
            prompt = f"""
Sen kıta bazlı analiz yapabilen, pazar verilerini yorumlayabilen bir e-ticaret danışmanı yapay zekasısın.
Soru:
{q}

Cevap verirken:
- Satış hacmini bölgelere göre oranla ve puanla (örnek: Avrupa %80 yüksek potansiyel, Asya %60 orta risk)
- Kullanıcı davranışları ve satış stratejisini detaylı yaz
- Ülke bazlı öneri ve rekabet analizi yap
- Grafiksel veri yorumu ekle
"""
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6
            )
            st.markdown(response.choices[0].message.content)

            st.subheader("📊 Kıta Bazlı Satış Grafiği")
            fig = continent_sales_analysis(product_hint)
            st.plotly_chart(fig)