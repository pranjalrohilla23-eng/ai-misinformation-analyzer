from transformers import pipeline
import urllib.parse
import feedparser

# Load models
fake_model = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")
sentiment_model = pipeline("sentiment-analysis")


def analyze_news(text, forwarded=False):
    text_en = text

    fake_result = fake_model(text_en)[0]
    sentiment = sentiment_model(text_en)[0]

    fake_label = fake_result['label']
    fake_score = fake_result['score']

    trust_score = fake_score if fake_label == "REAL" else (1 - fake_score)
    trust_score = int(trust_score * 100)

    flags = []

    if "urgent" in text_en.lower():
        flags.append("Uses urgency")
    if "forward" in text_en.lower():
        flags.append("Encourages blind sharing")
    if "!!!" in text_en:
        flags.append("Excessive punctuation")
    if sentiment['label'] == "NEGATIVE":
        flags.append("Emotionally manipulative tone")

    if forwarded:
        trust_score -= 10
        flags.append("Marked as forwarded message")

    trust_score = max(0, min(trust_score, 100))

    verdict = "⚠️ Likely Fake" if trust_score < 50 else "✅ Likely Reliable"

    return verdict, trust_score, flags


def generate_links(text):
    clean_query = text.replace("\n", " ").strip()[:80]
    encoded_query = urllib.parse.quote(clean_query)

    return {
        "Google News": f"https://news.google.com/search?q={encoded_query}",
        "Fact Check": f"https://www.google.com/search?q={encoded_query}+fact+check",
        "Alt News": f"https://www.altnews.in/?s={encoded_query}"
    }


def fetch_news(query):
    try:
        clean_query = query.replace("\n", " ").strip()[:100]
        encoded_query = urllib.parse.quote(clean_query)

        url = f"https://news.google.com/rss/search?q={encoded_query}"

        feed = feedparser.parse(url)

        articles = []
        for entry in feed.entries[:5]:
            articles.append(entry.title)

        return articles

    except Exception:
        return ["Could not fetch related news. Try a shorter or simpler input."]


def explain_flags(flags):
    if not flags:
        return "No major manipulation patterns detected."

    return "This content may be misleading because: " + ", ".join(flags)
