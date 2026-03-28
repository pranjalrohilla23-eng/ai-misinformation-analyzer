from transformers import pipeline
import urllib.parse
import feedparser

# Load models
fake_model = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")
sentiment_model = pipeline("sentiment-analysis")


def analyze_news(text, forwarded=False):
    text_en = text  # no translation now

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
    query = urllib.parse.quote(text[:80])

    return {
        "Google News": f"https://news.google.com/search?q={query}",
        "Fact Check": f"https://www.google.com/search?q={query}+fact+check",
        "Alt News": f"https://www.altnews.in/?s={query}"
    }


def fetch_news(query):
    url = f"https://news.google.com/rss/search?q={query}"
    feed = feedparser.parse(url)

    articles = []
    for entry in feed.entries[:5]:
        articles.append(entry.title)

    return articles


def explain_flags(flags):
    if not flags:
        return "No major manipulation patterns detected."

    return "This content may be misleading because: " + ", ".join(flags)
