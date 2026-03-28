from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import urllib.parse
import feedparser

# ================= MODELS =================

fake_model = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")
sentiment_model = pipeline("sentiment-analysis")
nli_model = pipeline("text-classification", model="facebook/bart-large-mnli")

semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# ================= HELPERS =================

def extract_claim(text):
    return text.strip().split(".")[0]


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
            articles.append(entry.title.lower())

        return articles

    except:
        return []


# ================= AI LOGIC =================

def semantic_evidence_score(text, articles):
    if not articles:
        return 0

    text_embedding = semantic_model.encode(text, convert_to_tensor=True)

    scores = []
    for article in articles:
        article_embedding = semantic_model.encode(article, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(text_embedding, article_embedding)
        scores.append(similarity.item())

    return max(scores)


def contradiction_score(claim, articles):
    if not articles:
        return 0

    scores = []

    for article in articles:
        result = nli_model(f"{article} </s> {claim}")[0]

        label = result['label']
        score = result['score']

        if label == "CONTRADICTION":
            scores.append(-score)
        elif label == "ENTAILMENT":
            scores.append(score)
        else:
            scores.append(0)

    if not scores:
        return 0

    return sum(scores) / len(scores)


# ================= MAIN FUNCTION =================

def analyze_news(text, forwarded=False):

    text_en = text.lower().strip()

    # -------- TYPE DETECTION --------

    if len(text_en.split()) < 8 and " is " in text_en:
        return "✅ Likely Reliable", 90, ["Simple factual statement"]

    if any(word in text_en for word in ["i think", "i believe"]):
        return "⚖️ Opinion (Not verifiable)", 50, ["Subjective statement"]

    flags = []

    if "urgent" in text_en:
        flags.append("Uses urgency")
    if "forward" in text_en:
        flags.append("Encourages blind sharing")
    if "!!!" in text_en:
        flags.append("Excessive punctuation")

    # -------- MODEL SCORE --------

    fake_result = fake_model(text_en)[0]
    sentiment = sentiment_model(text_en)[0]

    fake_label = fake_result['label']
    fake_score = fake_result['score']

    model_score = fake_score if fake_label == "REAL" else (1 - fake_score)

    # -------- EVIDENCE --------

    claim = extract_claim(text_en)
    articles = fetch_news(claim)

    semantic_score = semantic_evidence_score(claim, articles)
    contradiction = contradiction_score(claim, articles)

    # -------- FINAL SCORE --------

    final_score = (
        model_score * 0.3 +
        semantic_score * 0.4 +
        (0.5 + contradiction / 2) * 0.3
    ) * 100

    if sentiment['label'] == "NEGATIVE":
        flags.append("Emotionally manipulative tone")

    if forwarded:
        final_score -= 10
        flags.append("Marked as forwarded message")

    final_score = int(max(0, min(final_score, 100)))

    verdict = "⚠️ Likely Fake" if final_score < 50 else "✅ Likely Reliable"

    return verdict, final_score, flags, articles
