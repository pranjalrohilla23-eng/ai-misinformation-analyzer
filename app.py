import streamlit as st
from utils import analyze_news, generate_links, fetch_news, explain_flags
import pdfplumber
import pytesseract
from PIL import Image

st.set_page_config(page_title="AI Misinformation Analyzer", layout="centered")

# Styling
st.markdown("""
<style>
.big-title {
    font-size: 36px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🧠 AI Misinformation Analyzer</p>', unsafe_allow_html=True)
st.write("Detect fake news • Understand why • Verify smarter")

# History
if "history" not in st.session_state:
    st.session_state.history = []

# Input type
option = st.radio("Choose input type:", ["Text", "Upload PDF", "Upload Image"])

user_input = ""

if option == "Text":
    user_input = st.text_area("Paste message or claim:")

elif option == "Upload PDF":
    file = st.file_uploader("Upload PDF", type=["pdf"])
    if file:
        with pdfplumber.open(file) as pdf:
            user_input = "".join([page.extract_text() or "" for page in pdf.pages])

elif option == "Upload Image":
    file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if file:
        image = Image.open(file)
        user_input = pytesseract.image_to_string(image)

forwarded = st.checkbox("This is a forwarded WhatsApp message")

if st.button("Analyze") and user_input:

    verdict, score, flags = analyze_news(user_input, forwarded)

    # Save history
    st.session_state.history.append({
        "text": user_input[:100],
        "score": score,
        "verdict": verdict
    })

    # Result
    st.subheader(verdict)
    st.progress(score / 100)
    st.write(f"Trust Score: {score}%")

    if score < 40:
        st.error("Low credibility")
    elif score < 70:
        st.warning("Uncertain credibility")
    else:
        st.success("High credibility")

    # Breakdown
    st.subheader("📊 Analysis Breakdown")
    st.write("Model Score:", score)
    st.write("Pattern Risk:", len(flags))

    # Flags
    st.subheader("🚨 Red Flags")
    if flags:
        for f in flags:
            st.write(f"- {f}")
    else:
        st.write("No obvious red flags detected")

    # Explanation
    st.subheader("🧠 Explanation")
    st.write(explain_flags(flags))

    if st.button("Explain Simply"):
        st.write("This message may be trying to mislead using emotions or missing proof.")

    # Links
    st.subheader("🔎 Verify from sources")
    links = generate_links(user_input)
    for name, link in links.items():
        st.write(f"{name}: {link}")

    # News
    st.subheader("📰 Related News")
    news = fetch_news(user_input)
    for n in news:
        st.write("- " + n)

# History
st.subheader("📜 Recent Checks")
for item in st.session_state.history[-5:]:
    st.write(f"{item['verdict']} ({item['score']}%) - {item['text']}")
