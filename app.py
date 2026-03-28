import streamlit as st
from utils import analyze_news, generate_links
import pdfplumber
import pytesseract
from PIL import Image

st.set_page_config(page_title="AI Misinformation Analyzer", layout="centered")

st.markdown("""
<h1 style='text-align: center; color: #00ffe1;'>🧠 AI Misinformation Analyzer</h1>
""", unsafe_allow_html=True)

st.write("Explainable AI for misinformation detection")

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

forwarded = st.checkbox("This is a forwarded message")

if st.button("Analyze") and user_input:

    # BONUS TRIM
    if len(user_input) > 500:
        st.warning("Input too long, trimming...")
        user_input = user_input[:500]

    verdict, score, flags, articles = analyze_news(user_input, forwarded)

    st.subheader(verdict)
    st.progress(score / 100)
    st.write(f"Trust Score: {score}%")

    if score < 40:
        st.error("Low credibility")
    elif score < 70:
        st.warning("Uncertain credibility")
    else:
        st.success("High credibility")

    # Flags
    st.subheader("🚨 Red Flags")
    if flags:
        for f in flags:
            st.write(f"- {f}")
    else:
        st.write("No obvious red flags")

    # Explanation
    st.subheader("🧠 Explanation")
    st.write("This decision is based on semantic similarity, contradiction detection, and model confidence.")

    # Links
    st.subheader("🔎 Verify Yourself")
    links = generate_links(user_input)
    for name, link in links.items():
        st.write(f"{name}: {link}")

    # Evidence
    st.subheader("📰 Supporting Evidence")
    if articles:
        for a in articles:
            st.write("- " + a)
    else:
        st.write("No strong supporting evidence found")

# History
st.subheader("📜 Recent Checks")
for item in st.session_state.history[-5:]:
    st.write(item)

st.markdown("---")
st.write("Built by Pranjal 🚀")
