import streamlit as st
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
import re

# -------------------- Load Environment Variables --------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("❌ Missing OpenAI API key. Please set it in your .env file as OPENAI_API_KEY.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# -------------------- Load Excel Dataset --------------------
@st.cache_data
def load_data():
    df = pd.read_excel(r"C:\Users\lenovo\Desktop\patent\patent_data.xlsx")
    df.fillna("", inplace=True)
    return df

df = load_data()

# -------------------- Extract Keywords (Keep multi-word phrases) --------------------
def extract_keywords_from_llm(query):
    prompt = f"""
    You are a professional patent search assistant.
    From the following user query, extract only the key technical or domain-specific keywords or phrases.
    Keep multi-word technical terms together (e.g., "3D imaging", "thermal imaging system").
    Ignore generic words like "show", "find", "patent", "related", "give me", etc.

    Query: "{query}"

    Return only the keywords or phrases separated by commas.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in patent keyword extraction."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=80,
        temperature=0.2,
    )

    keywords_text = response.choices[0].message.content.strip()
    keywords = [kw.strip().lower() for kw in re.split(r',\s*', keywords_text) if kw.strip()]
    return keywords

# -------------------- Patent Search Logic --------------------
def search_patents(keywords):
    if not keywords:
        return pd.DataFrame()

    mask = pd.Series(False, index=df.index)

    for kw in keywords:
        # Use phrase-based matching
        pattern = re.escape(kw)
        kw_mask = (
            df["Industry Domain"].str.lower().str.contains(pattern, na=False) |
            df["Technology Area"].str.lower().str.contains(pattern, na=False) |
            df["Sub-Technology Area"].str.lower().str.contains(pattern, na=False) |
            df["Keywords"].str.lower().str.contains(pattern, na=False)
        )
        mask = mask | kw_mask  

    return df[mask]

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="AI-Powered Patent Search", layout="wide")
st.title(" AI-Powered Patent Search Engine")

query = st.text_input(
    "Enter your query:",
    placeholder="e.g. Show me patents related to 3D imaging"
)

if st.button("Search") or query:
    if not query.strip():
        st.warning("Please enter a search term.")
    else:
        with st.spinner(" Understanding your query..."):
            ai_keywords = extract_keywords_from_llm(query)

        st.info(f"🧩 Showing results for: **{', '.join(ai_keywords)}**")

        results = search_patents(ai_keywords)

        if results.empty:
            st.error(f"No patents found for: {', '.join(ai_keywords)}")
        else:
            st.success(f"✅ Found {len(results)} matching patents.")

            for _, row in results.iterrows():
                st.markdown("---")
                st.markdown(f"### 🧾 Patent Number: `{row['Patent Number']}`")
                st.markdown(f"**Title:** {row['Title']}**")
                st.markdown(f"**Abstract:** {row['Abstract']}**")
                st.markdown(f"**Industry Domain:** {row['Industry Domain']}**")
                st.markdown(f"**Technology Area:** {row['Technology Area']}**")
                st.markdown(f"**Sub-Technology Area:** {row['Sub-Technology Area']}**")
                st.markdown(f"**Keywords:** {row['Keywords']}**")
