import streamlit as st
import os, sys

# tells Python where to find search_logic.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from search_logic import get_recommendations

st.set_page_config(page_title="Lucknow Eats", page_icon="🍛")
st.title("🍛 Lucknow Eats — IIIT-L Food Finder")

query = st.text_input("What are you craving?",
    placeholder="spicy biryani, cheap snacks, rooftop cafe...")

budget = st.selectbox("Budget", ["Any", "₹", "₹₹", "₹₹₹"])
budget_filter = None if budget == "Any" else budget

if st.button("🔍 Find Food") and query:
    with st.spinner("Finding spots near campus..."):
        results = get_recommendations(query, budget_filter)
    for doc in results:
        m = doc.metadata
        st.markdown(f"**🍴 {m['name']}** — *{m.get('vibe','')} vibe*")
        st.markdown(f"💰 {m['budget']} | Avg ₹{m['avg_price']}")
        st.markdown(doc.page_content)
        st.divider()