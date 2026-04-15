import streamlit as st
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

# Load API key from Streamlit secrets before any langchain imports
os.environ["GOOGLE_API_KEY"] = st.secrets.get("GOOGLE_API_KEY", "")

from search_logic import get_recommendations

def run_app():
    st.set_page_config(page_title="IIIT-L Foodie Guide", page_icon="🍕")
    st.title("Lucknow Foodie Guide 🍕")
    st.markdown("### Helping IIIT Lucknow students find the best local eats!")

    query = st.text_input(
        "What are you in the mood for?",
        placeholder="e.g., Spicy Biryani or Aesthetic Cafe"
    )

    col1, col2 = st.columns(2)
    with col1:
        budget_tier = st.selectbox("Select Budget Tier", ["Any", "₹", "₹₹", "₹₹₹"])
    with col2:
        user_wallet = st.number_input("Your current budget (₹)", min_value=0, value=500)

    if st.button("Find Food"):
        if not query:
            st.warning("Please enter what you're looking for!")
            return

        b_filter = None if budget_tier == "Any" else budget_tier

        try:
            results = get_recommendations(query, budget_filter=b_filter)
        except Exception as e:
            st.error(f"Search failed: {e}")
            return

        if not results:
            st.error("No spots found! Try broadening your search.")
            return

        for res in results:
            with st.container():
                name     = res.metadata.get("name", "Unknown")
                vibe     = res.metadata.get("vibe", "—")
                budget   = res.metadata.get("budget", "—")
                avg_p    = res.metadata.get("avg_price", 0)

                st.subheader(f"📍 {name}")
                st.write(f"**Vibe:** {vibe} | **Budget tier:** {budget}")
                st.write(res.page_content)   # has location + signature dish baked in

                if user_wallet >= avg_p:
                    st.success(f"✅ You can afford a full meal here! (Est. ₹{avg_p})")
                else:
                    diff = avg_p - user_wallet
                    st.warning(f"⚠️ You're ₹{diff} short for a full meal here.")

                st.divider()

if __name__ == "__main__":
    run_app()