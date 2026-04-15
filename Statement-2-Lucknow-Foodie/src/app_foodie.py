import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from search_logic import get_recommendations
from langchain_google_genai import ChatGoogleGenerativeAI

# Focus: Student-Centric UI and Budget-Meal Planning Logic
def run_app():
    st.set_page_config(page_title="IIIT-L Foodie Guide", page_icon="🍕")
    st.title("Lucknow Foodie Guide 🍕")
    st.markdown("### Helping IIIT Lucknow students find the best local eats!")

    # User Inputs
    query = st.text_input("What are you in the mood for?", placeholder="e.g., Spicy Biryani or Aesthetic Cafe")

    col1, col2 = st.columns(2)
    with col1:
        budget_tier = st.selectbox("Select Budget Tier", ["Any", "₹", "₹₹", "₹₹₹"])
    with col2:
        user_wallet = st.number_input("Your current budget (₹)", min_value=0, value=500)

    if st.button("Find Food"):
        if query:
            # Map "Any" to None for the filter
            b_filter = None if budget_tier == "Any" else budget_tier

            # Fetch results from Member 1's logic
            results = get_recommendations(query, budget_filter=b_filter)

            if not results:
                st.error("No spots found! Try broadening your search.")
            else:
                for res in results:
                    with st.container():
                        st.subheader(f"📍 {res.metadata['name']}")
                        st.write(f"**Vibe:** {res.metadata['vibe']} | **Location:** {res.metadata['location']}")
                        st.info(f"✨ **Signature:** {res.metadata.get('signature_dish', 'Various items')}")

                        # --- CREATIVE FEATURE: Budget-Meal Planner ---
                        # Corrected variable 'avg_p' and logic
                        avg_p = res.metadata.get('avg_price', 0)
                        
                        if user_wallet >= avg_p:
                            st.success(f"✅ Student Choice: You can afford a full meal here! (Est. ₹{avg_p})")
                        else:
                            diff = avg_p - user_wallet
                            st.warning(f"⚠️ Wallet Alert: You're ₹{diff} short for a full meal here.")
                        
                        st.divider()
        else:
            st.warning("Please enter what you're looking for!")

if __name__ == "__main__":
    run_app()