import os
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_recommendations(query, budget_filter=None, proximity_weight=True):
    """
    Retrieves restaurants based on semantic similarity and filters.
    Includes a creative 'Proximity Boost' for IIIT-L students.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load the existing vector DB 
    db = Chroma(
        persist_directory=os.path.join(base_dir, "food_db"), 
        embedding_function=embeddings
    )
    
    # Apply metadata filtering
    search_kwargs = {}
    if budget_filter:
        search_kwargs["filter"] = {"budget_tier": budget_filter}

    # Perform Semantic Search
    docs = db.similarity_search(query, k=10, **search_kwargs)
    
    if proximity_weight:
        # IIIT Lucknow Coordinates 
        iiitl_lat, iiitl_lon = 26.78, 81.02
        
        def calculate_score(doc):
            dist = abs(doc.metadata.get('lat', 0) - iiitl_lat) + \
                   abs(doc.metadata.get('lon', 0) - iiitl_lon)
            return dist

        # Re-rank results: items closer to campus move to the top
        docs = sorted(docs, key=calculate_score)

    # Return the top 4 most relevant and closest spots
    return docs[:4]

if __name__ == "__main__":
    print("--- Testing Member 1 Search Logic ---")
    results = get_recommendations("Where can I find spicy biryani?", budget_filter="₹")
    for d in results:
        print(f"Found: {d.metadata['name']} in {d.metadata.get('vibe')}")