import os
import json
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

def get_recommendations(query, budget_filter=None, proximity_weight=True):
    """
    Retrieves restaurants based on semantic similarity and filters.
    Fixed to load from lucknow_eats.json instead of a missing folder.
    """
    # 1. Setup Paths based on your structure
    # __file__ is in src/, so we go up one level to reach 'dataset'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "..", "dataset", "lucknow_eats.json")
    
    # 2. Initialize Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    
    # 3. Load and Parse JSON
    if not os.path.exists(json_path):
        print(f"Error: Could not find JSON at {json_path}")
        return []

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 4. Convert JSON entries to LangChain Documents
    docs_to_index = []
    for item in data:
        # Create a string for semantic search (Name + Vibe + Signature Dish)
        content = f"{item.get('name')} {item.get('vibe')} {item.get('signature_dish', '')}"
        
        # Keep everything else in metadata for filtering and display
        doc = Document(
            page_content=content,
            metadata={
                "name": item.get("name"),
                "vibe": item.get("vibe"),
                "budget_tier": item.get("budget"), # Match your selectbox key
                "lat": item.get("lat", 0),
                "lon": item.get("lon", 0),
                "avg_price": item.get("avg_price", 0)
            }
        )
        docs_to_index.append(doc)

    # 5. Create In-Memory Vector Store
    # We use 'from_documents' instead of loading a persist_directory
    db = Chroma.from_documents(
        documents=docs_to_index,
        embedding=embeddings
    )
    
    # 6. Apply metadata filtering
    search_kwargs = {}
    if budget_filter and budget_filter != "Any":
        search_kwargs["filter"] = {"budget_tier": budget_filter}

    # 7. Perform Semantic Search
    results = db.similarity_search(query, k=10, **search_kwargs)
    
    # 8. Proximity Re-ranking (IIIT Lucknow Boost)
    if proximity_weight:
        iiitl_lat, iiitl_lon = 26.78, 81.02
        
        def calculate_score(doc):
            # Manhattan distance from campus
            dist = abs(doc.metadata.get('lat', 0) - iiitl_lat) + \
                   abs(doc.metadata.get('lon', 0) - iiitl_lon)
            return dist

        results = sorted(results, key=calculate_score)

    return results[:4]

if __name__ == "__main__":
    # Test logic
    print("--- Testing Member 1 Search Logic ---")
    try:
        results = get_recommendations("Where can I find spicy biryani?", budget_filter="₹")
        for d in results:
            print(f"Found: {d.metadata['name']} | Vibe: {d.metadata.get('vibe')}")
    except Exception as e:
        print(f"Test failed: {e}")