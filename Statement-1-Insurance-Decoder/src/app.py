# app.py
from data_loader import process_policy_document
from rag_engine import setup_rag_chain

def ask_insurance_bot(query):
    # Load documents and initialize the RAG chain 
    docs = process_policy_document("../docs/TITAN SECURE.pdf") 
    chain = setup_rag_chain(docs)
    
    # Enriched prompt to force Section/Clause citations and ELI5 formatting 
    enriched_query = (
        f"Answer the following query accurately based ONLY on the provided policy. "
        f"You MUST include the specific Section and Clause references in your text. "
        f"Then, provide a brief 'ELI5' (Explain Like I'm Five) summary. \n\n"
        f"Query: {query}"
    )
    
    # Execute RAG chain 
    result = chain.invoke({"query": enriched_query})
    
    answer = result["result"]
    sources = result["source_documents"]
    
    # Output the final answer 
    print(f"\n--- POLICY DECODER RESPONSE ---\n{answer}")
    
    # BONUS: Explicit Source Attribution logic 
    print("\n--- ✅ SOURCE ATTRIBUTION (BONUS) ---")
    unique_sources = set()
    for doc in sources:
        page_num = doc.metadata.get("page", 0) + 1
        source_name = doc.metadata.get("source", "Unknown Policy")
        # Prevent duplicate page listings 
        unique_sources.add(f"Document: {source_name} | Reference: Page {page_num}")
    
    for src in unique_sources:
        print(f"Verified via: {src}")

if __name__ == "__main__":
    # Test with a high-stakes query 
    user_q = "What happens if I forget to get a pre-authorization for an MRI?" 
    # Reference Expected: 40% penalty per Section 2.2 / Page 1 
    ask_insurance_bot(user_q)