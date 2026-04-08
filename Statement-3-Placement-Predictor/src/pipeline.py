from llm_extractor import extract_profile, chat_collect_profile
from regression_model import predict, train_and_save
from rag_engine import (
    build_vector_store,
    retrieve_experiences,
    smart_experience_matcher,
    generate_rag_response,
)
 
 
def run_pipeline(input_text: str = None, use_chat: bool = False) -> dict:
    """
    Full pipeline execution.
    Args:
        input_text: Pre-collected text (from resume or form)
        use_chat:   If True, runs interactive chat to collect profile
    Returns:
        dict with all pipeline outputs
    """
    # Step 1: Collect & Extract Profile
    if use_chat or input_text is None:
        profile = chat_collect_profile()
    else:
        profile = extract_profile(input_text)
 
    tech_stack = profile.get("raw_tech_stack", [])
    print(f"\n✅ Extracted Profile:")
    for k, v in profile.items():
        print(f"   {k}: {v}")
 
    # Step 2: Regression Prediction
    score = predict(profile)
    print(f"\n📊 Placement Readiness Score: {score}/100")
 
    # Step 3: RAG Retrieval
    experiences_rag = retrieve_experiences(tech_stack, top_k=3)
    print(f"\n🔍 Top RAG-retrieved Experiences:")
    for exp in experiences_rag:
        print(f"   {exp['rank']}. {exp['company']} - {exp['role']} (sim: {exp['similarity_score']})")
 
    # Step 4 (Bonus): Smart Cosine Matcher
    experiences_cosine = smart_experience_matcher(tech_stack, top_k=3)
    print(f"\n🎯 Smart Cosine Matcher Results:")
    for exp in experiences_cosine:
        print(f"   {exp['rank']}. {exp['company']} - {exp['role']} (cosine: {exp['cosine_similarity']})")
 
    # Step 5: Generate Mentor Response
    mentor_response = generate_rag_response(profile, experiences_cosine)
    print(f"\n🧑‍🏫 Mentor Advice:\n{mentor_response}")
 
    return {
        "profile": profile,
        "readiness_score": score,
        "rag_experiences": experiences_rag,
        "cosine_matches": experiences_cosine,
        "mentor_advice": mentor_response,
    }
 
 
if __name__ == "__main__":
    # Build vector store on first run
    build_vector_store()
    # Train model on first run
    train_and_save()
    # Run demo pipeline
    sample_text = """
    My CGPA is 8.7 on a 10-point scale. I have done 2 internships at 
    mid-size product companies working on Java Spring Boot and AWS microservices.
    I have contributed to 3 open-source projects on GitHub including a popular 
    React component library. My projects include a scalable task management app 
    using Node.js, Redis, and PostgreSQL. My tech stack includes Python, Java, 
    React, AWS, Docker, and Kubernetes. I'm good at communication and have led 
    my college tech club for 2 years. My DSA is strong—I've solved 400+ LeetCode 
    problems (200 medium, 50 hard).
    """
    result = run_pipeline(input_text=sample_text)