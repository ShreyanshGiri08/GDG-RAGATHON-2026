import os
import json
import re
from anthropic import Anthropic
 
client = Anthropic()  # reads ANTHROPIC_API_KEY from env
 
SYSTEM_PROMPT = """You are a placement profile extractor. 
Given a student's text (chat messages or resume content), extract their profile 
into a strict JSON object with EXACTLY these keys and value ranges:
 
{
  "Academic_Score": <float 0.0–10.0, CGPA on a 10-point scale>,
  "DSA_Skill": <int 1–10, data structures & algorithms ability>,
  "Project_Quality": <int 1–10, depth/impact of projects>,
  "Experience_Score": <int 1–10, internship/work experience value>,
  "OpenSource_Value": <1 or 10, 10 if has open-source contributions else 1>,
  "Soft_Skills": <int 1–10, communication & soft skills>,
  "Tech_Stack_Score": <int 1–10, breadth and relevance of tech stack>,
  "raw_tech_stack": <list of strings, detected technologies e.g. ["Python","AWS"]>
}
 
Rules:
- Return ONLY the JSON object, no markdown fences, no explanation.
- If a field cannot be determined, use a neutral default: 5 for ints, 6.0 for Academic_Score.
- CGPA: if given on 4-point scale, multiply by 2.5. If percentage, divide by 10.
- OpenSource_Value is binary: 10 if any open-source work mentioned, else 1.
- Tech_Stack_Score: 1–3 techs→3, 4–6 techs→5, 7–9 techs→7, 10+→10.
"""
 
 
def extract_profile(text: str) -> dict:
    """
    Extract a structured placement profile from free-form text.
    Returns a dict with model features + raw_tech_stack.
    """
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    raw = response.content[0].text.strip()
    # Strip any accidental markdown fences
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
    try:
        profile = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback defaults if parse fails
        profile = {
            "Academic_Score": 6.0,
            "DSA_Skill": 5,
            "Project_Quality": 5,
            "Experience_Score": 5,
            "OpenSource_Value": 1,
            "Soft_Skills": 5,
            "Tech_Stack_Score": 5,
            "raw_tech_stack": [],
        }
    return profile
 
 
def extract_profile_from_pdf_text(pdf_text: str) -> dict:
    """Wrapper: extract profile from resume PDF text content."""
    return extract_profile(pdf_text)
 
 
def chat_collect_profile() -> dict:
    """
    Interactive multi-turn chat to collect student info and extract profile.
    Used when no resume is uploaded.
    """
    print("\n=== Placement Profile Chatbot ===")
    print("Tell me about yourself! Share your CGPA, tech stack, projects, internships,")
    print("open-source work, and communication skills. Type 'done' when finished.\n")
 
    history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("done", "exit", "quit"):
            break
        history.append(user_input)
 
    combined = "\n".join(history)
    print("\n[Extracting profile from your inputs...]\n")
    return extract_profile(combined)