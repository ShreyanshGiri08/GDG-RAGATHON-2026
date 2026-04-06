# 🚀 GDG RAGATHON 2026: Official Submission Hub

**Organized by GDG ML Wing, IIIT Lucknow** | **April 7th – April 9th, 2026**

Welcome to the official repository for **RAGATHON 2026**. This 5-day challenge is designed to push your limits in **Retrieval-Augmented Generation (RAG)**, **Vector Databases**, and **Predictive ML Integration**.

---

## 📖 Essential Documentation

Before writing your first line of code, you **must** read the following documents located in the `docs/official/` folder:

1. [**Official Problem Statements**](./docs/official/GDG_Ragathon_2026.pdf): Deep dive into the tasks, scoring metrics, and datasets.

2. [**Contribution Guidelines**](./docs/official/CONTRIBUTION_GUIDELINES.pdf): Rules for Git etiquette, folder structure, and submission standards.

---

## 📂 Repository Structure

Participants must maintain their work within these specific directories. Do not modify the root structure.

Plaintext

```
GDG-RAGATHON-2026/
├── docs/
│   └── official/                    # 📜 Problem Statements & Guidelines
├── Statement-1-Insurance-Decoder/   # 🧊 Easy (20 pts)
│   ├── docs/                        # Sample Policy PDF
│   ├── src/                         # Core RAG Logic
│   └── README.md                    # Module Docs & Bonus Claim
├── Statement-2-Lucknow-Foodie/      # 🍕 Medium (40 pts)
│   ├── dataset/                     # Scraped Restaurant Data
│   ├── src/                         # Hybrid Search Logic
│   └── README.md                    # Module Docs
├── Statement-3-Placement-Predictor/ # 🎓 Hard (70 pts)
│   ├── data/                        # CSV Dataset & Interview Records
│   ├── src/                         # Hybrid AI Logic (LLM + Regression)
│   └── README.md                    # Module Docs & Bonus Claim
├── .gitignore                       # 🛡️ Essential (Ensures .env is not leaked)
└── requirements.txt                 # 📦 Project-wide dependencies
```

---

## 🏁 Getting Started

### 1. Fork & Clone

- **Fork:** One member must fork this repo to their account.
    
- **Collaborate:** Add your teammates under **Settings > Collaborators** on your fork.
    
- **Clone:**

Bash
```
git clone https://github.com/[YOUR_USERNAME]/GDG-RAGATHON-2026.git cd GDG-RAGATHON-2026
```
    

### 2. Environment Setup

Bash

```
# Create a virtual environment
python -m venv venv

# Activate it (Windows: .\venv\Scripts\activate | Mac/Linux: source venv/bin/activate)
source venv/bin/activate 

# Install dependencies
pip install -r requirements.txt
```

---

## ⚖️ Submission Rules & Git Etiquette

1. **The 3-Member Rule:** All team members **must** have a visible commit history. Teams with zero-commit members will face disqualification.
    
2. **Atomic Commits:** Use descriptive, single-purpose commit messages (e.g., `feat: added chromaDB indexing for statement 1`).
    
3. **No Secrets:** Never commit `.env` files or API Keys. Use `.env.example` as a template.
    
4. **Detailed READMEs:** Each statement folder must have a README documenting your **Architecture**, **Tech Stack**, and **Bonus Attempts**.
    

### 🏷️ Pull Request (PR) Title Format

`[SUBMISSION] TeamName - Statements [1,2,3] + [BONUS: FeatureName]`

Example:
`[SUBMISSION] Team-Alpha - Statements [1,3] + [BONUS: Resume-Parser, Smart-Matcher]`


### 🏁 Final Submission Step

Once you have created your Pull Request on GitHub, you **must** submit the URL of that PR in the **Official Google Form** shared in the event group.

> [!CAUTION] **Submissions without a corresponding Google Form entry will not be evaluated.**

---

## 📞 Support & Contact

For technical hurdles or clarifications, contact the GDG ML Wing leads:

- **C Sai Sujit:** +91 81056 26006
    
- **Vedant Humbe:** +91 99231 58762
    
- **Abhinav Neema:** +91 94257 02277
