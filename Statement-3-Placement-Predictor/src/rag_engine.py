import os
import re
import json
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from anthropic import Anthropic

client = Anthropic()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PDF_PATH = os.path.join(DATA_DIR, "INTERVIEW_EXPERIENCES.pdf")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

# --------------------------------------------------------------------------- #
# Interview experiences hardcoded from the PDF for portability
# --------------------------------------------------------------------------- #
INTERVIEW_EXPERIENCES = [
    {"id": "exp_1",  "company": "Google",      "role": "Software Engineer",     "level": "L3",           "difficulty": "Medium", "skills": ["Java","Data Structures","Graphs","Complexity Analysis"],    "focus": "Algorithmic Efficiency & Edge Case Handling",    "text": "Google SWE L3: Phone screen with sliding window + 4 onsite rounds. Heavy graph problems—shortest path in 2D grid, custom iterator. Forced O(N) thinking over O(N log N). Communication and trade-off discussion critical. Always dry-run with edge cases."},
    {"id": "exp_2",  "company": "Google",      "role": "Backend Engineer",       "level": "L4",           "difficulty": "Hard",   "skills": ["Go","Distributed Systems","gRPC","Multithreading"],          "focus": "Scalability & Concurrency",                      "text": "Google Backend L4: Design globally distributed rate-limiter. Redis vs in-memory caching trade-offs. Go concurrency (channels, mutexes). Thundering herd, graceful shutdowns. Optimize for a billion users."},
    {"id": "exp_3",  "company": "Google",      "role": "Data Scientist",         "level": "University Grad","difficulty": "Medium","skills": ["Python","Statistics","SQL","ML Theory"],                   "focus": "Experimental Design & Hypothesis Testing",       "text": "Google Data Scientist: A/B testing, product intuition, CTR metrics. Bayes Theorem, linear regression assumptions, SQL window functions. Explain P-values to non-technical stakeholders."},
    {"id": "exp_4",  "company": "Amazon",      "role": "SDE",                    "level": "SDE-1",        "difficulty": "Medium", "skills": ["C++","OOP","Linked Lists","Leadership Principles"],          "focus": "Coding Speed & Amazon LPs",                      "text": "Amazon SDE-1: OA with 2 medium DSA + workplace simulation. LP questions in every round. LRU Cache, Top K Frequent Elements. OOP design with encapsulation. STAR method for behavioral."},
    {"id": "exp_5",  "company": "Amazon",      "role": "Backend Engineer",       "level": "SDE-2",        "difficulty": "Hard",   "skills": ["Java","DynamoDB","Microservices","System Design"],           "focus": "Customer Obsession & Architectural Trade-offs",  "text": "Amazon Backend SDE-2: Flash Sale system—millions of concurrent requests. DB sharding, DAX caching, SQS message queues. Hard DP problem. Retry logic, idempotency keys, latency focus."},
    {"id": "exp_6",  "company": "Amazon",      "role": "Data Engineer",          "level": "L5",           "difficulty": "Medium", "skills": ["SQL","Python","Redshift","ETL Pipelines"],                  "focus": "Data Modeling & Warehouse Architecture",         "text": "Amazon Data Engineer L5: ETL pipelines, star-schema in Redshift, S3 raw logs. Recursive CTEs, window functions for YoY growth. OLAP vs OLTP, partitioning vs sorting keys."},
    {"id": "exp_7",  "company": "Microsoft",   "role": "Software Engineer",      "level": "SDE-1",        "difficulty": "Medium", "skills": ["C#","Azure","Trees","Arrays"],                               "focus": "Problem Solving & Code Quality",                 "text": "Microsoft SWE SDE-1: Clean code emphasis—proper naming, modularity. Binary tree serialization, string manipulation with stacks. Debugging round for multithreading bug. Azure/LINQ familiarity helps."},
    {"id": "exp_8",  "company": "Microsoft",   "role": "Full Stack Developer",   "level": "L62",          "difficulty": "Medium", "skills": ["TypeScript","React","NET Core","System Design"],             "focus": "End-to-End Feature Development",                 "text": "Microsoft Full Stack L62: Real-time task management UI with WebSockets. State persistence, error handling (skeletons/toasts), RESTful API design. Own a feature from DB to browser."},
    {"id": "exp_9",  "company": "Microsoft",   "role": "Data Scientist",         "level": "Senior",       "difficulty": "Hard",   "skills": ["Python","PyTorch","Statistics","NLP"],                      "focus": "Large Language Models & Mathematical Foundations","text": "Microsoft Senior Data Scientist: Transformer self-attention deep dive. Gradient descent, Adam vs SGD. SHAP/LIME explainability. Derive loss functions, discuss activation trade-offs."},
    {"id": "exp_10", "company": "Meta",        "role": "Software Engineer",      "level": "E4",           "difficulty": "Hard",   "skills": ["C++","Algorithms","System Design"],                         "focus": "Speed & Optimal DSA Solutions",                  "text": "Meta SWE E4: 2 LeetCode-style problems in 45 min—LCA binary tree, max path sum. Instagram News Feed design—pull vs push models, caching for celebrity accounts. Speed is everything."},
    {"id": "exp_11", "company": "Uber",        "role": "Backend Engineer",       "level": "L4",           "difficulty": "Hard",   "skills": ["Go","Distributed Systems","Geospatial Indexing","Kafka"],   "focus": "Real-time Systems",                              "text": "Uber Backend L4: Thread-safe worker pool in Go channels. Uber Ride Dispatch design—Quad-trees, H3 indexing for nearest driver. Kafka for car location event streams. CAP theorem, distributed locking."},
    {"id": "exp_12", "company": "Uber",        "role": "Data Scientist",         "level": "L3",           "difficulty": "Medium", "skills": ["Python","R","Experimentation","Pricing Models"],            "focus": "Marketplace Dynamics",                           "text": "Uber Data Scientist L3: Surge pricing algorithms, driver churn reduction. Wait time probability, supply-demand. A/B testing route-sharing feature—North Star metric, interference effects."},
    {"id": "exp_13", "company": "Airbnb",      "role": "Full Stack Developer",   "level": "L5",           "difficulty": "Medium", "skills": ["React","GraphQL","CSS-in-JS","UX Design"],                  "focus": "Craft & User Empathy",                           "text": "Airbnb Full Stack L5: Figma-to-React with pixel perfection, ARIA labels, responsive. GraphQL schema for Booking Management, DataLoaders for N+1 problem. Culture round. Performance profiling."},
    {"id": "exp_14", "company": "Airbnb",      "role": "Data Analyst",           "level": "Junior",       "difficulty": "Medium", "skills": ["SQL","Tableau","Business Metrics","Storytelling"],          "focus": "Data Storytelling",                              "text": "Airbnb Data Analyst Junior: Dataset analysis for revenue drop despite high traffic. SQL joins/aggregations, presentation to mock PM. Seasonality, listing cannibalization, actionable insights."},
    {"id": "exp_15", "company": "Netflix",     "role": "Backend Engineer",       "level": "Senior",       "difficulty": "Hard",   "skills": ["Java","Spring Boot","Chaos Engineering","Cassandra"],       "focus": "Cultural Fit & Reliability",                     "text": "Netflix Backend Senior: No LeetCode—deep dive past projects. Chaos Monkey testing, CDN architecture, 99.99% uptime. High Alignment Low Coupling culture. Self-driven engineers only."},
    {"id": "exp_16", "company": "Stripe",      "role": "Software Engineer",      "level": "L3",           "difficulty": "Medium", "skills": ["Ruby","API Design","Webhooks"],                             "focus": "Clean Code & Documentation",                     "text": "Stripe SWE L3: Work sample—integrate payment method into mock API, handle webhooks. Idempotency, network timeouts. Global Ledger system design. Code quality as if going to production."},
    {"id": "exp_17", "company": "Atlassian",   "role": "Software Engineer",      "level": "P3",           "difficulty": "Medium", "skills": ["Java","React","Jira API","Microservices"],                  "focus": "Values & Collaboration",                         "text": "Atlassian SWE P3: Values interview mandatory. Collaborative doc editor design—OT or CRDTs. Standard coding with unit tests. Admit what you don't know, show teamwork."},
    {"id": "exp_18", "company": "Atlassian",   "role": "Frontend Engineer",      "level": "P2",           "difficulty": "Medium", "skills": ["JavaScript","React","Testing Library","Webpack"],           "focus": "Component Library & Micro-frontends",            "text": "Atlassian Frontend P2: Component library shared across Jira and Trello. Global vs local state, bundle size optimization. Drag-and-drop Kanban. React Hooks, Jest/Cypress testing."},
    {"id": "exp_19", "company": "Adobe",       "role": "Software Engineer",      "level": "Entry",        "difficulty": "Medium", "skills": ["C++","Data Structures","OS","Graphics Algorithms"],         "focus": "CS Fundamentals & Performance",                  "text": "Adobe SWE Entry: Memory management, pointers, multithreading. Parallel image processing. Memory-efficient Linked Lists and Stacks. OS/DBMS concepts—buffers, garbage collection impact on CPU."},
    {"id": "exp_20", "company": "Salesforce",  "role": "Backend Engineer",       "level": "AMTS",         "difficulty": "Medium", "skills": ["Java","Apex","Multi-tenancy","Cloud Architecture"],        "focus": "SaaS Architecture & Design Patterns",            "text": "Salesforce Backend AMTS: Multi-tenant architecture—data isolation on shared DB. Singleton/Factory/Strategy patterns. Governor Limits. Build platforms others develop on top of."},
    {"id": "exp_21", "company": "Swiggy",      "role": "Backend Engineer",       "level": "SDE-2",        "difficulty": "Medium", "skills": ["Java","Redis","Kafka","Microservices"],                     "focus": "High Concurrency & Low Latency",                 "text": "Swiggy Backend SDE-2: 100k orders/minute during cricket match. Redis real-time tracking, Kafka order status. Custom Rate Limiter. Write-through vs Write-back caching, cache invalidation."},
    {"id": "exp_22", "company": "Swiggy",      "role": "Data Scientist",         "level": "Mid",          "difficulty": "Medium", "skills": ["Python","XGBoost","Logistics Optimization","Forecasting"],  "focus": "Last-Mile Delivery Optimization",                "text": "Swiggy Data Scientist: ETA prediction model with weather, traffic, restaurant prep time. Feature engineering for time-series, outlier handling. Interpretable, reliable models. Model monitoring in production."},
    {"id": "exp_23", "company": "Zomato",      "role": "Full Stack Developer",   "level": "SDE-1",        "difficulty": "Medium", "skills": ["React Native","Node.js","AWS"],                             "focus": "Mobile Performance",                             "text": "Zomato Full Stack SDE-1: Mobile-first—image loading optimization, list virtualization React Native, offline mode. Restaurant Discovery API design. GraphQL for slow 3G networks."},
    {"id": "exp_24", "company": "Razorpay",    "role": "Software Engineer",      "level": "SDE-2",        "difficulty": "Hard",   "skills": ["Go","PHP","MySQL","Payment Gateways"],                      "focus": "Transactional Integrity",                        "text": "Razorpay SWE SDE-2: Deadlock-free transaction system for money transfer. MySQL Isolation Levels, partial failure handling. Eventual consistency but audit-ready. Fintech precision."},
    {"id": "exp_25", "company": "Flipkart",    "role": "SDE",                    "level": "SDE-1",        "difficulty": "Hard",   "skills": ["Java","Machine Coding","LLD","HLD"],                        "focus": "Low-Level Design & Speed",                       "text": "Flipkart SDE SDE-1: Parking Lot System in 90 min—extensible classes, SOLID principles. DP (0/1 Knapsack), Graphs. No God Classes. Bug-free clean code speed is the metric."},
    {"id": "exp_26", "company": "Flipkart",    "role": "Data Analyst",           "level": "Junior",       "difficulty": "Medium", "skills": ["Excel","SQL","Python","Business Analytics"],                "focus": "Customer Retention Analytics",                   "text": "Flipkart Data Analyst Junior: Complex nested SQL for customer retention and churn over 6 months by region. Guesstimate round. SQL Window functions, logical assumption structure."},
    {"id": "exp_27", "company": "Meesho",      "role": "Backend Engineer",       "level": "SDE-2",        "difficulty": "Medium", "skills": ["Java","Spring Boot","Microservices","ElasticSearch"],       "focus": "Scalable Discovery",                             "text": "Meesho Backend SDE-2: Personalized product discovery for millions. ElasticSearch indexing, write-heavy workloads for seller price updates. Standard DSA Trees/Heaps. Reliable, simple, scalable systems."},
    {"id": "exp_28", "company": "CRED",        "role": "Full Stack Developer",   "level": "Senior",       "difficulty": "Hard",   "skills": ["React","Node.js","Framer Motion","Design Systems"],         "focus": "UI/UX Performance & Security",                   "text": "CRED Full Stack Senior: Complex 60fps animations in React—credit card flip, progress bar. Data encryption for card info. Virtual DOM vs direct DOM for animations. Artists who can code."},
    {"id": "exp_29", "company": "Zepto",       "role": "Backend Engineer",       "level": "SDE-1",        "difficulty": "Medium", "skills": ["Node.js","PostGIS","Real-time Systems","Redis"],            "focus": "10-Minute Delivery Constraints",                  "text": "Zepto Backend SDE-1: PostGIS geospatial queries for nearest delivery partner. Slot Booking real-time system. R-trees/Quad-trees for geospatial indexing. Every millisecond matters."},
    {"id": "exp_30", "company": "Dunzo",       "role": "Software Engineer",      "level": "Junior",       "difficulty": "Medium", "skills": ["Kotlin","Python","Optimization Algorithms"],                "focus": "Route Optimization",                             "text": "Dunzo SWE Junior: Traveling Salesman variation for multi-stop delivery route. Arrays/HashMaps coding. Dynamic pricing during high demand. Greedy algorithms for route optimization."},
    {"id": "exp_31", "company": "NVIDIA",      "role": "ML Engineer",            "level": "L3",           "difficulty": "Hard",   "skills": ["CUDA","C++","PyTorch","Kernels","Linear Algebra"],          "focus": "GPU Architecture & Optimization",                "text": "NVIDIA ML Engineer L3: Matrix multiplication CUDA kernel optimization using shared memory. PyTorch autograd internals, model inference latency. Memory coalescing, warp scheduling, TFLOPS."},
    {"id": "exp_32", "company": "NVIDIA",      "role": "Software Engineer",      "level": "SDE-2",        "difficulty": "Medium", "skills": ["C","C++","Device Drivers","Linux Internals"],               "focus": "System Software & Low-Level C",                  "text": "NVIDIA SWE SDE-2: Character driver implementation, DMA explanation. Bit manipulation, custom memory allocators. Pointer expertise, memory alignment, stack vs heap. Kernel-level debugging."},
    {"id": "exp_33", "company": "Intel",       "role": "Software Engineer",      "level": "Graduate",     "difficulty": "Medium", "skills": ["C","Assembly","Computer Architecture","Verilog"],           "focus": "Computer Organization",                          "text": "Intel SWE Graduate: Instruction pipelining, branch prediction, MESI cache coherency. Assembly code for array sum. Hennessy-Patterson architecture knowledge. How CPU executes C++."},
    {"id": "exp_34", "company": "Qualcomm",    "role": "Embedded Engineer",      "level": "Junior",       "difficulty": "Hard",   "skills": ["Embedded C","RTOS","I2C/SPI","Microcontrollers"],           "focus": "Real-time Constraints",                          "text": "Qualcomm Embedded Junior: Task scheduler on bare-metal, high-priority interrupts within microsecond window. I2C vs SPI, ISRs. No dynamic memory in critical sections. Reliability in mobile/automotive."},
    {"id": "exp_35", "company": "Oracle",      "role": "Backend Engineer",       "level": "IC2",          "difficulty": "Medium", "skills": ["Java","SQL","Database Internals","Cloud Infra"],            "focus": "RDBMS Internals",                                "text": "Oracle Backend IC2: B-Tree vs Hash Indexes, query optimization against full table scan. Java multithreading, Java Memory Model. ACID properties implementation, execution plans."},
    {"id": "exp_36", "company": "SAP",         "role": "Software Engineer",      "level": "Associate",    "difficulty": "Medium", "skills": ["Java","ABAP","Cloud Foundry","Distributed Systems"],        "focus": "Enterprise Design Patterns",                     "text": "SAP SWE Associate: Distributed financial data integrity. Clean code, unit testing. Dependency injection, interfaces. Boring, reliable code for enterprise supply chains."},
    {"id": "exp_37", "company": "Cisco",       "role": "Software Engineer",      "level": "Grade 4",      "difficulty": "Medium", "skills": ["Networking","TCP/IP","Python","C++","Routing Protocols"],   "focus": "Network Fundamentals",                           "text": "Cisco SWE Grade 4: URL-to-browser deep dive—DNS, TCP handshakes, BGP, ARP. Network Packet Filter with HashMaps. OSI model, router vs switch at packet level, Wireshark, socket programming."},
    {"id": "exp_38", "company": "VMware",      "role": "Backend Engineer",       "level": "MTS-1",        "difficulty": "Medium", "skills": ["Java","Virtualization","K8s","Distributed Systems"],        "focus": "Virtualization & Cloud Native",                  "text": "VMware Backend MTS-1: Hypervisor CPU overcommitment, containers vs VMs kernel sharing. Trees/Graphs coding. Kubernetes Tanzu—control plane vs data plane knowledge."},
    {"id": "exp_39", "company": "ServiceNow",  "role": "Software Engineer",      "level": "Junior",       "difficulty": "Medium", "skills": ["JavaScript","Java","SaaS","Workflow Engines"],             "focus": "Extensibility & Platform Engineering",           "text": "ServiceNow SWE Junior: Workflow Engine with drag-and-drop logic. Event-driven architecture, thousands of concurrent automated tasks. Flexible schemas and APIs for custom fields."},
    {"id": "exp_40", "company": "Snowflake",   "role": "Backend Engineer",       "level": "Senior",       "difficulty": "Hard",   "skills": ["C++","Database Internals","Storage Engines","Query Optimization"],"focus": "Columnar Storage & Performance",            "text": "Snowflake Backend Senior: Columnar storage formats, micro-partitioning, lock-free queue in C++. 45 min on query plan optimization. SIMD instructions, cache-conscious programming, beat benchmarks."},
    {"id": "exp_41", "company": "Amazon",      "role": "DevOps Engineer",        "level": "L4",           "difficulty": "Medium", "skills": ["AWS","Terraform","CI/CD","Shell Scripting"],                "focus": "Infrastructure as Code",                         "text": "Amazon DevOps L4: Terraform for HA VPC with public/private subnets, Auto Scaling. Blue-Green deployment, production rollback. AWS Well-Architected Framework. CloudWatch monitoring."},
    {"id": "exp_42", "company": "Microsoft",   "role": "Cloud Engineer",         "level": "Mid",          "difficulty": "Medium", "skills": ["Azure","PowerShell","ARM Templates","Networking"],          "focus": "Azure Cloud Security",                           "text": "Microsoft Cloud Engineer: Azure Functions, CosmosDB, Virtual Networks. Secure Public API with Azure API Management + OAuth2. LRS vs GRS vs ZRS storage. Security-first cloud design."},
    {"id": "exp_43", "company": "Google",      "role": "Cloud Engineer",         "level": "Senior",       "difficulty": "Hard",   "skills": ["GCP","Kubernetes","Anthos","Networking"],                   "focus": "Hybrid Cloud & Containerization",                "text": "Google Cloud Engineer Senior: Migrate legacy on-prem to GKE with Strangler Pattern. Istio service mesh, traffic splitting. Pods, Deployments, Ingress. SRE principles—Google invented the term."},
    {"id": "exp_44", "company": "Goldman Sachs","role": "Software Engineer",     "level": "Analyst",      "difficulty": "Medium", "skills": ["Java","Algorithms","Hashing","Math"],                       "focus": "Mathematical DSA & Logical Rigor",               "text": "Goldman Sachs SWE Analyst: Number theory (primes, GCD), combinatorics. First non-repeating character in stream of millions. Logical rigor, integrity behavioral round. Math-heavy coding."},
    {"id": "exp_45", "company": "JPMorgan",    "role": "Software Engineer",      "level": "Analyst",      "difficulty": "Medium", "skills": ["Java","SQL","Agile","Spring"],                              "focus": "Team Collaboration & Agile",                     "text": "JPMorgan SWE Analyst: Behavioral focus—conflict handling in teams. Basic Java/Spring and SQL joins/indexes. Agile and communication as important as coding in large corporate structure."},
    {"id": "exp_46", "company": "Bloomberg",   "role": "Software Engineer",      "level": "New Grad",     "difficulty": "Hard",   "skills": ["C++","Data Structures","Memory Management"],               "focus": "C++ Memory & Performance",                       "text": "Bloomberg SWE New Grad: Smart pointers, vector resizing, virtual functions. Trading Order Book with Heaps and Maps. Memory layout, cache miss avoidance. Deep C++ expertise required."},
    {"id": "exp_47", "company": "Bloomberg",   "role": "Data Scientist",         "level": "Mid",          "difficulty": "Medium", "skills": ["Python","Time Series","NLP","Finance"],                    "focus": "Financial Sentiment Analysis",                   "text": "Bloomberg Data Scientist: Twitter feed → stock market volatility model. LSTM vs Transformers for time-series. NLP signal extraction from massive text datasets. Information is core product."},
    {"id": "exp_48", "company": "Fintech Startup","role": "Backend Engineer",    "level": "SDE-2",        "difficulty": "Hard",   "skills": ["Go","PostgreSQL","Ledger Design","Distributed Transactions"],"focus": "Idempotency & Data Consistency",              "text": "Fintech Startup Backend SDE-2: Idempotency in payment retries. Double-entry bookkeeping ledger design. Postgres locks, unique constraints. Saga pattern. No eventual consistency—audit-ready."},
    {"id": "exp_49", "company": "AI Startup",  "role": "ML Engineer",           "level": "Mid",          "difficulty": "Medium", "skills": ["Python","LangChain","OpenAI API","Vector DBs"],            "focus": "RAG & Generative AI",                            "text": "AI Startup ML Engineer: RAG Q&A bot over PDFs using LangChain and Pinecone. Prompt engineering to reduce hallucinations. Build products around raw LLMs. LLM evaluation frameworks."},
    {"id": "exp_50", "company": "PayPal",      "role": "Backend Engineer",       "level": "SDE-2",        "difficulty": "Medium", "skills": ["Node.js","Microservices","API Gateways"],                   "focus": "API Design & Rate Limiting",                     "text": "PayPal Backend SDE-2: API gateway for millions of requests—protect internal services. Circuit Breakers (Hystrix). Service discovery, distributed tracing with Zipkin/Jaeger."},
]


def _get_embedding(text: str) -> list:
    """Get embedding from Anthropic's voyage model via the embeddings API."""
    # Use OpenAI-compatible embedding via anthropic client isn't available,
    # so we use a simple TF-IDF-like approach with chromadb's default embedder
    # which uses all-MiniLM-L6-v2 under the hood via sentence-transformers
    return None  # ChromaDB handles embeddings internally


def build_vector_store():
    """Build ChromaDB collection from interview experiences."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete existing collection to rebuild fresh
    try:
        chroma_client.delete_collection("interview_experiences")
    except Exception:
        pass

    ef = embedding_functions.DefaultEmbeddingFunction()
    collection = chroma_client.create_collection(
        name="interview_experiences",
        embedding_function=ef,
    )

    documents = [exp["text"] for exp in INTERVIEW_EXPERIENCES]
    metadatas = [
        {
            "company": exp["company"],
            "role": exp["role"],
            "level": exp["level"],
            "difficulty": exp["difficulty"],
            "skills": ", ".join(exp["skills"]),
        }
        for exp in INTERVIEW_EXPERIENCES
    ]
    ids = [exp["id"] for exp in INTERVIEW_EXPERIENCES]

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"[RAG] Vector store built with {len(documents)} experiences.")
    return collection


def get_collection():
    """Load existing collection or build it."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        ef = embedding_functions.DefaultEmbeddingFunction()
        collection = chroma_client.get_collection(
            name="interview_experiences",
            embedding_function=ef,
        )
        return collection
    except Exception:
        return build_vector_store()


def retrieve_experiences(tech_stack: list, top_k: int = 3) -> list:
    """
    RAG retrieval: find top-K most relevant interview experiences for a given tech stack.
    Returns list of dicts with company, role, and experience text.
    """
    query = "Interview experience for someone skilled in: " + ", ".join(tech_stack)
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=top_k)

    experiences = []
    for i in range(len(results["ids"][0])):
        exp_id = results["ids"][0][i]
        meta = results["metadatas"][0][i]
        doc = results["documents"][0][i]
        distance = results["distances"][0][i]
        cosine_similarity = 1 - distance  
        experiences.append({
            "rank": i + 1,
            "company": meta["company"],
            "role": meta["role"],
            "level": meta["level"],
            "difficulty": meta["difficulty"],
            "skills": meta["skills"],
            "experience": doc,
            "similarity_score": round(max(cosine_similarity, 0), 4),
        })
    return experiences


# --------------------------------------------------------------------------- #
# BONUS: Smart Experience Matcher using pure Cosine Similarity
# --------------------------------------------------------------------------- #

def _skill_vector(skills: list, all_skills: list) -> np.ndarray:
    """Create a binary skill vector for cosine similarity."""
    vec = np.zeros(len(all_skills))
    for skill in skills:
        for i, s in enumerate(all_skills):
            if skill.lower() in s.lower() or s.lower() in skill.lower():
                vec[i] = 1
    return vec


def smart_experience_matcher(student_tech_stack: list, top_k: int = 3) -> list:
    """
    BONUS: Pure cosine similarity matcher.
    Compares student's tech stack against each interview experience's skill set.
    Returns top-K matches sorted by cosine similarity score.
    """
    # Build global skill vocabulary
    all_skills = list(
        set(skill for exp in INTERVIEW_EXPERIENCES for skill in exp["skills"])
    )

    student_vec = _skill_vector(student_tech_stack, all_skills)

    scored = []
    for exp in INTERVIEW_EXPERIENCES:
        exp_vec = _skill_vector(exp["skills"], all_skills)
        # Cosine similarity
        denom = np.linalg.norm(student_vec) * np.linalg.norm(exp_vec)
        if denom == 0:
            sim = 0.0
        else:
            sim = float(np.dot(student_vec, exp_vec) / denom)
        scored.append((sim, exp))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    return [
        {
            "rank": i + 1,
            "company": exp["company"],
            "role": exp["role"],
            "level": exp["level"],
            "difficulty": exp["difficulty"],
            "skills": ", ".join(exp["skills"]),
            "experience": exp["text"],
            "cosine_similarity": round(sim, 4),
        }
        for i, (sim, exp) in enumerate(top)
    ]


def generate_rag_response(profile: dict, experiences: list) -> str:
    """
    Use LLM to generate a personalized mentorship response
    combining the student profile with retrieved interview experiences.
    """
    context = "\n\n".join(
        f"[{i+1}] {exp['company']} - {exp['role']} ({exp['level']}):\n{exp['experience']}"
        for i, exp in enumerate(experiences)
    )

    prompt = f"""You are a senior placement mentor. A student has the following profile:
{json.dumps(profile, indent=2)}

Based on their profile, here are the most relevant interview experiences from seniors:

{context}

Provide personalized, actionable placement advice for this student in 3 sections:
1. **Strengths to Leverage** (based on their profile)
2. **Key Focus Areas** (what to improve before interviews)
3. **Recommended Interview Strategies** (specific tips from the retrieved experiences above)

Keep it concise, specific, and motivating."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


if __name__ == "__main__":
    build_vector_store()
    sample_stack = ["Python", "AWS", "Machine Learning", "SQL"]
    print("\n=== RAG Retrieval ===")
    results = retrieve_experiences(sample_stack, top_k=3)
    for r in results:
        print(f"{r['rank']}. {r['company']} - {r['role']} | Similarity: {r['similarity_score']}")

    print("\n=== Cosine Similarity Matcher (Bonus) ===")
    results2 = smart_experience_matcher(sample_stack, top_k=3)
    for r in results2:
        print(f"{r['rank']}. {r['company']} - {r['role']} | Cosine: {r['cosine_similarity']}")