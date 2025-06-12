import json
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import re
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
from dotenv import load_dotenv
import time
import uuid

# Configuration
MODEL_ID = "gemma2-9b-it"  # Groq-supported model
SLIDES_PER_TOPIC = 5
TOTAL_SLIDES = 40
STATE_FILE = "presentation_state.json"
OUTPUT_FILE = "presentation.json"
FAISS_INDEX_FILE = "faiss_index.bin"
PDF_FILE = os.path.join("Documents", "altai_final_14072020_cs_accessible2_jsd5pdf_correct-title_3AC24743-DE11-0B7C-7C891D1484944E0A_68342.pdf")

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables.")

# Always start fresh
if os.path.exists(STATE_FILE):
    os.remove(STATE_FILE)

# Initialize state
def initialize_state(prompt):
    """Initialize the presentation state with an outline and section summaries."""
    return {
        "prompt": prompt,
        "slides": [],
        "current_batch": 0,
        "outline": [
            "Introduction", "Human Agency and Oversight", "Technical Robustness and Safety",
            "Privacy and Data Governance", "Transparency", "Diversity, Non-discrimination and Fairness",
            "Societal and Environmental Well-being", "Accountability"
        ],
        "section_summaries": {
            section: [] for section in [
                "Introduction", "Human Agency and Oversight", "Technical Robustness and Safety",
                "Privacy and Data Governance", "Transparency", "Diversity, Non-discrimination and Fairness",
                "Societal and Environmental Well-being", "Accountability"
            ]
        },
        "used_chunks": set()
    }

def save_state(state):
    """Save the state to a JSON file."""
    state_copy = state.copy()
    state_copy["used_chunks"] = list(state["used_chunks"])
    with open(STATE_FILE, 'w') as f:
        json.dump(state_copy, f, indent=4)

def load_state():
    """Load existing state if available."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                state["used_chunks"] = set(state.get("used_chunks", []))
                return state
        except json.JSONDecodeError as e:
            print(f"⚠️ Error loading state file: {e}. Starting fresh.")
    return None

# Parse PDF
def parse_document(file_path):
    """Extract text from a PDF file."""
    try:
        with open(file_path, 'rb') as file:
            pdf = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
        return text
    except Exception as e:
        print(f"⚠️ Error parsing PDF: {e}")
        return ""

# Split content
def split_document(text):
    """Split document into tagged chunks based on ALTAI topics."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    chunks = splitter.split_text(text)
    tagged_chunks = []
    for idx, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        if "human agency" in chunk_lower or "oversight" in chunk_lower or "autonomy" in chunk_lower:
            topic = "Human Agency and Oversight"
        elif "robustness" in chunk_lower or "safety" in chunk_lower or "accuracy" in chunk_lower or "reliability" in chunk_lower or "cybersecurity" in chunk_lower or "encryption" in chunk_lower:
            topic = "Technical Robustness and Safety"
        elif "privacy" in chunk_lower or "data governance" in chunk_lower or "gdpr" in chunk_lower or "dpia" in chunk_lower:
            topic = "Privacy and Data Governance"
        elif "transparency" in chunk_lower or "explainability" in chunk_lower or "traceability" in chunk_lower:
            topic = "Transparency"
        elif "diversity" in chunk_lower or "fairness" in chunk_lower or "bias" in chunk_lower or "accessibility" in chunk_lower:
            topic = "Diversity, Non-discrimination and Fairness"
        elif "societal" in chunk_lower or "environmental" in chunk_lower or "sustainability" in chunk_lower or "workforce" in chunk_lower:
            topic = "Societal and Environmental Well-being"
        elif "accountability" in chunk_lower or "auditability" in chunk_lower or "risk management" in chunk_lower:
            topic = "Accountability"
        else:
            topic = "Introduction"
        tagged_chunks.append({"text": chunk, "topic": topic, "index": idx})
    return tagged_chunks

# Create FAISS index with TF-IDF
def create_faiss_index(tagged_chunks):
    """Create a FAISS index for document chunks using TF-IDF embeddings."""
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        texts = [chunk["text"] for chunk in tagged_chunks]
        tfidf_matrix = vectorizer.fit_transform(texts).toarray().astype(np.float32)
        dim = tfidf_matrix.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(tfidf_matrix)
        faiss.write_index(index, FAISS_INDEX_FILE)
        return index, tagged_chunks, vectorizer
    except Exception as e:
        print(f"⚠️ Error creating FAISS index: {e}")
        raise

# Retrieve relevant chunks
def retrieve_chunks(index, tagged_chunks, vectorizer, query, section, k=15, used_chunks=None):
    """Retrieve relevant document chunks using TF-IDF and FAISS."""
    if used_chunks is None:
        used_chunks = set()
    section_chunks = [i for i, chunk in enumerate(tagged_chunks) if chunk["topic"] == section and chunk["index"] not in used_chunks]
    if not section_chunks:
        section_chunks = [i for i, chunk in enumerate(tagged_chunks) if chunk["index"] not in used_chunks]
    query_tfidf = vectorizer.transform([query]).toarray().astype(np.float32)
    distances, indices = index.search(query_tfidf, k * 2)
    selected_chunks = []
    selected_indices = []
    for idx in indices[0]:
        if idx in section_chunks and len(selected_chunks) < k:
            selected_chunks.append(tagged_chunks[idx]["text"])
            selected_indices.append(tagged_chunks[idx]["index"])
    print(f"Query: {query}, Section: {section}")
    print("Retrieved chunks:")
    for i, chunk in enumerate(selected_chunks):
        print(f"Chunk {i+1}: {chunk[:200]}...")
    used_chunks.update(selected_indices)
    return selected_chunks, used_chunks

# Validate generated content
def validate_content(content, context):
    """Check if generated content aligns with document context."""
    content_words = set(content.lower().split())
    context_words = set(context.lower().split())
    common_words = content_words.intersection(context_words)
    overlap_ratio = len(common_words) / len(content_words) if content_words else 0
    if overlap_ratio < 0.3:
        print(f"⚠️ Warning: Content may contain information not in context. Overlap ratio: {overlap_ratio:.2f}")
    return True

# Initialize Groq client
def initialize_groq_client():
    """Initialize the Groq client with API key."""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        return client
    except Exception as e:
        print(f"⚠️ Error initializing Groq client: {e}")
        raise

# Test Groq API connection
def test_groq_client(client):
    """Test the Groq API connection with a simple request."""
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": "Test connection"}],
            max_tokens=10
        )
        print("✅ Groq API connection successful.")
        return True
    except Exception as e:
        print(f"⚠️ Groq API test failed: {e}")
        return False

# Safe JSON parsing
def safe_parse_json(text):
    """Parse JSON with fallback for malformed responses."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text = text.replace('\n', '\\n').replace('"', '\\"')
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{[^{}]*"title"\s*:\s*".*?",\s*"content"\s*:\s*".*?"\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            title_match = re.search(r'"title"\s*:\s*"([^"]*)"', text)
            content_match = re.search(r'"content"\s*:\s*"([^"]*)"', text)
            if title_match and content_match:
                return {"title": title_match.group(1), "content": content_match.group(1)}
            print(f"⚠️ Failed to parse JSON: {text[:200]}...")
            return None

# Generate a slide
def generate_slide_content(prompt, context, previous_slides, section, slide_number, client, section_summaries):
    """Generate a single slide using the Groq API, ensuring content aligns with ALTAI."""
    slide_summary = "; ".join([f"Slide {i+1}: {s['title']} - {s['content'][:100]}..." for i, s in enumerate(previous_slides[-3:])])
    section_summary = " ".join(section_summaries.get(section, []))

    title_templates = {
        "Introduction": [
            "What is ALTAI?", "Purpose of ALTAI", "Key Objectives",
            "Development Process", "Fundamental Rights Focus"
        ],
        "Human Agency and Oversight": [
            "Human Agency and Autonomy", "Human Oversight Mechanisms",
            "Preventing Over-Reliance", "User Awareness", "Social Interaction Risks"
        ],
        "Technical Robustness and Safety": [
            "Resilience to Attacks", "General Safety", "Accuracy",
            "Reliability and Reproducibility", "Cybersecurity Measures"
        ],
        "Privacy and Data Governance": [
            "Privacy Protection", "Data Governance Practices", "GDPR Compliance",
            "Data Protection Impact Assessment", "Privacy-by-Design"
        ],
        "Transparency": [
            "Traceability Measures", "Explainability of Decisions",
            "Communication of Limitations", "User Notification", "Data Quality"
        ],
        "Diversity, Non-discrimination and Fairness": [
            "Avoiding Unfair Bias", "Accessibility and Universal Design",
            "Stakeholder Participation", "Fairness Definitions", "Bias Monitoring"
        ],
        "Societal and Environmental Well-being": [
            "Environmental Impact", "Work and Skills Impact",
            "Societal and Democratic Effects", "Sustainability Goals", "Worker Consultation"
        ],
        "Accountability": [
            "Auditability Mechanisms", "Risk Management Processes",
            "Third-Party Auditing", "Ethics Review Boards", "Redress by Design"
        ]
    }
    subtopic = title_templates.get(section, ["General"])[(slide_number - 1) % len(title_templates.get(section, ["General"]))]
    suggested_title = subtopic

    full_prompt = f"""
You are an expert slide generator creating a presentation for a general audience about the Assessment List for Trustworthy AI (ALTAI). Generate a single slide using ONLY the information from the document context below. Do NOT introduce any information not explicitly present in the context. Ensure content is clear, concise (3-5 sentences or bullet points), and easy to understand, mirroring ALTAI’s question-driven or definitional style. The slide should build on previous slides and maintain narrative continuity.

Document Context:
{context[:1800]}

Previous Slides:
{slide_summary}

Section Summary:
{section_summary}

Topic: {prompt}
Section: {section} (Slide {slide_number})
Suggested Title: {suggested_title}

Return output ONLY in JSON format:
{{"title": "{suggested_title}", "content": "..."}}

STRICT RULE: Use ONLY the information in the provided context. Do NOT add external information or assumptions. Output MUST be a valid JSON object with 'title' and 'content' fields, and nothing else.
"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates clear and concise slide content in JSON format for a general audience."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.2,
                max_tokens=300
            )
            decoded = response.choices[0].message.content.strip()

            with open(f"raw_slide_{slide_number}_attempt_{attempt+1}.txt", "w") as debug_file:
                debug_file.write(decoded)

            slide = safe_parse_json(decoded)
            if not slide or not isinstance(slide, dict) or "title" not in slide or "content" not in slide:
                raise ValueError("Invalid JSON structure: 'title' and 'content' fields required.")

            validate_content(slide["content"], context)
            section_summaries[section].append(slide["content"][:100])
            return slide, section_summaries
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed for slide {slide_number}: {e}")
            if attempt == max_retries - 1:
                print(f"⚠️ Max retries reached for slide {slide_number}")
                if suggested_title == "What is ALTAI?" and section == "Introduction":
                    fallback_content = "ALTAI is a self-assessment tool developed by the High-Level Expert Group to ensure AI systems are trustworthy, based on seven key requirements."
                    return {"title": suggested_title, "content": fallback_content}, section_summaries
                if suggested_title == "Purpose of ALTAI" and section == "Introduction":
                    fallback_content = "ALTAI provides a practical framework for organizations to assess AI systems’ alignment with ethical principles and fundamental rights."
                    return {"title": suggested_title, "content": fallback_content}, section_summaries
                if suggested_title == "Key Objectives" and section == "Introduction":
                    fallback_content = "ALTAI aims to provide a self-assessment tool to ensure AI systems are trustworthy, focusing on seven key requirements: human agency, robustness, privacy, transparency, fairness, well-being, and accountability."
                    return {"title": suggested_title, "content": fallback_content}, section_summaries
                if suggested_title == "Accuracy" and section == "Technical Robustness and Safety":
                    fallback_content = "ALTAI requires assessing AI accuracy through metrics tailored to the task, ensuring reliable predictions for unseen data."
                    return {"title": suggested_title, "content": fallback_content}, section_summaries
                if suggested_title == "Reliability and Reproducibility" and section == "Technical Robustness and Safety":
                    fallback_content = "ALTAI emphasizes ensuring AI systems produce consistent and reproducible results across different conditions."
                    return {"title": suggested_title, "content": fallback_content}, section_summaries
                if suggested_title == "Cybersecurity Measures" and section == "Technical Robustness and Safety":
                    fallback_content = "ALTAI requires implementing encryption, intrusion detection, and regular security audits to protect AI systems."
                    return {"title": suggested_title, "content": fallback_content}, section_summaries
                return {
                    "title": suggested_title,
                    "content": f"Unable to generate content due to error: {str(e)}"
                }, section_summaries
        time.sleep(2)
    return {
        "title": suggested_title,
        "content": "Unable to generate content due to API error."
    }, section_summaries

# Main generation loop
def main():
    """Generate a 40-slide JSON presentation from the ALTAI PDF using Groq API."""
    prompt = "Generate a presentation about the Assessment List for Trustworthy AI (ALTAI) for a general audience."

    # Load or initialize state
    state = load_state()
    if state is None:
        state = initialize_state(prompt)
    used_chunks = state.get("used_chunks", set())

    # Initialize Groq client
    print("⚠️ Initializing Groq client...")
    try:
        client = initialize_groq_client()
        if not test_groq_client(client):
            print("❌ Groq API connection failed. Exiting.")
            return
    except Exception as e:
        print(f"❌ Failed to initialize Groq client: {e}")
        return

    print("📄 Reading PDF and generating context...")
    document_text = parse_document(PDF_FILE)
    if not document_text:
        print("❌ No text extracted from PDF. Exiting.")
        return

    tagged_chunks = split_document(document_text)
    index, tagged_chunks, vectorizer = create_faiss_index(tagged_chunks)

    total_batches = TOTAL_SLIDES // SLIDES_PER_TOPIC

    for batch in range(state["current_batch"], total_batches):
        print(f"\n🚀 Generating batch {batch + 1}/{total_batches}")
        section = state["outline"][batch % len(state["outline"])]

        for i in range(SLIDES_PER_TOPIC):
            slide_number = batch * SLIDES_PER_TOPIC + i + 1
            if slide_number > TOTAL_SLIDES:
                break

            subtopics = {
                "Introduction": ["what is altai", "purpose", "key objectives", "development", "fundamental rights"],
                "Human Agency and Oversight": ["agency", "oversight", "over-reliance", "awareness", "social risks"],
                "Technical Robustness and Safety": ["attacks", "safety", "accuracy", "reliability", "cybersecurity"],
                "Privacy and Data Governance": ["privacy", "governance", "gdpr", "dpia", "privacy-by-design"],
                "Transparency": ["traceability", "explainability", "limitations", "notification", "data quality"],
                "Diversity, Non-discrimination and Fairness": ["bias", "accessibility", "participation", "fairness", "universal design"],
                "Societal and Environmental Well-being": ["environment", "work", "society", "sustainability", "consultation"],
                "Accountability": ["auditability", "risk management", "third-party", "ethics board", "redress"]
            }
            query = f"{section} {subtopics.get(section, [''])[i % len(subtopics.get(section, ['']))]}"

            context_chunks, used_chunks = retrieve_chunks(index, tagged_chunks, vectorizer, query, section, k=15, used_chunks=used_chunks)
            context = " ".join(context_chunks)

            print(f"Slide {slide_number} - Query: {query}, Context length: {len(context)} characters")

            slide, state["section_summaries"] = generate_slide_content(
                prompt, context, state["slides"], section, slide_number, client, state["section_summaries"]
            )
            print(f"✅ Slide {slide_number}: {slide['title']}")
            state["slides"].append(slide)

            time.sleep(2)

        state["current_batch"] += 1
        state["used_chunks"] = used_chunks
        save_state(state)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(state["slides"], f, indent=4)

    # Validate output
    with open(OUTPUT_FILE, "r") as f:
        slides = json.load(f)
    for i, slide in enumerate(slides, 1):
        if "Unable" in slide["content"]:
            print(f"⚠️ Slide {i} failed: {slide['title']}")

    print(f"\n✅ All slides saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
