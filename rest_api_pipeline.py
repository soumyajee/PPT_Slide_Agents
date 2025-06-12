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
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import shutil
from pathlib import Path
import uvicorn
# Configuration
MODEL_ID = "gemma2-9b-it"
SLIDES_PER_TOPIC = 5
TOTAL_SLIDES = 40
STATE_DIR = "presentation_states"
OUTPUT_DIR = "presentation_outputs"
FAISS_INDEX_DIR = "faiss_indices"
Path(STATE_DIR).mkdir(exist_ok=True)
Path(OUTPUT_DIR).mkdir(exist_ok=True)
Path(FAISS_INDEX_DIR).mkdir(exist_ok=True)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables.")

# FastAPI app
app = FastAPI(title="Presentation Generation API")

# Pydantic model for request
class QueryRequest(BaseModel):
    query: str
    session_id: str

# Initialize state
def initialize_state(prompt, session_id):
    state_file = os.path.join(STATE_DIR, f"state_{session_id}.json")
    if os.path.exists(state_file):
        os.remove(state_file)
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

def save_state(state, session_id):
    state_file = os.path.join(STATE_DIR, f"state_{session_id}.json")
    state_copy = state.copy()
    state_copy["used_chunks"] = list(state["used_chunks"])
    with open(state_file, 'w') as f:
        json.dump(state_copy, f, indent=4)

def load_state(session_id):
    state_file = os.path.join(STATE_DIR, f"state_{session_id}.json")
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
                state["used_chunks"] = set(state.get("used_chunks", []))
                return state
        except json.JSONDecodeError as e:
            print(f"⚠️ Error loading state file: {e}. Starting fresh.")
    return None

# Parse PDF
def parse_document(file_path):
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

# Create FAISS index
def create_faiss_index(tagged_chunks, session_id):
    faiss_index_file = os.path.join(FAISS_INDEX_DIR, f"faiss_index_{session_id}.bin")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    texts = [chunk["text"] for chunk in tagged_chunks]
    tfidf_matrix = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    dim = tfidf_matrix.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(tfidf_matrix)
    faiss.write_index(index, faiss_index_file)
    return index, tagged_chunks, vectorizer

# Retrieve chunks
def retrieve_chunks(index, tagged_chunks, vectorizer, query, section, k=15, used_chunks=None):
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
    used_chunks.update(selected_indices)
    return selected_chunks, used_chunks

# Validate content
def validate_content(content, context):
    content_words = set(content.lower().split())
    context_words = set(context.lower().split())
    common_words = content_words.intersection(context_words)
    overlap_ratio = len(common_words) / len(content_words) if content_words else 0
    if overlap_ratio < 0.3:
        print(f"⚠️ Warning: Content may contain information not in context. Overlap ratio: {overlap_ratio:.2f}")
    return True

# Initialize Groq client
def initialize_groq_client():
    try:
        client = Groq(api_key=GROQ_API_KEY)
        return client
    except Exception as e:
        print(f"⚠️ Error initializing Groq client: {e}")
        raise

# Test Groq API connection
def test_groq_client(client):
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

# Generate slide content
def generate_slide_content(prompt, context, previous_slides, section, slide_number, client, section_summaries):
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

# Generate presentation
async def generate_presentation(prompt, pdf_path, session_id):
    state = load_state(session_id)
    if state is None:
        state = initialize_state(prompt, session_id)

    client = initialize_groq_client()
    if not test_groq_client(client):
        raise HTTPException(status_code=500, detail="Groq API connection failed.")

    document_text = parse_document(pdf_path)
    if not document_text:
        raise HTTPException(status_code=400, detail="No text extracted from PDF.")

    tagged_chunks = split_document(document_text)
    index, tagged_chunks, vectorizer = create_faiss_index(tagged_chunks, session_id)

    total_batches = TOTAL_SLIDES // SLIDES_PER_TOPIC
    used_chunks = state.get("used_chunks", set())

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
        save_state(state, session_id)

    output_file = os.path.join(OUTPUT_DIR, f"presentation_{session_id}.json")
    with open(output_file, 'w') as f:
        json.dump(state["slides"], f, indent=4)

    return state["slides"]

# FastAPI Endpoints
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    pdf_dir = "uploaded_pdfs"
    Path(pdf_dir).mkdir(exist_ok=True)
    pdf_path = os.path.join(pdf_dir, f"{session_id}_{file.filename}")
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return JSONResponse(content={"session_id": session_id, "pdf_path": pdf_path})

@app.post("/generate_slides")
async def generate_slides(request: QueryRequest):
    session_id = request.session_id
    prompt = request.query
    pdf_path = os.path.join("uploaded_pdfs", [f for f in os.listdir("uploaded_pdfs") if f.startswith(session_id)][0])
    try:
        slides = await generate_presentation(prompt, pdf_path, session_id)
        return JSONResponse(content={"slides": slides, "session_id": session_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating slides: {str(e)}")

@app.get("/get_slides/{session_id}")
async def get_slides(session_id: str):
    output_file = os.path.join(OUTPUT_DIR, f"presentation_{session_id}.json")
    if not os.path.exists(output_file):
        raise HTTPException(status_code=404, detail="Slides not found.")
    with open(output_file, 'r') as f:
        slides = json.load(f)
    return JSONResponse(content={"slides": slides})
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)