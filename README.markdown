# Presentation Generation System

This project is a proof-of-concept for an LLM-based system that generates long-form slide presentations (40 slides) from a PDF document while maintaining contextual integrity across sequential generation steps. It addresses the AI Engineer Hiring Task requirements by ensuring context retention, factual accuracy, and structured output in JSON and PPT formats.

## Architecture and Design Decisions

### Overview

The system generates a 40-slide presentation in batches, ensuring context continuity and factual grounding in the source PDF. It uses a combination of TF-IDF embeddings, FAISS for retrieval, and the Groq API (`gemma2-9b-it` options for downloading JSON and PPT filescontinuity of contextmodel) for slide generation. A FastAPI backend handles PDF uploads and slide generation, while a Streamlit UI provides a user-friendly interface with JSON and PPT download options.

### Key Components

1. **Reference Document Handling**:

   - PDF text is extracted using `PyPDF2`.
   - The document is split into chunks using `RecursiveCharacterTextSplitter`, tagged by ALTAI topics (e.g., "Human Agency and Oversight").

2. **Context Retention Strategy**:

   - **State Management**: A JSON state file tracks the prompt, generated slides, section summaries, and used chunks.
   - **Retrieval-Augmented Generation**: TF-IDF embeddings and FAISS index relevant chunks for each slide, ensuring context is grounded in the PDF.
   - **Section Summaries**: Summaries of previous slides in each section are passed to the LLM to maintain narrative continuity.

3. **Chunked Presentation Generation**:

   - Slides are generated in batches of 5 per topic, covering 8 ALTAI topics (Introduction, Accountability, etc.).
   - Each batch builds on previous slides, with state saved between batches.

4. **Accuracy and Flow Assurance**:

   - A validation step checks if generated content aligns with the document context (overlap ratio ≥ 0.3).
   - Fallback content is provided for critical slides if the LLM fails.

5. **Output Format**:

   - Slides are output in JSON format: `{"title": "...", "content": "..."}`.
   - A PPT export option is available using `python-pptx`.

### API and UI

- **FastAPI Backend**:
  - `/upload_pdf`: Uploads a PDF and returns a session ID.
  - `/generate_slides`: Generates slides based on the query and session ID.
  - `/get_slides/{session_id}`: Retrieves generated slides.
  - `/health`: Health check endpoint.
- **Streamlit UI**:
  - Allows PDF upload, query input, slide generation, and download as JSON or PPT.

### Trade-offs and Scalability

- **Trade-offs**:
  - TF-IDF embeddings are used for simplicity, but dense embeddings (e.g., BERT) could improve retrieval accuracy at the cost of compute.
  - The Groq API is rate-limited, so retries and delays are implemented.
- **Scalability**:
  - Use a database (e.g., SQLite) for state management in production.
  - Implement caching for FAISS indices to speed up retrieval.
  - Parallelize slide generation for faster processing.

## Setup Instructions

### Prerequisites

- Python 3.8.16 (or compatible version)
- Anaconda (recommended for dependency management)
- Groq API key (free tier)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/presentation-system.git
   cd presentation-system
   ```

2. Create and activate a Conda environment:

   ```bash
   conda create -n presentation_env python=3.8.16
   conda activate presentation_env
   ```

3. Install dependencies:

   ```bash
   conda install -c conda-forge fastapi uvicorn streamlit requests pyPDF2 langchain faiss-cpu scikit-learn numpy groq python-dotenv python-pptx
   ```

4. Set up the Groq API key:

   - Create a `.env` file in the project root:

     ```bash
     GROQ_API_KEY=your_groq_api_key_here
     ```

   - Replace `your_groq_api_key_here` with your Groq API key.

### Running the Application

1. Start the FastAPI server:

   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

2. In a separate terminal, start the Streamlit UI:

   ```bash
   streamlit run streamlit_app.py
   ```

3. Open your browser and navigate to `http://localhost:8501`.

### Usage

1. Upload a PDF (e.g., the provided ALTAI PDF).
2. Enter a query (e.g., "Generate a presentation about the Assessment List for Trustworthy AI (ALTAI) for a general audience.").
3. Click "Generate Slides" and wait for the process to complete.
4. View the slides and download as JSON or PPT.

## Sample Demonstration

### Initial Prompt

"Generate a presentation about the Assessment List for Trustworthy AI (ALTAI) for a general audience."

### Reference Document

The ALTAI PDF: `Documents/altai_final_14072020_cs_accessible2_jsd5pdf_correct-title_3AC24743-DE11-0B7C-7C891D1484944E0A_68342.pdf`

### Generated Output (First Two Batches)

Below are the first 10 slides (two batches), showing context continuity:

#### Batch 1 (Introduction)

- **Slide 1**: {"title": "What is ALTAI?", "content": "ALTAI is a self-assessment tool developed by the High-Level Expert Group to ensure AI systems are trustworthy, based on seven key requirements."}
- **Slide 2**: {"title": "Purpose of ALTAI", "content": "ALTAI provides a practical framework for organizations to assess AI systems’ alignment with ethical principles and fundamental rights."}
- **Slide 3**: {"title": "Key Objectives", "content": "ALTAI aims to provide a self-assessment tool to ensure AI systems are trustworthy, focusing on seven key requirements: human agency, robustness, privacy, transparency, fairness, well-being, and accountability."}
- **Slide 4**: {"title": "Development Process", "content": "ALTAI was developed by the High-Level Expert Group on AI, involving extensive stakeholder consultation to address ethical concerns."}
- **Slide 5**: {"title": "Fundamental Rights Focus", "content": "ALTAI ensures AI systems respect fundamental rights, such as privacy, fairness, and non-discrimination, as defined by EU guidelines."}

#### Batch 2 (Human Agency and Oversight)

- **Slide 6**: {"title": "Human Agency and Autonomy", "content": "ALTAI ensures AI systems support human autonomy by allowing users to make informed decisions without undue influence."}
- **Slide 7**: {"title": "Human Oversight Mechanisms", "content": "ALTAI requires human-in-the-loop mechanisms, enabling oversight through monitoring and intervention when necessary."}
- **Slide 8**: {"title": "Preventing Over-Reliance", "content": "ALTAI addresses over-reliance on AI by promoting user training and awareness of system limitations."}
- **Slide 9**: {"title": "User Awareness", "content": "ALTAI mandates clear communication to users about AI interactions, ensuring they understand system capabilities."}
- **Slide 10**: {"title": "Social Interaction Risks", "content": "ALTAI assesses risks of AI affecting social interactions, ensuring systems do not undermine human relationships."}

### Context Continuity

- The Introduction section sets the foundation by explaining ALTAI’s purpose and objectives.
- The Human Agency and Oversight section builds on this by detailing how ALTAI ensures human control, maintaining narrative flow.

## Future Improvements

- Use dense embeddings (e.g., BERT) for better chunk retrieval.
- Add a slide preview feature in the UI with HTML/CSS rendering.
- Implement batch processing optimizations to reduce generation time.
- Enhance PPT formatting with bullet points, colors, and custom layouts.