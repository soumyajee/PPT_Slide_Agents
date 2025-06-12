# Architecture Walkthrough

## Thought Process
The goal was to build a system that generates a 40-slide presentation while maintaining context across sequential generation steps. The main challenge was context degradation, where LLMs lose track of prior content. To address this:

1. **Document Processing**:
   - Used `PyPDF2` to extract text from the PDF and `RecursiveCharacterTextSplitter` to create manageable chunks.
   - Tagged chunks by ALTAI topics to ensure relevant content retrieval.

2. **Context Retention**:
   - Implemented state management to save the prompt, slides, section summaries, and used chunks between batches.
   - Used FAISS with TF-IDF embeddings to retrieve relevant chunks for each slide, ensuring the LLM has the right context.
   - Passed section summaries and previous slide summaries to the LLM to maintain narrative flow.

3. **Generation Strategy**:
   - Generated slides in batches of 5 to manage API rate limits and ensure state persistence.
   - Validated each slide’s content against the document to prevent hallucinations.

4. **User Interface**:
   - Built a FastAPI backend for API-driven slide generation.
   - Added a Streamlit UI for user interaction, with JSON and PPT download options.

## Trade-offs Considered
- **TF-IDF vs. Dense Embeddings**:
  - Chose TF-IDF for simplicity and lower compute requirements, but dense embeddings (e.g., BERT) could improve retrieval accuracy.
- **Local State vs. Database**:
  - Used JSON files for state management due to the prototype nature of the task. A database (e.g., SQLite) would be better for production.
- **Groq API**:
  - Used the free-tier Groq API (`gemma2-9b-it`), which introduced rate limits. Added delays and retries to handle this, but a local LLM (e.g., via Hugging Face) could avoid such constraints.

## Scalability and Improvements
- **Database Integration**:
  - Replace JSON state files with a database to handle multiple users and large-scale deployments.
- **Parallel Processing**:
  - Parallelize slide generation within batches to reduce total generation time.
- **Caching**:
  - Cache FAISS indices and LLM responses to improve performance for repeated queries.
- **Enhanced UI**:
  - Add a live preview of slides in the Streamlit UI using HTML/CSS.
  - Improve PPT formatting with bullet points, themes, and images.
- **Error Handling**:
  - Add more robust error handling for edge cases, such as corrupted PDFs or network failures.

This architecture balances the task’s requirements with practical constraints, providing a functional prototype that can be scaled for production use.