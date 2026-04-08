Minimum requirements for this project:
        a. LLM Model (Google Gemini, Claude, groq)
        b. Embedding Model (openai, gemini, hf)
        c. Vector database (inMemory, ondisk, Cloudbase)



Day 1	Foundations & Environment Setup	Project Architecture & Dev Environment	LLMOps overview, production-grade architecture walkthrough, Python environment setup, Google API key configuration, project folder structure, dependency management	Set up the project repo, configure API keys, install all dependencies, make a test LLM call to validate the setup

Day 2	Production Readiness	Exception Handling & Logging Modules	Why production code needs structured error handling, custom exception classes, structured logging module, log levels, file & console handlers, integrating both into a reusable utility layer	Build the exception management module, implement the logging framework, test with deliberate failure scenarios to see both modules in action

Day 3	Data Ingestion Pipeline	Shared RAG Backbone	Document loading strategies (PDF, DOCX, TXT), chunking approaches & trade-offs (size, overlap), embedding generation via Google API, FAISS DB setup & indexing, similarity search validation	Build the end-to-end ingestion pipeline, ingest a sample corpus into FAISS, run retrieval queries, tune chunk parameters and observe impact on results

Day 4	RAG Project 1	Document Analyzer	Single-document analysis chain, prompt engineering for structured analysis, output formatting, integrating exception handling & logging into the RAG flow	Build the Document Analyzer from scratch using the shared pipeline, test across different document types, review structured outputs

Day 5	RAG Project 2	Document Comparer	Multi-document comparison logic, side-by-side analysis design, difference extraction & summarisation, handling edge cases (mismatched formats, varying lengths)	Build the Document Comparer, compare multiple document pairs, validate comparison accuracy

Day 6	RAG Projects 3 & 4 + Full-Stack	Single & Multi Doc Chat + FastAPI & Frontend	Single Document Chat (conversational retrieval), Multi Document Chat (cross-corpus Q&A), FastAPI endpoint design for all four projects, HTML/CSS frontend walkthrough, wiring UI to backend	Build both chat modules, expose all four projects via FastAPI, integrate the frontend, run and demo the complete app locally

Day 7	Containerisation & Cloud Deployment	Docker to AWS ECS Fargate	Dockerfile creation, building & testing the image locally, pushing to AWS ECR, ECS Fargate concepts (tasks, services, clusters), deploying the container, validating the live production endpoint	Write the Dockerfile, build & test locally, push to ECR, deploy on ECS Fargate, hit the production URL, end-to-end demo of the live solution
