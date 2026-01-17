This shows how batching is implemented in the RAG indexing pipeline (`src/idarati_pipeline/pipeline.py`) including resume, caching, and parallel embed+upload when optimization is enabled.

```mermaid
flowchart TD
    A[Start RAGPipeline.run] --> B[load_procedures]
    B --> C[compute total_batches]
    C --> D[load status file]
    D --> E{optimized_texts cache exists?}

    E -- yes --> F[embed+upload remaining batches]
    E -- no --> G{text_optimizer enabled?}

    G -- no --> H[optimize_texts (fallback)]
    H --> F

    G -- yes --> I[optimize + embed + upload in batches]

    subgraph BatchLoop[Batch Loop]
        I --> J[for batch in procedures]
        J --> K[optimize_batch]
        K --> L[submit _embed_and_upload_batch]
        L --> M[sleep BASE_DELAY]
        M --> J
    end

    subgraph UploadRemaining[Embed + Upload Remaining]
        F --> N[for batch in remaining]
        N --> O[_generate_embeddings_for_batch]
        O --> P[uploader.upload_batch]
        P --> Q[save status (last_completed_batch)]
        Q --> N
    end

    J --> R[save optimized cache]
    R --> S[summary: success/failure]
    Q --> S
```
