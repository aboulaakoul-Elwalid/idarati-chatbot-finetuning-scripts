This end-to-end flow stitches the Idarati scraping path with JSON-to-Markdown transformation and the embedding pipeline that batches and uploads vectors.

```mermaid
flowchart TD
    A[Start data collection] --> B{Scrape path}

    B -->|Frontend HTML| C[CLI: src/main.py --frontend-only]
    C --> D[collect_procedure_links]
    D --> E[snapshot_pages HTML]
    E --> F[parse_procedure_page]
    F --> G[data/processed/idarati_procedures.json]

    B -->|API intercept| H[Playwright: intercept /procedures]
    H --> I[data/raw/idarati_procedures_intercepted.json]
    I --> J[fetch procedure details JSON]
    J --> K[data/raw/idarati_full_details.json]

    G --> L[idarati_json_to_markdown.py]
    K --> L
    L --> M[Markdown files + ready_for_embedding.json]

    M --> N[Idarati embed pipeline]
    N --> O[optimize texts (optional)]
    O --> P[generate embeddings]
    P --> Q[batch upload]
    Q --> R[Chroma or Supabase]
```
