A high-level flow of the Idarati frontend scraping and parsing path executed by the CLI (`src/main.py`) when `--frontend-only` or `--merge` is used.

```mermaid
flowchart TD
    A[CLI: src/main.py] --> B[run_frontend_scrape]
    B --> C[collect_procedure_links(theme)]
    C --> D[Playwright: load theme page]
    D --> E[scroll to load cards]
    E --> F[extract /procedure links]
    F --> G[snapshot_pages(urls)]
    G --> H[Playwright: visit each procedure URL]
    H --> I[HTML snapshots list]
    I --> J[write_json -> data/raw/idarati_{theme}_pages.json]
    J --> K[run_parse_procedures]
    K --> L[parse_procedure_page(html, url)]
    L --> M[procedure dicts]
    M --> N[write_json -> data/processed/idarati_procedures.json]
```
