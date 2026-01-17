This captures the alternate Idarati API scraping flow that intercepts procedure IDs via Playwright and then fetches full details with retries (`src/idarati_collect_all_procedures_ids.py` + `src/idarati_collect_data_of_each_procedure.py`).

```mermaid
flowchart TD
    A[Playwright: open thematic page] --> B[click sub-thematic card]
    B --> C[intercept /procedures JSON response]
    C --> D[collect procedure IDs]
    D --> E[write data/raw/idarati_procedures_intercepted.json]

    E --> F[load IDs]
    F --> G[for each id]
    G --> H[fetch main/documents/legal/durations]
    H --> I[retry on error or 429]
    I --> J[append to results]
    J --> K[periodic autosave]
    K --> L[write data/raw/idarati_full_details.json]
```
