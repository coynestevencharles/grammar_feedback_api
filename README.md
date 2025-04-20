# Grammar Feedback API

Back end application that performs the following tasks:

- Splits the sentences in an input text.
- Performs grammatical error correction using an LLM.
- Uses ERRANT to extract edits from the sentence pair.
- Generates feedback for each edit.

Currently, there are two feedback systems supported, `rule-based` and `llm-based`:

- `rule-based` operates directly on the ERRANT edits, filling simple templates.
- `llm-based` first uses an LLM to "refine" the edits to address corner cases and break down multiple changes on a given word. Then, it generates feedback using annother LLM prompt.

## Launching

```bash
uvicorn application:app --host 0.0.0.0 --port 8000
```
