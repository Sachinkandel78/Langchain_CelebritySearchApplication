# Langchain Celebrity Search Application

Practice project for learning **LangChain** with the **Groq API** (fast open-weight LLM inference) and **Streamlit** for the UI.

The repo has two versions of the app, showing a natural progression from a simple LLM call to a multi-step chain with memory:

| File | What it demonstrates |
|---|---|
| `main.py` | Bare-bones: send user input straight to a Groq-hosted LLM and print the response. |
| `examples1.py` | A `SequentialChain` of three prompts, each feeding into the next, with `ConversationBufferMemory` attached to each step. |

## How it works

### `main.py` — single LLM call
1. Takes a celebrity name from a Streamlit text input.
2. Sends it directly to `ChatGroq` (model: `llama-3.1-8b-instant`).
3. Displays the raw response.

This is the "hello world" of LangChain + Groq — no chaining, no memory, just one prompt in, one completion out.

### `examples1.py` — chained prompts with memory
Three prompts run in sequence, each output becoming the next input:

1. **`first_input_prompt`** — `"Tell me about celebrity {name}."` → produces `person`
2. **`second_input_prompt`** — `"when was {person} born?"` → produces `dob`
3. **`third_input_prompt`** — `"Mention 5 major events happened around {dob} in the world."` → produces `description`

These are wired together with `SequentialChain`, and each step has its own `ConversationBufferMemory` so you can inspect the intermediate outputs (`person_memory`, `dob_memory`, `description_memory`) in the Streamlit UI via expandable sections.

This pattern is useful for learning:
- How `LLMChain` wraps a prompt + LLM + memory into one reusable unit
- How `SequentialChain` passes `output_key` from one chain into the `input_variables` of the next
- How memory lets you audit what each intermediate step produced

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

Note: `examples1.py` also imports `langchain_groq` and `langchain_classic`, which aren't currently listed in `requirements.txt`. You'll want to add them:
```bash
pip install langchain-groq langchain-classic
```

### 2. Add your Groq API key
**Don't hardcode it in `constants.py`.** Instead:

1. Create a `.env` file in the project root (already covered by `.gitignore`):
   ```
   GROQ_API_KEY=your_key_here
   ```
2. Load it with `python-dotenv` instead of importing a plaintext constant:
   ```python
   import os
   from dotenv import load_dotenv

   load_dotenv()
   groq_api_key = os.environ["GROQ_API_KEY"]
   ```
3. Get a free API key from [console.groq.com](https://console.groq.com/keys).

> If you've already committed a real key in `constants.py`, revoke it in the Groq console and generate a new one — treat it as compromised.

### 3. Run the app
```bash
streamlit run main.py
```
or
```bash
streamlit run examples1.py
```

## Requirements
- Python 3.9+
- A Groq API key (free tier available)
- See `requirements.txt` for Python packages

## Learning notes
- `langchain_classic.chains.LLMChain` and `SequentialChain` are from LangChain's legacy chain API; newer LangChain code tends to favor **LCEL** (`prompt | llm | parser` pipelines) or **LangGraph** for multi-step flows. Worth exploring as a next step once these examples feel comfortable.
- The `examples1.py` file ends with a comment about extending into RAG (retrieval-augmented generation) — a natural next practice project would be adding a vector store (e.g. FAISS/Chroma) so the celebrity chain can pull facts from real documents instead of relying purely on the model's parametric knowledge.

## License
Practice/learning project — no license specified.
