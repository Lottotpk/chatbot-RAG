# Local RAG Chatbot from Scratch

A retrieval-augmented generation (RAG) chatbot that runs **fully locally** and is built without RAG frameworks: no LangChain, LlamaIndex, Ollama or hosted LLM APIs. Chunking, the vector store, retrieval, prompt assembly, guardrails and source citation are all written by hand, so each part of the pipeline can be inspected.

## Architecture

```
documents ──► sentence-aware chunker ──► all-mpnet-base-v2 (768-d) ──► VectorDB (PyTorch tensors, .pt files)
                                                                            │
user question ──► Llama Guard 3 (1B) ──► embed ──► cosine top-k (k=5) ◄─────┘
                    │ unsafe → refuse                     │
                                                          ▼
                                          prompt with retrieved context
                                                          │
                                          Gemma 3 1B-it (8-bit quantised)
                                                          │
                                          answer + cited document IDs / URLs
```

| Component | Implementation |
|---|---|
| Chunking (`storing_data.py`) | NLTK sentence splitting packed into ~128-character chunks with 16-character overlap. Sentences are never cut mid-way. |
| Vector store (`vectordb.py`) | Custom class: tensors and metadata (document ID, source URL) saved with `torch.save`. Cosine-similarity top-k, optionally filtered to one document. |
| Generator (`main.py`) | `google/gemma-3-1b-it` with 8-bit `bitsandbytes` quantisation. The prompt tells the model to answer only from context, say "I don't know" otherwise, and stay within five sentences. |
| Guardrail (`guardrails.py`) | `meta-llama/Llama-Guard-3-1B` screens each question and refuses unsafe ones before retrieval |
| Citations | Every answer lists the source documents and URLs of the retrieved chunks |

## Evaluation

The chatbot was run on three question sets over a 20-document corpus (the dataset is not included in this repo):

| Set | Questions | What it tests | Outputs |
|---|---|---|---|
| Single-passage | 40 | Answer sits in one chunk | `results/single.csv` |
| Multi-passage | 40 | Answer must be combined from several chunks | `results/multi.csv` |
| Unanswerable | 40 | Model should abstain rather than hallucinate | `results/no.csv` |

Each CSV contains the question, the reference answer (where one exists) and the model's answer.

> Next step: add automatic scoring (e.g. LLM-as-judge correctness, and the abstention rate on the unanswerable set) and report the numbers here.

## Setup

Requires Python 3.12 and a Hugging Face account with access to [Gemma 3](https://huggingface.co/google/gemma-3-1b-it) and [Llama Guard 3](https://huggingface.co/meta-llama/Llama-Guard-3-1B). A CUDA GPU is recommended.

```bash
# GPU
pip install -r pytorch_cuda_requirements.txt
# or CPU
pip install -r pytorch_cpu_requirements.txt

pip install -r requirements.txt
huggingface-cli login
```

| Task | Command |
|---|---|
| Build the vector store from `dataset/documents.csv` | `python storing_data.py` |
| Chat (single question, guardrail on) | in `main.py`, call `main()` instead of `evaluate_answer()`, then run `python main.py` |
| Regenerate evaluation outputs | `python main.py` |

## Limitations
- One question per turn, with no conversation memory.
- The 1B-parameter model was chosen to fit a consumer GPU. Larger models would answer multi-passage questions better.

## Tech stack
Python · PyTorch · Hugging Face Transformers · Sentence-Transformers · bitsandbytes · NLTK · pandas

## License
MIT
