# chatbot-RAG
The project about chatbot using Retrieval Augmented Generation (RAG) that runs **locally**. This project try to use as few bulit-in library function as possible (no LangChain, Ollama, OpenAI APIs used).

## Requirements
- Python version 3.12. You can check by running `python --version`
- A Huggingface account that has access to [gemma](https://huggingface.co/google/gemma-3-1b-it) and [Llama Guard](https://huggingface.co/meta-llama/Llama-Guard-3-1B).
- Nvidia [CUDA](https://developer.nvidia.com/cuda-zone) if you want to use GPU to run the script (faster, recommended).

## Files contained in this repository
- `./database` The folder contains the saved vector database. 
- `./results` The folder of the LLM's answer to the question in `./dataset` (not available here). You can see the results here in this folder to see the performance of the LLM.
- `vectordb.py` The vector database data structure to store the content in the documents.
- `storing_data.py` The script to load the documents, and convert them into vector database. Then, upload the vector database into `./database`.
- `guardrails.py` The guardrails to filter out the inappropriate question to the LLM.
- `main.py` The main function to ask the question with the LLM. Also, this script is used to produce the results in `./results`.

## To use this chatbot
1. If you want to use [CUDA](https://developer.nvidia.com/cuda-zone), run
```
pip install -r pytorch_cuda_requirements.txt
pip install -r requirements.txt
```
Otherwise, run
```
pip install -r pytorch_cpu_requirements.txt
pip install -r requirements.txt
```
2. Log in to the huggingface account by running
```
huggingface-cli login
```
You need to get the API key from the website in order to get access to huggingface and LLM models.

3. To chat with the chatbot, run
```
python main.py
```

## Final thoughts
- Note that this script only answer 1 question at a time and no follow-up questions feature has not implemented yet. 
- You can use a better LLM model because this repo only use 1B parameters pretrained model (not enough GPU to run any bigger model...).
