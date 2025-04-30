from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM
from vectordb import VectorDB
from sentence_transformers import SentenceTransformer
from guardrails import detect_appropriate
import torch
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")
# embedding model
print("Initializing embedding transformer model...")
model_embed = SentenceTransformer("all-mpnet-base-v2", device = device)
print("Embedding transformer model initialized")

# base LLM model
# Note: run `huggingface-cli login` first in the terminal before using
print("Initializing LLM model...")
model_id = "google/gemma-3-1b-it"

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = Gemma3ForCausalLM.from_pretrained(
    model_id, quantization_config=quantization_config,
    attn_implementation="eager"
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_id)
print("LLM model initialized.")

# Our vector database data structure
print("Loading vector database...")
vectordb = VectorDB()
print("Vector database successfully retrieved.")


def prompt_format(query: str, retrieved_context: list) -> str:
    formatted_cntxt =  "- " + "\n- ".join(ctxt[0] for ctxt in retrieved_context)
    prompt = f"""Use the following pieces of context to answer the question at the end.
Pay attention to the context of the question rather than just looking for similar keywords in the corpus.
If the context doesn't provide enough information, just say that you don't know, don't try to make up an answer.
Use five sentences maximum and keep the answer as concise as possible.
{formatted_cntxt}
Question: {query}
Answer:
"""
    return prompt


def output_clean(input_text: str, answer: str) -> str:
    txt_to_remove = ["<bos>", "<eos>", "<start_of_turn>", "<end_of_turn>"]
    for rmv_txt in txt_to_remove:
        answer = answer.replace(rmv_txt, "")
    answer = answer.strip("user\n").replace(input_text, "").replace("model\n", "")
    return answer


def ask(input_text: str, doc_id: int = None, print_answer = True, k: int = 5, guard: bool = False) -> tuple:
    if guard and not detect_appropriate(input_text):
        answer = "Sorry, I cannot provide an answer because the question contains inappropriate topics."
        print(f"Answer:\n{answer}")
        return (answer, None)

    embedded_text = model_embed.encode(input_text, convert_to_tensor = True)
    retrieved = []
    if doc_id is not None:
        retrieved = vectordb.get_topk_similar(embedded_text, k, doc_id)
    else:
        retrieved = vectordb.get_topk_similar(embedded_text, k)
    input_text = prompt_format(input_text, retrieved)
    messages = [
    [
        {
            "role": "user",
            "content": [{"type": "text", "text": input_text},]
        },
    ],
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    
    with torch.inference_mode():
        outputs = model.generate(**inputs, 
                                 max_new_tokens=512,
                                 temperature=0.5)
    outputs = tokenizer.batch_decode(outputs)

    answer = output_clean(input_text, outputs[0])

    source = "Relevant source(s) in order:\n"
    included_id = set()
    for id, _ in retrieved:
        metadata = vectordb.get_metadata(id)
        if metadata[0] not in included_id:
            source += f"Document ID: {metadata[0]}, URL: {metadata[1]}\n"
            included_id.add(metadata[0])
    if print_answer:
        print(f"Answer:\n{answer}")
        print(source)
    return (answer, included_id)


def evaluate_answer():
    df = pd.read_csv("./dataset/single_passage_answer_questions.csv")
    df["llm_ans"] = pd.Series()
    # df["similarity"] = pd.Series()
    for i in range(len(df)):
        if i % 5 == 0:
            print(f"Done answering {i} questions")
        query, doc_id = df.iloc[i]["question"], df.iloc[i]["document_index"]
        llm_ans, _ = ask(query, doc_id, False)
        df.at[i, "llm_ans"] = llm_ans

        # Calculate dot product similarity
        # actual_ans = df.iloc[i]["answer"]
        # vec_actual = model_embed.encode(actual_ans, convert_to_tensor = True)
        # vec_llm = model_embed.encode(llm_ans, convert_to_tensor = True)
        # df.at[i, "similarity"] = (torch.dot(vec_actual, vec_llm) / \
        #             (torch.linalg.vector_norm(vec_actual) * torch.linalg.vector_norm(vec_llm))).item()
    df.to_csv("./results/single.csv", index = False)

    df = pd.read_csv("./dataset/multi_passage_answer_questions.csv")
    df["llm_ans"] = pd.Series()
    # df["similarity"] = pd.Series()
    for i in range(len(df)):
        query, doc_id = df.iloc[i]["question"], df.iloc[i]["document_index"]
        llm_ans, _ = ask(query, doc_id, False)
        df.at[i, "llm_ans"] = llm_ans

        # Calculate dot product similarity
        # actual_ans = df.iloc[i]["answer"]
        # vec_actual = model_embed.encode(actual_ans, convert_to_tensor = True)
        # vec_llm = model_embed.encode(llm_ans, convert_to_tensor = True)
        # df.at[i, "similarity"] = (torch.dot(vec_actual, vec_llm) / \
        #             (torch.linalg.vector_norm(vec_actual) * torch.linalg.vector_norm(vec_llm))).item()
        if i % 5 == 0:
            print(f"Done answering {i} questions")
    df.to_csv("./results/multi.csv", index = False)

    df = pd.read_csv("./dataset/no_answer_questions.csv")
    df["llm_ans"] = pd.Series()
    for i in range(len(df)):
        if i % 5 == 0:
            print(f"Done answering {i} questions")
        query, doc_id = df.iloc[i]["question"], df.iloc[i]["document_index"]
        llm_ans, _ = ask(query, doc_id, False)
        df.at[i, "llm_ans"] = llm_ans
    df.to_csv("./results/no.csv", index = False) 


def main():
    input_text = input("Enter the question here: ")
    ask(input_text, guard = True)


if __name__ == "__main__":
    # main()
    evaluate_answer()