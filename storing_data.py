import pandas as pd
import torch
import nltk
from sentence_transformers import SentenceTransformer
from vectordb import VectorDB
nltk.download("punkt_tab")

device = "cuda" if torch.cuda.is_available() else "cpu"
model_embed = SentenceTransformer("all-mpnet-base-v2", device = device)


def txt_chunk(txt: str, chunk_size: int = 128, chunk_overlap: int = 16) -> list:
    """Convert texts into list of multiple sentences (chunk).

    Parameters
    ----------
    txt : string
        The input texts to be convert into chunks
    chunk_size : int
        The minimum string length of each chunk
    chunk_overlap : int
        The overlap length between chunk

    Returns
    -------
    list
        The list of chunks obtained
    """
    chunk_list = []
    sen_list = nltk.sent_tokenize(txt)
    word_len = 0
    chunk = ""
    overlap = ""
    flag1 = True
    flag2 = True

    for word in sen_list:
        if flag1:
            chunk += word
            flag1 = False
        else:
            chunk += " " + word
            word_len += 1
        word_len += len(word)
        if word_len > chunk_size - chunk_overlap:
            if flag2:
                overlap += word
                flag2 = False
            else:
                overlap += " " + word
        if word_len > chunk_size:
            chunk_list.append(chunk)
            chunk = overlap + " "
            word_len = len(overlap) + 1
            overlap = ""
            flag1 = True
            flag2 = True
    if chunk != "":
        chunk_list.append(chunk)
    return chunk_list 


def chunk_embed(chunk_list: list) -> torch.Tensor:
    """Embed chunk into vector of certain dimensions (in this case, it is 768 dimensions).

    Parameters
    ----------
    chunk_list : list
        The list of chunks
    
    Returns
    -------
    torch.Tensor
        The list of vector (vector is in torch.Tensor data type instead of numpy)
    """
    return model_embed.encode(chunk_list, convert_to_tensor = True)


def query_retrieval(query: str, 
                    vectordb: VectorDB, 
                    model: SentenceTransformer = model_embed,
                    top_k: int = 5) -> list:
    """Retrieve the top k list of similar vector to the query.

    Parameters
    ----------
    query : str
        The input query from the user
    vectordb : VectorDB
        The vector database to search
    model : SentenceTransformer
        The transformer model to embed query into vector
    top_k : int
        The k number of returned list, default set to 5

    Returns
    -------
    list
        The top k-th most similar contexts to the query 
    """
    query_vector = model.encode(query, convert_to_tensor = True)
    return vectordb.get_topk_similar(query_vector, top_k)


def main():
    df = pd.read_csv("./dataset/documents.csv")
    vectordb = VectorDB()

    for i, txt in enumerate(df["text"]):
        chunk = txt_chunk(txt)
        embedded_vector = chunk_embed(chunk)
        for id, vec in zip(chunk, embedded_vector):
            print(f"adding {id[:30]}... to vector database")
            vectordb.add_vector(id, vec, (i, df.iloc[i]["source_url"]))
        print(f"Document {i} successfully embedded")
    
    print(f"Saving to json...")
    vectordb.save_to_json()
    print(f"Saving successful")


if __name__ == "__main__":
    main()