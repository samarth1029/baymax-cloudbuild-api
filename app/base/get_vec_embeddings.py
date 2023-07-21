from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd


def get_embeddings():
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    patient_queries = pd.read_csv(r"app/data/all_queries.csv")['0'].to_list()
    query_embeddings = model.encode(patient_queries, convert_to_tensor=True)
    return query_embeddings.numpy()
