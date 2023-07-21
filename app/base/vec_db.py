import string
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pinecone
from sklearn.metrics.pairwise import cosine_similarity
from base.gpt2lm import get_model_path
from base.get_vec_embeddings import get_embeddings


def load_chatbot_model(path, device="cuda"):
    tokenizer = GPT2Tokenizer.from_pretrained(path)
    model = GPT2LMHeadModel.from_pretrained(path).to(device)
    return model, tokenizer


def generate_embedding(model, tokenizer, input_text, device="cuda"):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    last_hidden_state = outputs.hidden_states[-1].squeeze(0)
    embeddings = last_hidden_state.mean(dim=0)
    return embeddings.tolist()


def send_embeddings_to_pinecone(pinecone_index, response_embeddings):
    pinecone_index.upsert(ids=[str(i) for i in range(len(response_embeddings))], vectors=response_embeddings)


def query_pinecone(pinecone_index, user_query_embedding):
    return pinecone_index.query(queries=[user_query_embedding], top_k=3)


def retrieve(pinecone_responses, all_responses):
    similar_response_indices = pinecone_responses[1][0]
    return [all_responses[idx] for idx in similar_response_indices]


if __name__ == "__main__":
    import pandas as pd
    path = get_model_path()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_chatbot_model(path, device)

    pinecone.init(api_key="YOUR_PINECONE_API_KEY")
    pinecone_index = pinecone.Index(index_name="your_index_name")

    all_responses = pd.read_csv(r"app/data/all_queries.csv")['0'].to_list()
    response_embeddings = [generate_embedding(model, tokenizer, response, device) for response in
                           all_responses]
    send_embeddings_to_pinecone(pinecone_index, response_embeddings)
    print("Chatbot: Hi, how can I assist you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye!")
            break

        user_query_embedding = generate_embedding(model, tokenizer, user_input, device)
        pinecone_responses = query_pinecone(pinecone_index, user_query_embedding)
        similar_responses = retrieve(pinecone_responses, all_responses)
        for response in similar_responses:
            print("Chatbot:", response)
