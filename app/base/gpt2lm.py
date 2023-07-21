import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_chatbot_model(path, device="cpu"):
    tokenizer = GPT2Tokenizer.from_pretrained(path)
    model = GPT2LMHeadModel.from_pretrained(path).to(device)
    return model, tokenizer


def generate_response(model, tokenizer, input_text, device="cpu"):
    prompt_input = "The conversation between user and Baymax.\n[|User|] {input}\n[|Baymax|]"

    sentence = prompt_input.format_map({'input': input_text})
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        beam_output = model.generate(**inputs, max_length=512, num_beams=3, repetition_penalty=1.2, early_stopping=True, eos_token_id=198, num_return_sequences=1)

        return tokenizer.decode(beam_output[0], skip_special_tokens=True)


def get_model_path():
    return "jianghc/medical_chatbot"
