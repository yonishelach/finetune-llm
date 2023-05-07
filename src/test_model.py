import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def generate_response(model, tokenizer, prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Create the attention mask and pad token id
    attention_mask = torch.ones_like(input_ids)
    pad_token_id = tokenizer.eos_token_id

    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        attention_mask=attention_mask,
        pad_token_id=pad_token_id
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)


def load_my_model(model_name: str, tokenizer_name: str = None):
    tokenizer_name = tokenizer_name or model_name
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    prompt = "What is mlrun?"
    response = generate_response(model, tokenizer, prompt)
    print("Generated response:", response)
    # Test the chatbot
    prompt = "What is an MLRun function?"
    response = generate_response(model, tokenizer, prompt)
    print("Generated response:", response)
