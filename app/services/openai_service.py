import requests
import json
import os
import torch
from transformers import AutoTokenizer, AutoModel, GPT2LMHeadModel, GPT2Tokenizer
#from openai import OpenAI

# OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
# openai_client = OpenAI(api_key=OPENAI_API_KEY)

#OPENAI_EMBEDDING_MODEL = 'text-embedding-ada-002'
PROMPT_LIMIT = 3750
#CHATGPT_MODEL = 'gpt-4-1106-preview'

tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
model = AutoModel.from_pretrained('facebook/contriever')
tokenizer_gpt = GPT2Tokenizer.from_pretrained('gpt2')
model_gpt = GPT2LMHeadModel.from_pretrained('gpt2')

# def get_embedding(chunk):
#     url = 'https://api.openai.com/v1/embeddings'
#     headers = {
#         'Content-Type': 'application/json; charset=utf-8',
#         'Authorization': f"Bearer {OPENAI_API_KEY}"
#     }
#     data = {
#         'model': OPENAI_EMBEDDING_MODEL,
#         'input': chunk
#     }
    
#     response = requests.post(url, headers=headers, data=json.dumps(data))
#     print(response)
#     response_json = response.json()
    
#     # Print the entire response for debugging
#     print("Response JSON:", response_json)
    
#     try:
#         embedding = response_json["data"][0]["embedding"]
#         return embedding
#     except KeyError as e:
#         print(f"KeyError: {e}")
#         print("Response JSON:", response_json)
#         raise
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         print("Response JSON:", response_json)
#         raise

def get_embedding(chunk):
    # Tokenize the input text
    inputs = tokenizer(chunk, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling to get sentence embeddings
    def mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
    # print(embeddings)
    # Convert the tensor to a list
    return embeddings.squeeze().tolist()

# def get_llm_answer(prompt):
#   messages = [{"role": "system", "content": "You are a helpful assistant."}]
#   messages.append({"role": "user", "content": prompt})

#   url = 'https://api.openai.com/v1/chat/completions'
#   headers = {
#       'content-type': 'application/json; charset=utf-8',
#       'Authorization': f"Bearer {OPENAI_API_KEY}"            
#   }
#   data = {
#       'model': CHATGPT_MODEL,
#       'messages': messages,
#       'temperature': 1, 
#       'max_tokens': 1000
#   }
#   response = requests.post(url, headers=headers, data=json.dumps(data))
#   response_json = response.json()
#   completion = response_json["choices"][0]["message"]["content"]
#   return completion

def get_llm_answer(prompt):
    # Tokenize the input text
    inputs = tokenizer_gpt.encode(prompt, return_tensors='pt')
    print(inputs)

    # Generate a response using the model
    outputs = model_gpt.generate(inputs, max_length=1000, temperature=1.0)
    print(outputs)

    # Decode the generated response
    completion = tokenizer_gpt.decode(outputs[0], skip_special_tokens=True)
    print(completion)
    return completion