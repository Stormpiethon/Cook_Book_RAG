# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
import tf_keras as keras
from pinecone import Pinecone
from openai import OpenAI as oai
import textwrap


# API keys injected via environment variables (e.g. Docker -e flags)


# %%
client = oai(api_key=os.environ.get('OPENAI_API_KEY'))

# %%
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
index = pc.Index('recipes')


# %%
index.describe_index_stats()

# %%
def ask_gpt_completions(system_prompt, user_prompt, model='gpt-3.5-turbo', temp=0.9):
  temperature = temp
  completion = client.chat.completions.create(
      model=model,
      temperature=temperature,
      messages =[
        {"role":"system",
         "content":system_prompt},
        {"role": "user",
         "content": user_prompt}])
  return completion.choices[0].message.content, completion

# %%
def ask_gpt_response(system_prompt, user_prompt, model='gpt-5-chat-latest'):
  response = client.responses.create(
      model=model,
      input=[
          {"role":"developer",
          "content":system_prompt},
          {"role":"user",
           "content":user_prompt}])
  return response.output_text, response

# %%
def get_embeddings_openai(text, model="text-embedding-3-small"):
  text = text.replace("\n"," ")
  return client.embeddings.create(input=text, model=model).data[0].embedding

# %%
def get_context(query, embed_model="text-embedding-3-small", k=5, index=index):
  query_embeddings=get_embeddings_openai(query, model=embed_model)
  pinecone_response = index.query(vector=query_embeddings, top_k=k, include_metadata=True)
  contexts=[item['metadata']['text'] for item in pinecone_response['matches']]
  return contexts, query

# %%
def augmented_query(user_query, embed_model='text-embedding-3-small', k=5):
  contexts, query = get_context(user_query, embed_model, k=k)
  return "Context:\n"+"\n\n--------------------\n\n".join(contexts)+"\n\n -------------------\n\n User Query:\n\n"+query

# %%
get_context("How to make pancakes?")

# %%
primer = f"You are a cooking assistant AI chat bot. A system trained for the sole purpose of providing cooking recipes based on user questions or requests. You can only solely answer and provide recipes using the VectorDB that was fed to your architecture during data ingestion. If you don't know the answer, you will say 'I can only provide answers regarding cooking recipes.'"

# %%
def ask_RAG_completion(Query):
  embed_model = 'text-embedding-3-small'
  primer = f"You are a cooking assistant AI chat bot. A system trained for the sole purpose of providing cooking recipes based on user questions or requests. You can only solely answer and provide recipes using the VectorDB that was fed to your architecture during data ingestion. If you don't know the answer, you will say 'I can only provide answers regarding cooking recipes.'"
  llm_model = 'gpt-4.1'
  user_prompt = augmented_query(Query, embed_model)
  response = ask_gpt_completions(primer, user_prompt, model =llm_model)

  lines = response[0]
  return "\n".join(textwrap.TextWrapper(width=120, break_long_words=False).wrap(lines))

# %%
ask_RAG_completion("How do I make pancakes?")

# %%



