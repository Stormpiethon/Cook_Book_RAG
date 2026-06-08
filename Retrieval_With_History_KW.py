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
from dotenv import load_dotenv


# %%
load_dotenv()


# %%
client = oai(api_key=os.environ.get('OPENAI_API_KEY'))

# %%
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
index = pc.Index('recipes')


# %%
index.describe_index_stats()

# %%
def ask_gpt_completions(system_prompt, user_prompt, model='gpt-3.5-turbo', temp=0.5):
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
def get_context(query, embed_model="text-embedding-3-small", k=3, index=index):
  query_embeddings=get_embeddings_openai(query, model=embed_model)
  pinecone_response = index.query(vector=query_embeddings, top_k=k, include_metadata=True)
  contexts=[item['metadata']['text'] for item in pinecone_response['matches'] if item['score']>0.35]
  return contexts, query

# %%
def augmented_query(user_query, chat_history, embed_model='text-embedding-3-small', k=5):
  contexts, query = get_context(user_query, embed_model, k=k)

  history = ""
  for message in chat_history[-4:]:
    history+=f"{message['role'].capitalize()}: \n{message['content']}\n\n"
  return f"Chat History:\n{history}\n\n--------------------\n\nContext:\n" + "\n\n--------------------\n\n".join(contexts) + "\n\n--------------------\n\nUser Query:\n\n" + query
  #return "Context:\n"+"\n\n--------------------\n\n".join(contexts)+"\n\n -------------------\n\n User Query:\n\n"+query

# %%
get_context("How to make pancakes?")

# %%
primer = f"You are a cooking assistant AI chat bot. A system trained for the sole purpose of providing cooking recipes based on user questions or requests. You can only solely answer and provide recipes using the VectorDB that was fed to your architecture during data ingestion. If you don't know the answer, you will say 'I can only provide answers regarding cooking recipes.'"

# %%
def ask_RAG_completion(Query, chat_history, primer=primer):
  embed_model = 'text-embedding-3-small'
  llm_model = 'gpt-4.1'
  user_prompt = augmented_query(Query, chat_history, embed_model)
  response = ask_gpt_completions(primer, user_prompt, model =llm_model)

  lines = response[0]
  return "\n".join(textwrap.TextWrapper(width=120, break_long_words=False).wrap(lines))

# %% [markdown]
# ### Context History test

# %%
# Create a fresh code cell and run this:
test_contexts, test_query = get_context("How to make pancakes?")

print(f"Total recipes retrieved: {len(test_contexts)}")
print("\n--- First Recipe Preview ---")
if test_contexts:
    print(test_contexts[0][:300]) # Prints the first 300 characters of the first recipe
else:
    print("Warning: No recipes found! Check your Pinecone index data.")

# %%
# Create a test history array with 5 items (to test the 4-turn constraint)
mock_history = [
    {"role": "user", "content": "Hi, I am looking for breakfast ideas."},
    {"role": "assistant", "content": "I can help with that! What are you craving?"},
    {"role": "user", "content": "Something sweet."},
    {"role": "assistant", "content": "How about homemade pancakes or waffles?"},
    {"role": "user", "content": "Pancakes sound great. Do you have a recipe?"}
]

# Run your prompt compilation
compiled_prompt = augmented_query("Do you have a recipe?", mock_history)
print(compiled_prompt)

# %%
# Step A: Initialize an empty list just like the real UI will
live_history = []

# Step B: Turn 1 - Valid Culinary Query
turn_1_query = "Give me a potato recipe."
turn_1_reply = ask_RAG_completion(turn_1_query, live_history)

print("--- TURN 1 RESPONSE ---")
print(turn_1_reply)

# Append Turn 1 to the history scrapbook
live_history.append({"role": "user", "content": turn_1_query})
live_history.append({"role": "assistant", "content": turn_1_reply})


# Step C: Turn 2 - Follow-up dependent on context memory
turn_2_query = "What if I don't have butter?"
turn_2_reply = ask_RAG_completion(turn_2_query, live_history)

print("\n--- TURN 2 RESPONSE ---")
print(turn_2_reply)


# Step D: Turn 3 - Guardrail Test (Out-of-Scope)
turn_3_query = "Can you help me write a python script for a calculator?"
turn_3_reply = ask_RAG_completion(turn_3_query, live_history)

print("\n--- TURN 3 GUARDRAIL RESPONSE ---")
print(turn_3_reply)

# %%
ask_RAG_completion("Give me a waffle recipe.", [])


