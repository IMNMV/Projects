#script to work with TwitterBot 
#based off of @Yohei's babyAGI project. 

#made with the help of GPT-4

import openai
import requests
import json
from pymongo import MongoClient
from collections import deque
from collections import deque

from typing import Dict, List
import numpy as np
import time
import re


# Set API Keys
OPENAI_API_KEY = "##"

with open('/Users/your_name/Desktop/apikeystorage/api_key', 'r') as f:
    api_key = f.read().strip()

# Set Variables

YOUR_DATABASE_NAME = "automation"
YOUR_COLLECTION_NAME = "Logic1"


selected_model = "3.5-turbo"


# Configure OpenAI
openai.api_key = OPENAI_API_KEY

# Connect to MongoDB
client = MongoClient('mongodb://localhost:##/')
db = client[YOUR_DATABASE_NAME]
collection = db[YOUR_COLLECTION_NAME]




def summarize_context(context, selected_model):
    prompt = f"You are an AI summarization bot. Please summarize version of the following text without removing too many pertinent details: {context}"
    
    if selected_model == "3":
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=500,
            n=1,
            stop=None,
            temperature=0.5,
        )
        summarized_context = response.choices[0].text.strip()
        return summarized_context
    else:
        model_name = f"gpt-{selected_model}" if selected_model == "3.5-turbo" else f"gpt-{selected_model}"
        url = 'https://api.openai.com/v1/chat/completions'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": f"Please provide a summarized version of the following text without removing too many pertinent details: {context}"},
                {"role": "user", "content": prompt}
            ]
        }
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        summarized_context = response.json()['choices'][0]['text'].strip()
        tokens_used = response.json()['usage']['total_tokens']
        #print(f"\033[91mTokens used for this API call: {tokens_used}\033[0m")
        return summarized_context



#count tokens

def count_tokens(text):
    tokens_used = len(text.split())
    return tokens_used



def get_embeddings(prompt):
    response = openai.Embedding.create(
        input=prompt,
        model="text-embedding-ada-002"
    )
    embeddings = response['data'][0]['embedding']
    return np.array(embeddings)

import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_relevant_context(prompt_embeddings, query_embeddings, threshold=0.8, max_tokens=500):
    conversations = collection.find({"embeddings": {"$exists": True}})
    
    if not conversations:
        return None

    relevant_context = []
    accumulated_tokens = 0
    total_tokens = 0  # Add a new variable to track total tokens in the MongoDB call

    for conversation in conversations:
        embeddings = np.array(conversation["embeddings"])
        if embeddings.size == 0:
            continue

        similarity = cosine_similarity(prompt_embeddings, embeddings)

        #print(f"Similarity: {similarity}")  # Debugging line



        current_entry_tokens = count_tokens(conversation["thought_name"])

        #print(f"Current entry tokens: {current_entry_tokens}")  # Debugging line
        #print(f"Accumulated tokens: {accumulated_tokens}")  # Debugging line

        if similarity > threshold and accumulated_tokens + current_entry_tokens <= max_tokens:
            relevant_context.append({"thought_name": conversation["thought_name"]})
            accumulated_tokens += current_entry_tokens

        total_tokens += current_entry_tokens  # Increment the total tokens count

        if total_tokens >= max_tokens:  # Check if the total tokens have reached the limit
            break

    return relevant_context





def execution_agent(objective: str, thought: str, prompt_embeddings, selected_model: str, context: List[str]) -> str:
    # Join multiple contexts with a newline character
    context_string = "\n".join([c["thought_name"] if isinstance(c, dict) else str(c) for c in context]) if context else ""

    print("\033[94m\n*******RELEVANT CONTEXT******\n\033[0m")
    if context_string:
        print(context_string)
    else:
        print("\033[91m\033[1mNONE FOUND. FREESTYLING THIS ONE CHIEF\033[0m\033[0m")
        
    prompt = f"Your thought: {thought}\nResponse:"


    if selected_model == "3":
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.5,
        )
        output_text = response.choices[0].text
    else:
        model_name = f"gpt-{selected_model}" if selected_model == "3.5-turbo" else f"gpt-{selected_model}"
        url = 'https://api.openai.com/v1/chat/completions'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        context_string = f"Context: {context}\n" if context is not None else ""
        payload = {
            "model": model_name,
            "messages": [
                #{"role": "system", "content": f"You are an AI who makes a plan of action with ONE thought based on the following objective: {objective}.\n{context_string}.\n{prompt}"},
                {"role": "system", "content": f"You are an AI built to detect logical fallacies and errors in logical thinking and correct them. ere are some previous thoughts you've had about the topic:\n{context_string}. Now, in a moment I will give you the response you will need to make your decision on. Explain why it is a logical fallacy, suggestions for the response so it not a logical fallacy, if it is not a logical fallacy but just an error in thinking, explain that. If there is no error print, Eh? No work here need be done. Here is the response to ruminate on and then give your response."},

                {"role": "user", "content": prompt}
            ]
        }
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        tokens_used = response.json()['usage']['total_tokens']
        print(f"\033[91mTokens used for this API call: {tokens_used}\033[0m")
        output_text = response.json()["choices"][0]["message"]["content"]
        #for i, thought in enumerate(thought_list, 1):
           #print(f"\033[92mthought {i}: {thought['thought_name']}\033[0m")

    output_text

    #print("execution agent output: " + output_text)
    
    return output_text, thought





def thought_creation_agent(objective: str, result: Dict, thought_description: str, thought_list: List[str], selected_model: str):
    prompt = f"You are a thought creation AI that uses the result of an execution agent to create new thoughts with the following input in mind: {objective}, The last completed thought has the result: {result}. This result was based on this thought description: {thought_description}. These are incomplete thoughts: {', '.join(thought_list)}. Based on the result, create new thoughts to be completed by the AI system that do not overlap with incomplete thoughts and build toward discerning if the input falls into any fallacies or lapses in logical thinking. Delete any thoughts that are out-dated because we have built upon their output. Return the thoughts as a numbered list, like:\n1. First thought\n2. Second thought\nStart the thought list with number 1."
    if selected_model == "3":
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0.5,
            max_tokens=200,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        output_text = response.choices[0].text
        #print(output_text)
    else:
        model_name = f"gpt-{selected_model}" if selected_model == "3.5-turbo" else f"gpt-{selected_model}"
        url = 'https://api.openai.com/v1/chat/completions'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        tokens_used = response.json()['usage']['total_tokens']
        print(f"\033[91mTokens used for this API call: {tokens_used}\033[0m")
        output_text = response.json()["choices"][0]["message"]["content"]
        #print("thought creation agent output: " + output_text)

        #print(output_text)

    output_text = output_text.strip()
    if '\n' in output_text:
        new_thoughts = output_text.split('\n')
    else:
        new_thoughts = [thought.strip() for thought in output_text.split('.') if thought.strip()]

    thought_id = len(thought_list) + 1
    return [{"thought_id": thought_id, "thought_name": thought_name, "is_complete": False} for thought_name in new_thoughts]




#def prioritization_agent(this_thought_id: int, selected_model: str):
    #thought_names = [t["thought_name"] for t in thought_list]
#def prioritization_agent(this_thought_id: int, selected_model: str, thought_list: list):
def prioritization_agent(this_thought_id: int, selected_model: str, thought_list: list, objective: str):

    thought_names = [t["thought_name"] for t in thought_list]   
    next_thought_id = int(this_thought_id) + 1
    prompt = f"""You are a thought prioritization AI thoughted with cleaning the formatting of and reprioritizing the following thoughts in a logical manner that build upon one another: {thought_names}. Consider your the objective when formatting, cleaning and building:{objective}. Do not remove any thoughts. Return your thoughts as a numbered list, like:
    {next_thought_id}| First thought
    {next_thought_id + 1}| Second thought"""

    
    if selected_model == "3":
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=200,
            n=1,
            stop=None,
            temperature=0.5,
        )
        tokens_used = response.choices[0].metadata['tokens']
        new_thoughts = response.choices[0].text.strip().split('\n')
        #print(new_thoughts)
    else:
        model_name = f"gpt-{selected_model}" if selected_model == "3.5-turbo" else f"gpt-{selected_model}"
        url = 'https://api.openai.com/v1/chat/completions'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        tokens_used = response.json()['usage']['total_tokens']
        print(f"\033[91mTokens used for this API call: {tokens_used}\033[0m")
        new_thoughts = response.json()["choices"][0]["message"]["content"].strip().split('\n')
        #print("new thoughts output: f{new_thoughts}")


    new_thought_list = []
    for thought_string in new_thoughts:
        thought_parts = thought_string.strip().split("|", 1)
        if len(thought_parts) == 2:
            thought_id = int(thought_parts[0].strip())
            thought_name = thought_parts[1].strip()
            new_thought_list.append({"thought_id": thought_id, "thought_name": thought_name})
            #print("new_thought_list: f{new_thought_list}")

    
    return new_thought_list


            
            #Initialize the thought list with the first thought
def final_execution_agent(objective, thought_list, prompt_embeddings, query_embeddings, selected_model):
    prefix = "This is the final thought based on the updated thought list and the objective. Take into account all the information from the previous iterations and generate a succinct final conclusion in no more than 150 characters so it fits in a tweet in a manner that Aristotle would speak. Do NOT use the @ in your response. Do not make a quirky response with a hastag like #Logic or #Criticalthinking - in fact never use the # at all : "
    objective = prefix + objective
    context = find_relevant_context(prompt_embeddings, query_embeddings)
    
    
    prompt = f"Your thoughts: {[t['thought_name'] for t in thought_list]}\nResponse:"
    
    if selected_model == "3":
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.5,
        )
        output_text = response.choices[0].text

    else:
        model_name = f"gpt-{selected_model}" if selected_model == "3.5-turbo" else f"gpt-{selected_model}"
        url = 'https://api.openai.com/v1/chat/completions'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": f"You are an AI who generates a final conclusion based on the following input: {objective}.\nTake into account these thoughts: {[t['thought_name'] for t in thought_list]} and provide a concise resposne noting any logical fallacies the original input falls into, and or any lapses in critical thinking yielding an incorrect answer. \nResponse:"},
                {"role": "user", "content": prompt}
            ]
        }
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        tokens_used = response.json()['usage']['total_tokens']
        output_text = response.json()["choices"][0]["message"]["content"]
        
        
        return output_text
    

    #output_text = error_agent(output_text.strip(), selected_model, thought_list)
    #output_text = execution_agent(prompt_embeddings, selected_model, thought_list)
   # output_text = execution_agent(objective, "", prompt_embeddings, selected_model)
#objective = input("Enter the objective: ")

def main(objective: str):
    
        
    #def main(objective: str):
        
    thought_list = deque([])
    
    # Add the user-defined objective to the thought_list
    thought_list.append({"thought_id": 1, "thought_name": objective})
    
    # Get embeddings for the user-defined objective
    prompt_embeddings = get_embeddings(objective)
    #print(prompt_embeddings)
    
    #convert thought names to an acceptable format
    thought_names = " ".join([t["thought_name"] for t in thought_list])
    #print(thought_names)
    query_embeddings = get_embeddings(thought_names)
    #print(query_embeddings)
    
    
    #iteration_count = 0
    #while thought_list and iteration_count < iterations:
     # Update the relevant context and query_embeddings for each iteration
    context = find_relevant_context(prompt_embeddings, query_embeddings)
    print(context)
    thought_names = " ".join([t["thought_name"] for t in thought_list])
    query_embeddings = get_embeddings(thought_names)
    
    thought = thought_list.popleft()
    print(f"\033[95m\033[1m\n*****CURRENT THOUGHTS*****\n{thought['thought_name']}\033[0m\033[0m")
     
     #create one thought to solve a thought
    result, executed_thought = execution_agent(objective, thought["thought_name"], context, selected_model, context)
    
     #update based on last thoughts completion
    new_thoughts = thought_creation_agent(objective, result, thought["thought_name"], [t["thought_name"] for t in thought_list], selected_model)
    
    
    print("\033[96m\033[1m\n*****THINKING INTENSELY, PLEASE WAIT*****\n\033[0m\033[0m", result)
     
    
    #save db     
    for new_thought in new_thoughts:
        if collection.find_one({"thought_name": new_thought["thought_name"]}) is None:
            new_thought_embeddings = get_embeddings(new_thought["thought_name"])
            collection.insert_one({"thought_name": new_thought["thought_name"], "result": "", "timestamp": time.time(), "embeddings": new_thought_embeddings.tolist()})
        else:
            collection.update_one({"thought_name": new_thought["thought_name"]}, {"$set": {"result": result, "timestamp": time.time(), "embeddings": prompt_embeddings.tolist()}})
     
             
     
     # Append newly made thoughts - might consider killing thoughts here
    for new_thought in new_thoughts:
        thought_list.append(new_thought)
        print(f"\033[92mNew thought: {new_thought['thought_name']}\033[0m")
     
         # Save the new thought to the MongoDB database
        if collection.find_one({"thought_name": new_thought["thought_name"]}) is None:
            collection.insert_one({"thought_name": new_thought["thought_name"], "result": "", "timestamp": time.time(), "embeddings": []})
     
         # Remove the old thought from the MongoDB database
         # collection.delete_one({"thought_name": thought["thought_name"]})
    
    #prioritized_thoughts = prioritization_agent(thought["thought_id"], selected_model)
    #prioritized_thoughts = prioritization_agent(thought["thought_id"], selected_model, thought_list)
    prioritized_thoughts = prioritization_agent(thought["thought_id"], selected_model, thought_list, objective)

     
    thought_list = deque(prioritized_thoughts)
    #iteration_count += 1
    
       
    
    # Print final result
    final_result = final_execution_agent(objective, list(thought_list), prompt_embeddings, query_embeddings, selected_model)
    print("\033[33m\033[1m""\n*****FINAL RESULT*****\n\033[0m\033[0m", final_result)
    
    return final_result

if __name__ == "__main__":
    objective = input("Enter the objective: ")
    main(objective) 
    

 

        
