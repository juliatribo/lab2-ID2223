import gradio as gr
import torch
from threading import Thread
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, TextStreamer, TextIteratorStreamer
from datetime import datetime
import requests
import os

model_name = "arejula27/lora_model"


url = "https://api.themoviedb.org/3/movie/popular?language=en-US&page=1"

token = os.environ.get("TOKEN")

headers = {
    "accept": "application/json",
    "Authorization": token
}

response = requests.get(url, headers=headers)
results = response.json()["results"]
# create a string with the list of movie titles and overview
formatted_data = "\n".join(
    [f"{result['title']}: {result['overview']}" for result in results])

# Model's parameters
max_seq_length = 2048
dtype = None
load_in_4bit = True
system_message = f"""
You are a movie recommender assistant. Your task is to suggest movies to users based on their preferences. Start by presenting the following list of popular movies to the user as a reference:

 {formatted_data}
Then, ask the user for more information about what they are in the mood for. You can inquire about specific genres (e.g., action, comedy, drama), moods (e.g., uplifting, suspenseful), or themes (e.g., time travel, crime). Alternatively, ask the user if they have a favorite movie or actor in mind, or if they would like you to suggest something similar to the listed films. Use their input to suggest personalized movie recommendations.
Example questions you can ask the user:

"Are any of these movies interesting to you?"
"What genre or type of movie are you in the mood for?"
"Do you have a favorite movie or actor?"
"Would you like a recommendation based on a movie you’ve enjoyed before?"
Tailor your recommendations based on the user's responses, and always try to suggest a variety of films to match different tastes or moods.
"""
max_tokens = 1024
temperature = 1.5
top_p = 0.95

# Load model and tokenizer from pretrained
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Enable native 2x faster inference
FastLanguageModel.for_inference(model)

# Create a text streamer
text_streamer = TextIteratorStreamer(
    tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
)


# Define inference function
def respond(message, history):
    global formatted_data, last_update_date, system_message

    # Add system message
    messages = [{"role": "system", "content": system_message}]

    # Include chat history
    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    # Lastly append user's message
    messages.append({"role": "user", "content": message})

    # Tokenize the input
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    # Generate arguments
    generate_kwargs = dict(
        input_ids=inputs,
        streamer=text_streamer,
        max_new_tokens=max_tokens,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    # Predict
    partial_message = ""
    for new_token in text_streamer:
        if new_token != "<":
            partial_message += new_token
            yield partial_message


# Define Gradio UI
gr = gr.ChatInterface(
    respond,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(
        placeholder="Ask me anything...", container=False, scale=7
    ),
    title="Lora Chatbot",
    description="chatbot like Chatgpt but worst :D",
    theme="soft",
    examples=[
        "Which are the most popular movies?",
        "Can you give me the overview of a popular movie?",
        "Can you suggest a movie?",
    ],
)

if __name__ == "__main__":
    gr.launch(debug=True)
