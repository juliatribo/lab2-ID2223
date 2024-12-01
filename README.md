# lab2-ID2223

In this project, we have fine-tuned a model using LoRA to create a movie recommender assistant in HuggingFace.

We tried to improve the model's performance with the following approaches:
- Model centric approach:
  - First of all, we experimented with several pre-trained models, including Meta-Llama-3.1-8B-bnb-4bit, Mistral-Small-Instruct-2409, Phi-3.5-mini-instruct, Llama-3.2-3B-Instruct-bnb-4bit, and Llama-3.2-1B-Instruct. We observed that larger models tended to produce more accurate results but also required longer execution times. Given our time constraints for training, we opted for the smallest model.
  - Additionally, we tuned various hyperparameters, including the maximum number of steps and the number of epochs. As expected, increasing the steps or epochs resulted in a lower error, but due to time constraints, we limited the training to 8000 steps.
  - Finally, we adjusted the batch size and found that increasing it improved execution speed but also increased GPU usage. Given the limited GPU resources, we decided to reduce the batch size to its original value of 2.

- Data-centric approach
  - We used the FineTome which is a subset of the `arcee-ai/The-Tome`, a curated dataset for training LLM focused on instruction following.
  - Then we used custom tools like get_chat_template to support various model formats and standardize_sharegpt to convert ShareGPT-style datasets into a generic format for easier training.
  - Finally, we designed a custom prompt to indicate the expected behaviour of our model. To do this, we provided not only instructions but also real-time data from an external API.

The model can be used in a [Huggingface space](https://huggingface.co/spaces/arejula27/lab2_id223)
