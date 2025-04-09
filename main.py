# Step 1: Import necessary libraries
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Step 2: Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Step 3: Define the text generation function
def generate_text(prompt, max_length=100, num_return_sequences=1):
    # Encode the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate text
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Step 4: User prompt
user_prompt = "Once upon a time in a distant galaxy,"

# Step 5: Generate text
generated_paragraph = generate_text(user_prompt)
print(generated_paragraph)