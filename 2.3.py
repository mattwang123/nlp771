import torch
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer, set_seed
import random

# Load DistilGPT2
print("Loading DistilGPT2...")
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

def compute_perplexity(text):
    """Compute perplexity of given text"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()

    return perplexity

def shuffle_words(text):
    """Shuffle words in the text"""
    words = text.split()
    random.shuffle(words)
    return ' '.join(words)

def generate_text(prompt, temperature, max_length=500):
    """Generate text with given temperature"""
    set_seed(42)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        if temperature == 0:
            # Greedy decoding
            outputs = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        else:
            # Temperature sampling
            outputs = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id
            )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Part (a): Perplexity Analysis
print("="*50)
print("PART (A): PERPLEXITY ANALYSIS")
print("="*50)

# Original paragraph
original_text = """In the 2000-01 season, the average NBA team attempted 13.7 3-pointers per game. In the 2024-25 campaign, that figure jumped to 37.6. No center averaged even four assists 25 years ago. Nikola Jokić just reached double digits. Hand-checking was allowed and zone defense was not. That has since flipped. Only one international player won an MVP award in the 20th century. We're now in the middle of a seven-year streak of only having international MVPs. We have an In-Season Tournament. We have a Play-In Tournament. That's two more tournaments than we had at the turn of the century."""

print("Original text:")
print(f'"{original_text}"')
print()

# Compute perplexity of original text
original_perplexity = compute_perplexity(original_text)
print(f"Original text perplexity: {original_perplexity:.2f}")

# Shuffle words and compute perplexity
random.seed(42)
shuffled_text = shuffle_words(original_text)
shuffled_perplexity = compute_perplexity(shuffled_text)

print(f"Shuffled text:")
print(f'"{shuffled_text}"')
print()
print(f"Shuffled text perplexity: {shuffled_perplexity:.2f}")

print(f"\nPerplexity difference: {shuffled_perplexity - original_perplexity:.2f}")
print(f"Shuffled text is {shuffled_perplexity/original_perplexity:.1f}x more perplexing")

# Part (b): Sampling Comparison
print("\n" + "="*50)
print("PART (B): SAMPLING COMPARISON")
print("="*50)

prompt = "Once upon a time"
temperatures = [0, 0.3, 0.6, 0.9, 1.2, 1.5]

print(f'Prompt: "{prompt}"')
print("Generating 500 tokens with different temperatures...\n")

for temp in temperatures:
    print(f"Temperature = {temp}")
    print("-" * 30)

    generated = generate_text(prompt, temp, max_length=500)

    print(generated)
    print("\n" + "="*50 + "\n")