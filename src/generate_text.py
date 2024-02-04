# Argument parser
import argparse
parser = argparse.ArgumentParser(description='Text Generation with Grammar Constraints')
parser.add_argument('--input-file', type=str, default='cefr_leveled_texts.csv', help='Name of input file in folder dat (default: cefr_leveled_texts.csv)')
parser.add_argument('--output-file', type=str, default='controlled_generated_texts.csv', help='Name of output file for texts in folder dat (default: controlled_generated_texts.csv)')
parser.add_argument('--level', type=str, default=None, help='Level of constructs to generate (default: None)')
parser.add_argument('--num-stories', type=int, default=50, help='Number of stories to generate (default: 50)')
parser.add_argument('--num-candidates', type=int, default=5, help='Number of sentene candidates to generate (default: 5)')
parser.add_argument('--prompt-length', type=int, default=50, help='Characters of the story prompt (default: 50)')
args = parser.parse_args()

import torch
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import config
import random
sys.path.append('../src')
import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(config.SEED)

model = AutoModelForCausalLM.from_pretrained(config.GENERATION_MODEL, device_map="auto", torch_dtype=torch.float16, cache_dir=config.CACHE_DIR)
tokenizer = AutoTokenizer.from_pretrained(config.GENERATION_MODEL, cache_dir=config.CACHE_DIR)

df = pd.read_json('../dat/egp_merged.json')
levels = ["A1", "A2", "B1", "B2", "C1", "C2"] if args.level is None else [args.level]
level_models = {level: models.load_model(level, df) for level in levels}

def generate_candidate(input_ids, max_token_sentence = 64, tok_k=10, eos_chars = [".", "!", "?"]):
    generated_tokens = torch.tensor([[]], dtype=torch.int, device=device)
    with torch.no_grad():
        for _ in range(max_token_sentence):
            next_token_logits = model(torch.cat([input_ids, generated_tokens], dim=1)).logits
            probs = torch.nn.functional.softmax(next_token_logits[:, -1, :], dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, tok_k)
            renormalized_top_k_probs = top_k_probs / top_k_probs.sum()
            top_k_id = torch.multinomial(renormalized_top_k_probs, num_samples=1).item()
            next_token_id = top_k_indices[0, top_k_id]
            
            next_token = tokenizer.decode(next_token_id)
            generated_tokens = torch.cat([generated_tokens, torch.tensor([[next_token_id]]).to(device)], dim=1)
            if any(eos_char in next_token for eos_char in eos_chars):
                break

    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

def write_story(level, story, num_candidates=3, max_len = 1024, add_info=False):
    info = f", which is described as {description[level]}" if add_info else ""
    prompt = f"<s>[INST] Continue the writing on CEFR level {level}{info}. Do not talk about the CEFR level. [/INST] "
    while len(story) < max_len:
        inputs = tokenizer(prompt + story, return_tensors="pt").to(device)
        candidates = [generate_candidate(inputs.input_ids) for i in range(num_candidates)]
        if num_candidates == 1: # no ranking
            story += " " + candidates[0]
        else: # ranking by level model
            scores = models.get_scores(level_models[level], candidates)
            mean_scores = torch.mean(scores.float(),dim=1)
            story += " " + candidates[torch.argmax(mean_scores)]
    return story

cefr_texts = pd.read_csv('../dat/' + args.input_file)
storyPrompts = cefr_texts.text.apply(lambda text: text[:text.find(' ', args.prompt_length)].strip().lstrip('\ufeff')).unique()
random.shuffle(storyPrompts)

for story in storyPrompts[:args.num_stories]:
    print("_" * 100)
    print(story)
    for level in level_models.keys():
        print(level)
        text = write_story(level, story, args.num_candidates, add_info=False)
        print(text)
        new_row = {"label": level, "story": story, "text": text}
        pd.DataFrame([new_row]).to_csv('../dat/' + args.output_file, mode='a', index=False, header=not os.path.exists('../dat/' + args.output_file))