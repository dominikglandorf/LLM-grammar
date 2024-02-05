# Argument parser
import argparse
parser = argparse.ArgumentParser(description='Text Generation with Grammar Constraints')
parser.add_argument('--input-file', type=str, default='cefr_leveled_texts.csv', help='Name of input file in folder dat (default: cefr_leveled_texts.csv)')
parser.add_argument('--output-file', type=str, default='controlled_generated_texts_gpt.csv', help='Name of output file for texts in folder dat (default: controlled_generated_texts.csv)')
parser.add_argument('--level', type=str, default=None, help='Level of constructs to generate (default: None)')
parser.add_argument('--num-stories', type=int, default=50, help='Number of stories to generate (default: 50)')
parser.add_argument('--prompt-length', type=int, default=50, help='Characters of the story prompt (default: 50)')
args = parser.parse_args()

import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import config
import random

random.seed(config.SEED)
from openai import OpenAI
client = OpenAI(api_key=config.OPENAI_API_KEY)


description = {
    "C2": "Can produce clear, smoothly flowing, complex texts in an appropriate and effective style and a logical structure which helps the reader identify significant points.",
    "C1": "Can produce clear, well-structured texts of complex subjects, underlining the relevant salient issues, expanding and supporting points of view at some length with subsidiary points, reasons and relevant examples, and rounding off with an appropriate conclusion.",
    "B2": "Can produce clear, detailed texts on a variety of subjects related to their field of interest, synthesising and evaluating information and arguments from a number of sources.",
    "B1": "Can produce straightforward connected texts on a range of familiar subjects within their field of interest, by linking a series of shorter discrete elements into a linear sequence.",
    "A2": "Can produce a series of simple phrases and sentences linked with simple connectors like “and”, “but” and “because”.",
    "A1": "Can give information about matters of personal relevance (e.g. likes and dislikes, family, pets) using simple words/signs and basic expressions. Can produce simple isolated phrases and sentences."
}

def write_story(level, story, max_len = 256, add_info=False):
    info = f" ({description[level]})" if add_info else ""
    prompt = f"Continue the writing with as many grammar constructs on CEFR level {level} as possible {info}. Do not talk about the CEFR level. The story begins with "
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt + story}
    ]

    response = client.chat.completions.create(model=config.OPENAI_MODEL, messages=messages, max_tokens=max_len)
    return story + " " + response.choices[0].message.content

cefr_texts = pd.read_csv('../dat/' + args.input_file)
storyPrompts = cefr_texts.text.apply(lambda text: text[:text.find(' ', args.prompt_length)].strip().lstrip('\ufeff')).unique()
random.shuffle(storyPrompts)
if os.path.exists('../dat/' + args.output_file):
    output_df = pd.read_csv('../dat/' + args.output_file)
else:
    output_df = pd.DataFrame()
levels = ["A1", "A2", "B1", "B2", "C1", "C2"] if args.level is None else [args.level]

for story in storyPrompts[:args.num_stories]:
    print("_" * 100)
    print(story)
    for level in levels:
        print(level)
        if len(output_df) and len(output_df.loc[(output_df['label']==level) & (output_df['story']==story)]) > 0: continue
        text = write_story(level, story, add_info=True)
        print(text)
        new_row = {"label": level, "story": story, "text": text}
        pd.DataFrame([new_row]).to_csv('../dat/' + args.output_file, mode='a', index=False, header=not os.path.exists('../dat/' + args.output_file))