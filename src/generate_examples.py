import pandas as pd
import re
import openai
import sys
import os
import re
import argparse

sys.path.append(os.path.dirname(os.getcwd()))
import config
openai.api_key = config.OPENAI_API_KEY

DATA_PATH = '../dat/'

# Argument parser
parser = argparse.ArgumentParser(description='Data augmentation for EGP with GPT.')
parser.add_argument('--examples-per-batch', type=int, default=20, help='Positive and negative examples per batch (default: 20)')
parser.add_argument('--batches', type=int, default=5, help='Batches (default: 5)')
parser.add_argument('--samples-per-level', type=int, default=1, help='Samples per CEFR level (default: 1)')
parser.add_argument('--input-file', type=str, default="egponline.csv", help='Name of input file in folder dat (default: egponline.csv)')
parser.add_argument('--output-file', type=str, default="egpaugmented.json", help='Name of output file in folder dat (default: egpaugmented.json)')
args = parser.parse_args()

# Check if output already exists
if os.path.exists(DATA_PATH + args.output_file):
    egp_samples = pd.read_json(DATA_PATH + args.output_file)
else:
    # Read data
    egp = pd.read_csv('../dat/' + args.input_file)
    egp_samples = egp.groupby('Level', group_keys=False).apply(lambda x: x.sample(args.samples_per_level))
    egp_samples['Example'] = egp_samples['Example'].str.replace(r"\(.*\)", "", regex=True).str.strip()

    # Create prompts
    def get_prompt(construction):
        return f'Create {args.examples_per_batch} more examples for the grammatical construction on CEFR level {construction["Level"]} in the category "{construction["SuperCategory"]}: {construction["SubCategory"]}" with guideword "{construction["guideword"]}" and the rule: "{construction["Can-do statement"]}"\n\nExamples:\n\n{construction["Example"]}\n\nOutput format:\n1. [EXAMPLE 1]\n2. [EXAMPLE 2]'

    egp_samples['prompt'] = egp_samples.apply(get_prompt, axis=1)
    egp_samples['augmented_examples'] = [[] for _ in range(len(egp_samples))]
    egp_samples['augmented_negative_examples'] = [[] for _ in range(len(egp_samples))]
    egp_samples.to_json(DATA_PATH + args.output_file)


# Now use this prompt to generate new examples until there are enough positive examples
# Requests the API
def get_examples(construction):
    print(construction['prompt'])
    messages = [
        {"role": "system", "content": "You are an English as a foreign language teacher who is knowledgable about grammar."},
        {"role": "user", "content": construction['prompt']}
    ]
    response = openai.ChatCompletion.create(model=config.OPENAI_MODEL, messages=messages )
    msg_content = response.choices[0].message.content
    print(f'{msg_content}\n\n')
    lines = msg_content.split('\n')
    positive_examples = [re.sub(r'^\d+\.\s*', '', line).strip() for line in lines if re.match(r'^\d+\.', line)]
    messages.append(response.choices[0].message)

    # negative examples
    messages.append({"role": "user", "content": "Rewrite each example with the same content but without using the rule."})
    response = openai.ChatCompletion.create(model=config.OPENAI_MODEL, messages=messages)
    msg_content = response.choices[0].message.content
    print(f'{msg_content}\n\n')
    lines = msg_content.split('\n')
    negative_examples = [re.sub(r'^\d+\.\s*', '', line).strip() for line in lines if re.match(r'^\d+\.', line)]
    return positive_examples, negative_examples

# Iterate through rows of egp_samples
# Check if the number of augmented_examples < args.batches * args.examples_per_batch
# While the above condition is true, generate examples_per_batch more positive and negative examples, append it to the array and save the file
for index, row in egp_samples.iterrows():
    while len(row['augmented_examples']) < args.batches * args.examples_per_batch:
        try:
            positive_examples, negative_examples = get_examples(row)

            # Append new examples
            egp_samples.at[index, 'augmented_examples'].extend(positive_examples)
            egp_samples.at[index, 'augmented_negative_examples'].extend(negative_examples)

            # Save progress
            egp_samples.to_json(DATA_PATH + args.output_file)
        except:
            print(f"Error")
            break