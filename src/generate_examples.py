import pandas as pd
import re
import sys
import os
import re
import argparse
import random
import signal

sys.path.append(os.path.dirname(os.getcwd()))
import config
from openai import OpenAI
client = OpenAI(api_key=config.OPENAI_API_KEY)

DATA_PATH = '../dat/'

# Argument parser
parser = argparse.ArgumentParser(description='Data augmentation for EGP with GPT.')
parser.add_argument('--examples-per-batch', type=int, default=20, help='Positive and negative examples per batch (default: 20)')
parser.add_argument('--batches', type=int, default=3, help='Batches (default: 3)')
parser.add_argument('--samples-per-level', type=int, default=1, help='Samples per CEFR level (default: 1)')
parser.add_argument('--negative-ratio', type=float, default=1.0, help='Ratio of negative examples to generate (default: 1.0)')
parser.add_argument('--input-file', type=str, default="egponline.csv", help='Name of input file in folder dat (default: egponline.csv)')
parser.add_argument('--output-file', type=str, default="egpaugmented.json", help='Name of output file in folder dat (default: egpaugmented.json)')
parser.add_argument('--min-nr', type=int, default=1, help='Minimum construct ID (default: 1)')
parser.add_argument('--max-nr', type=int, default=1222, help='Minimum construct ID (default: 1222)')
parser.add_argument('--level', type=str, default=None, help='Level to consider for generation (default: all')
args = parser.parse_args()

def signal_handler(sig, frame):
    print('You pressed Ctrl+C! Exiting gracefully.')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Check if output already exists
if os.path.exists(DATA_PATH + args.output_file):
    egp_samples = pd.read_json(DATA_PATH + args.output_file)
else:
    egp = pd.read_csv('../dat/' + args.input_file, index_col=0)
    egp_samples = egp
    #egp_samples = egp.groupby(['Level', 'type'], group_keys=False).apply(lambda x: x.sample(min(len(x), args.samples_per_level), random_state=config.SEED+3))
    egp_samples['Example'] = egp_samples['Example'].str.replace(r"\(.*\)", "", regex=True).str.strip()

    def get_prompt(construction):
        lexical_range = ''
        if not pd.isna(construction["Lexical Range"]):
            if construction["Lexical Range"] == 1:
                lexical_range = 'low'
            elif construction["Lexical Range"] == 2:
                lexical_range = 'medium'
            elif construction["Lexical Range"] == 3:
                lexical_range = 'high'
            lexical_range = f'Use words of {lexical_range} difficulty in the rule.'
        return f'Learn the grammar rule "{construction["Can-do statement"]}" ({construction["SuperCategory"]}, {construction["SubCategory"]}, {construction["guideword"]}). It is CEFR level {construction["Level"]}. {lexical_range}\nExamples:\n{construction["Example"]}\nCreate {args.examples_per_batch} more examples using that rule.'

    egp_samples['prompt'] = egp_samples.apply(get_prompt, axis=1)
    egp_samples['augmented_examples'] = [[] for _ in range(len(egp_samples))]
    egp_samples['augmented_examples_source'] = [[] for _ in range(len(egp_samples))]
    egp_samples['augmented_examples_model'] = [[] for _ in range(len(egp_samples))]
    egp_samples['augmented_examples_response'] = [[] for _ in range(len(egp_samples))]
    egp_samples['augmented_negative_examples'] = [[] for _ in range(len(egp_samples))]
    egp_samples['augmented_negative_examples_source'] = [[] for _ in range(len(egp_samples))]
    egp_samples['augmented_negative_examples_model'] = [[] for _ in range(len(egp_samples))]
    egp_samples['augmented_negative_examples_response'] = [[] for _ in range(len(egp_samples))]
    egp_samples.to_json(DATA_PATH + args.output_file)


# Now use this prompt to generate new examples until there are enough positive and negative examples
def get_examples(construction, create_negative_examples=True):
    print(construction['prompt'])
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": construction['prompt']}
    ]
    response = client.chat.completions.create(model=config.OPENAI_MODEL, messages=messages, presence_penalty=0.5, max_tokens=4095)
    msg_content = response.choices[0].message.content
    positive_response = msg_content
    print(f'{msg_content}\n\n')
    # matches numbers and whitespace at the beginning of each line
    pattern = r"^\d+\.\s+(.*)"
    matches = re.findall(pattern, msg_content, re.MULTILINE)
    positive_examples = [match for match in matches] # removes numbers and prefixes of the example
    print(positive_examples)
    messages.append(response.choices[0].message)

    # negative examples
    negative_examples = []
    negative_response = ""
    if create_negative_examples:
        messages.append({"role": "user", "content": "Rewrite each created example as a minimal pair that does not show the usage of the given rule."})
        response = client.chat.completions.create(model=config.OPENAI_MODEL, messages=messages, temperature=0.4)
        msg_content = response.choices[0].message.content
        negative_response = msg_content
        print(f'{msg_content}\n\n')
        pattern = r' \(.*\)' # remove explanations in parenthesis
        msg_content = re.sub(pattern, '', msg_content)
        # in case there are two lines (one being the original example, the other being the rewritten example)
        pattern = r"((\S*Original\S*|A).*:.*\n|(?<=\d\.).*\n\s*(-|\s+(.+:.)+)|(?<=\d\.).*->)"
        msg_content = re.sub(pattern, "", msg_content)
        # removes numbers and prefixes of the example
        pattern = r"^\d+\.\s?(.+:.)?(.*)"
        matches = re.findall(pattern, msg_content, re.MULTILINE)
        negative_examples = [match[1] for match in matches]
        print(negative_examples)
        
    if positive_examples == negative_examples: # probably caused by parsing errors
        negative_examples = []
    return positive_examples, negative_examples, positive_response, negative_response

# iterate through rows of egp_samples
# check if the number of augmented_examples < args.batches * args.examples_per_batch
# while the above condition is true, generate examples_per_batch more positive and negative examples, append it to the array and save the file
target_number = args.batches * args.examples_per_batch
for index, row in egp_samples.iterrows():
    if int(row['#']) < args.min_nr or int(row['#']) > args.max_nr:
        continue
    if row['Level'] != args.level and args.level is not None:
        continue
    print(row['#'])
    print(len(row['augmented_examples']))
    while len(row['augmented_examples']) < target_number:
        try:
            create_negative_examples = len(row['augmented_negative_examples']) < int(target_number * args.negative_ratio)
            positive_examples, negative_examples, positive_response, negative_response = get_examples(row, create_negative_examples)

            source_number = random.randint(1, 1000000)
            egp_samples.at[index, 'augmented_examples'].extend(positive_examples)
            egp_samples.at[index, 'augmented_examples_source'].extend([source_number] * len(positive_examples))
            egp_samples.at[index, 'augmented_examples_model'].extend([config.OPENAI_MODEL] * len(positive_examples))
            egp_samples.at[index, 'augmented_examples_response'].extend([positive_response])
            egp_samples.at[index, 'augmented_negative_examples'].extend(negative_examples)
            egp_samples.at[index, 'augmented_negative_examples_source'].extend([source_number] * len(negative_examples))
            egp_samples.at[index, 'augmented_negative_examples_model'].extend([config.OPENAI_MODEL] * len(negative_examples))
            egp_samples.at[index, 'augmented_negative_examples_response'].extend([negative_response])

            egp_samples.to_json(DATA_PATH + args.output_file) # not losing progress in case of API errors
        except:
            print(f"Error")
            break