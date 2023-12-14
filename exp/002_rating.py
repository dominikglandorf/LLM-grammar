# Exp 2: Creating a rating routine for augmented examples

import pandas as pd
import random
import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
import config
random.seed(config.SEED)

# Load JSON into a DataFrame
df = pd.read_json('../dat/egpaugmented.json')

# Initialize a list to store the ratings for each row
all_positive_ratings = []
all_negative_ratings = []

# Instruction for the rater
prompt = "Please rate the example (1=NO USE or 2=USE OF RULE): "

def get_rating(example):
    # Ask for the rating
    print(f"Example: {example}")
    rating = input(prompt)
    
    # Validate input
    while not rating.isdigit() or int(rating) not in [1, 2]:
        print("Invalid input.")
        rating = input(prompt)

    return int(rating)-1

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    # Retrieve the list of examples for the current row
    positive_examples = row['augmented_examples']
    negative_examples = row['augmented_negative_examples']
    all_examples = positive_examples + negative_examples
    positive = [True] * len(positive_examples) + [False] * len(negative_examples)
    examples = list(zip(all_examples, positive))
    # The order should be random, so that the rater cannot guess whether an example is positive or not
    random.shuffle(examples)

    # Show basic information about the construction to the user
    print(row[['Level','SuperCategory','SubCategory']])
    print(row['Can-do statement'])
    print(row['Example'])
    
    # Initialize a list to store the ratings for the current row
    positive_ratings = []
    negative_ratings = []
    
    # Iterate through the list of examples and ask for ratings
    for example, positive in examples:
        rating = get_rating(example)
        if positive:
            positive_ratings.append(rating)
        else:
            negative_ratings.append(rating)
      
    all_positive_ratings.append(positive_ratings)
    all_negative_ratings.append(negative_ratings)

# Add the all_ratings list as a new column in the DataFrame
df['ratings'] = all_positive_ratings
df['negative_ratings'] = all_negative_ratings

# Save the updated DataFrame to a JSON file
df.to_json('../dat/egp_rated.json')
