import pandas as pd

# Load JSON into a DataFrame
df = pd.read_json('../dat/egpaugmented.json')

# Initialize a list to store the ratings for each row
all_ratings = []

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    # Retrieve the list of examples for the current row
    examples = row['augmented_examples']

    print(row[['Level','SuperCategory','SubCategory']])
    print(row['Can-do statement'])
    print(row['Example'])
    
    # Initialize a list to store the ratings for the current row
    ratings = []
    
    # Iterate through the list of examples
    for example in examples:
        # Ask for the rating
        print(f"Example: {example}")
        rating = input("Please rate the example (1 or 0): ")
        
        # Validate input
        while not rating.isdigit() or int(rating) not in [0, 1]:
            print("Invalid input. Please enter 1 or 0.")
            rating = input("Please rate the example (1 or 0): ")
        
        # Append the rating to the list for the current row
        ratings.append(int(rating))
    
    # Append the ratings list for the current row to the all_ratings list
    all_ratings.append(ratings)

# Add the all_ratings list as a new column in the DataFrame
df['ratings'] = all_ratings

# Save the updated DataFrame to a JSON file
df.to_json('../dat/egp_rated.json')
