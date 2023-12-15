# this script trains feed forward networks to distinguish the presence of a certain construction in sentences from its absence

import torch
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from sentence_transformers import SentenceTransformer
import random
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import config
import pandas as pd
import numpy as np
import argparse

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "CPU")

DATA_PATH = '../dat/'

# Argument parser
parser = argparse.ArgumentParser(description='Train classifiers on augmented EGP.')
parser.add_argument('--input-file', type=str, default="egpaugmented.json", help='Name of input file in folder dat (default: egpaugmented.csv)')
parser.add_argument('--output-dir', type=str, default="models", help='Name of output directory for model checkpoints (default: models)')
args = parser.parse_args()

training_data = DATA_PATH + args.input_file

# hyperparameters
batch_size=64
num_epochs=60
input_dim=1024
hidden_dim=32
random_negatives=True
lr=0.0001
emb_model='llmrails/ember-v1'

# read data and load embeddings model
df = pd.read_json(training_data)
embeddings_model = SentenceTransformer(emb_model)

# model definition
class FeedforwardNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeedforwardNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.relu(self.fc1(x))
        output = self.sigmoid(self.fc2(hidden))
        return output

# set up the dataset structure
class SentenceDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx], device=device), torch.tensor(self.labels[idx], device=device)

def get_dataset(row, random_negatives=True):
    # assemble dataset for one construction
    # 50% positive examples
    sentences = list(row['augmented_examples'])
    labels = [1] * len(row['augmented_examples'])

    if random_negatives:
        half = int(len(sentences) / 2)

        # 25% random negative examples (positive from other constructions)
        neg_examples = [example for sublist in df.loc[df['#'] != row['#'], 'augmented_examples'].to_list() for example in sublist]
        random.shuffle(neg_examples)
        sentences += neg_examples[:half]
        labels += [0] * len(neg_examples[:half])

        # 25% augmented negative examples
        aug_neg_examples = row['augmented_negative_examples']
        random.shuffle(aug_neg_examples)
        sentences += aug_neg_examples[:half]
        labels += [0] * len(aug_neg_examples[:half])
    else:
        sentences += row['augmented_negative_examples']
        labels = [0] * len(row['augmented_negative_examples'])

    embeddings = embeddings_model.encode(sentences)
    return SentenceDataset(embeddings, labels)

# train model for each construction in the augmented dataset
for idx, construction in df.iterrows():
    print(construction['Can-do statement'])
    dataset = get_dataset(construction, random_negatives)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = FeedforwardNN(input_dim, hidden_dim).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_steps = 0
        for inputs, targets in loader:
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_steps += 1

    print(f'Loss: {total_loss / train_steps}')
    
    torch.save(model, f"../{args.output_dir}/{construction['#']}.pth")