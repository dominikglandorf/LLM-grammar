# Argument parser
import argparse
parser = argparse.ArgumentParser(description='Full training for EGP classification')
parser.add_argument('--input-file', type=str, default='../dat/egp_merged.json', help='Name of input file in folder dat (default: egp_merged.json)')
parser.add_argument('--level', type=str, default="A1", help='Level of constructs to train')
args = parser.parse_args()

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from sklearn.metrics import precision_score, recall_score
import random
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import config

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
random.seed(config.SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_json('../dat/egp_merged.json')

class SentenceDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.sentences[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
    
def get_dataset(row, tokenizer, max_len, df, random_negatives=True, ratio = 0.5, max_positive_examples=500):
    # assemble dataset for one construction
    # 50% positive examples
    unique_examples = list(set(row['augmented_examples']))
    sentences = unique_examples[:max_positive_examples]
    labels = [1] * len(sentences)

    num_augs = int(len(sentences) * (1-ratio)) if random_negatives else len(sentences)
    # augmented negative examples
    aug_neg_examples = list(set(row['augmented_negative_examples']).difference(set(row['augmented_examples'])))
    random.shuffle(aug_neg_examples)
    unique_negatives = aug_neg_examples[:num_augs]
    sentences += unique_negatives
    labels += [0] * len(unique_negatives)
    
    if random_negatives:
        num_rands = max_positive_examples - len(unique_negatives) # fill to an even number
        # rest: random negative examples (positive from other constructions)
        neg_examples = [example for sublist in df.loc[df['#'] != row['#'], 'augmented_examples'].to_list() for example in sublist]
        random.shuffle(neg_examples)
        sentences += neg_examples[:num_rands]
        labels += [0] * len(neg_examples[:num_rands])
    assert len(sentences) == 2 * max_positive_examples
    assert sum(labels) == max_positive_examples
    return SentenceDataset(sentences, labels, tokenizer, max_len)

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=config.CACHE_DIR)
max_len = 128  # Max length for BERT

class NonlinearTaskHead(torch.nn.Module):
    def __init__(self, input_dim, num_labels, hidden_dim=16):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.classifier = torch.nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        hidden = self.relu(self.fc1(x))
        output = self.classifier(hidden)
        return output

class MultiTaskBERT(torch.nn.Module):
    def __init__(self, bert, task_heads):
        super().__init__()
        self.bert = bert
        self.task_heads = torch.nn.ModuleList(task_heads)

    def forward(self, input_ids, attention_mask, task_id):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        task_output = self.task_heads[task_id](pooled_output)
        return task_output

criterion = torch.nn.CrossEntropyLoss()

def train(model, dataloaders, lr=0.0001, num_epochs=3):
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    
    for epoch in range(num_epochs):  # Number of epochs
        model.train()  # Set the model to training mode
        num_batches = len(dataloaders[0])
        total_loss = train_steps = 0

        for batches in tqdm(train_loaders, total=num_batches):
            for task_id, batch in enumerate(batches):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
    
                outputs = model(input_ids, attention_mask=attention_mask, task_id=task_id)
                loss = criterion(outputs, labels)
                loss.backward()
                total_loss += loss.item()
                train_steps += 1

            optimizer.step()
            optimizer.zero_grad()
        
        avg_train_loss = total_loss / train_steps
        train_losses.append(avg_train_loss)
        print(f'Training loss: {avg_train_loss}')

def train_level_model(level="A1", max_constructs=500, batch_size=8):
    print(f"Level {level}")
    df_level = df[df['Level'] == level]
    num_classifiers = min(len(df_level), max_constructs)
    backbone_model = BertModel.from_pretrained('bert-base-uncased', cache_dir=config.CACHE_DIR)
    task_heads = [NonlinearTaskHead(backbone_model.config.hidden_size, 2) for _ in range(num_classifiers)]
    multi_task_model = MultiTaskBERT(backbone_model, task_heads).to(device)
    datasets = [get_dataset(df_level.iloc[idx], tokenizer, max_len) for idx in tqdm(range(num_classifiers))]
    dataloaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in datasets]
    train(multi_task_model, dataloaders, verbose=False, num_epochs=5)
    torch.save(multi_task_model.state_dict(), '../models/bert/multi_task_model_state_dict_' + level + '.pth')

train_level_model(args.level, max_constructs=5)