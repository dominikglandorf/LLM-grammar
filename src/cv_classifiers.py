# Argument parser
import argparse
parser = argparse.ArgumentParser(description='CV for EGP classification')
parser.add_argument('--input-file', type=str, default='../dat/egp_merged.json', help='Name of input file in folder dat (default: egp_merged.json)')
parser.add_argument('--output-file', type=str, default='../dat/cv_results.json', help='Name of output file for results in folder dat (default: cv_results.json)')
parser.add_argument('--level', type=str, default="A1", help='Level of constructs to train')
parser.add_argument('--fold', type=int, default=0, help='Number of fold to train')
args = parser.parse_args()

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from transformers import BertModel, BertTokenizer
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score
import random
from tqdm import tqdm
import os
import pickle
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import config

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
random.seed(config.SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=config.CACHE_DIR)
max_len = 128

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

def validate(model, dataloaders):
    model.eval() 
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_dataloaders = [loader[1] for loader in dataloaders]
    for task_id, val_loader in tqdm(enumerate(val_dataloaders), total=len(val_dataloaders)):
        val_loss = 0
        val_steps = 0
        correct = 0
        total = 0
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
    
                outputs = model(input_ids, attention_mask=attention_mask, task_id=task_id)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_steps += 1
    
                predicted = outputs.argmax(dim=1)  # Assuming a classification task
    
                # Accumulate all targets and predictions
                all_targets.extend(labels.tolist())
                all_predictions.extend(predicted.tolist())
    
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / val_steps
        val_losses.append(avg_val_loss)
        val_accuracies.append(correct / total)
        val_precisions.append(precision_score(all_targets, all_predictions, average='binary'))
        val_recalls.append(recall_score(all_targets, all_predictions, average='binary'))
    return val_losses, val_accuracies, val_precisions, val_recalls

criterion = torch.nn.CrossEntropyLoss()

def train(model, dataloaders, lr=0.0001, num_epochs=5):
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    train_losses = []
    val_losses=[]
    val_accuracies=[]
    val_precisions=[]
    val_recalls=[]
    for epoch in range(num_epochs):  # Number of epochs
        model.train()  # Set the model to training mode
        total_loss = train_steps = 0

        train_loaders = zip(*[loader[0] for loader in dataloaders])
        num_batches = len(dataloaders[0][0])     

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
        
        batch_val_losses, val_accuracies, val_precisions, val_recalls = validate(model, dataloaders)
        print(f'Validation loss: {np.mean(batch_val_losses)}')
        print(f'Mean Accuracy: {np.mean(val_accuracies)}')
        val_losses.append(np.mean(batch_val_losses))
        if len(val_losses) > 1 and val_losses[-1] > val_losses[-2]:
            break

    return model, train_losses, val_losses, val_accuracies, val_precisions, val_recalls

def get_loaders(dataset, train_ids, test_ids, batch_size=8):
    train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_ids))
    val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_ids))
    return train_dataloader, val_dataloader

def train_level_model(level, fold, train_ids, test_ids, max_constructs=500, batch_size=8, num_epochs=5, file_path="results.json"):
    df = pd.read_json(args.input_file)
    print(f"Level: {level}, Fold: {fold}")
    df_level = df[df['Level'] == level]
    num_classifiers = min(len(df_level), max_constructs)
    backbone_model = BertModel.from_pretrained('bert-base-uncased', cache_dir=config.CACHE_DIR)
    task_heads = [NonlinearTaskHead(backbone_model.config.hidden_size, 2) for _ in range(num_classifiers)]
    multi_task_model = MultiTaskBERT(backbone_model, task_heads).to(device)
    datasets = [get_dataset(df_level.iloc[idx], tokenizer, max_len, df) for idx in tqdm(range(num_classifiers))]
    
    dataloaders = [get_loaders(dataset, train_ids, test_ids, batch_size) for dataset in datasets]
    _, train_losses, val_losses, val_accuracies, val_precisions, val_recalls = train(multi_task_model, dataloaders, num_epochs=num_epochs)
    new_row = pd.DataFrame({
        "level": [level],
        "fold": [fold],
        "train_losses": [train_losses],
        "val_losses": [val_losses],
        "val_accuracies": [val_accuracies],
        "val_precisions": [val_precisions],
        "val_recalls": [val_recalls]
    })
    if os.path.exists(file_path):
        df_results = pd.read_json(file_path)
    else:
        df_results = pd.DataFrame()
    df_results = pd.concat([df_results, new_row], ignore_index=True)
    df_results.to_json(file_path, index=False)

# Generate/read the split
num_folds = 5
split_file = 'split.pkl'
len_dat = 1000
if os.path.exists(split_file):
    with open(split_file, 'rb') as file:
        split = pickle.load(file)
else:
    k_fold = KFold(n_splits=num_folds, shuffle=True)
    split = list(k_fold.split(list(range(len_dat))))

    with open(split_file, 'wb') as file:
        pickle.dump(split, file)

train_ids, test_ids = split[args.fold]
assert len(set(train_ids).intersection(set(test_ids))) == 0
train_level_model(args.level, args.fold, train_ids, test_ids, max_constructs=500, num_epochs=5, file_path=args.output_file)