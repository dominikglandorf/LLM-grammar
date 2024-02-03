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
    
def get_dataset(row, tokenizer, max_len, random_negatives=True):
    # assemble dataset for one construction
    # 50% positive examples
    sentences = list(row['augmented_examples'])
    labels = [1] * len(row['augmented_examples'])

    if random_negatives:
        ratio = 0.5
        num_rands = int(len(sentences) * ratio)
        num_augs = int(len(sentences) * (1-ratio))

        # 25% random negative examples (positive from other constructions)
        neg_examples = [example for sublist in df.loc[df['#'] != row['#'], 'augmented_examples'].to_list() for example in sublist]
        random.shuffle(neg_examples)
        sentences += neg_examples[:num_rands]
        labels += [0] * len(neg_examples[:num_rands])

        # 25% augmented negative examples
        aug_neg_examples = row['augmented_negative_examples']
        random.shuffle(aug_neg_examples)
        sentences += aug_neg_examples[:num_augs]
        labels += [0] * len(aug_neg_examples[:num_augs])
    else:
        sentences += row['augmented_negative_examples']
        labels = [0] * len(row['augmented_negative_examples'])

    return SentenceDataset(sentences, labels, tokenizer, max_len)

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir="/mnt/qb/work/meurers/mpb672/cache")
max_len = 128  # Max length for BERT

def get_loaders(dataset, batch_size=8):
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader

class TaskHead(torch.nn.Module):
    def __init__(self, bert_hidden_size, num_labels, dropout_rate=0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(bert_hidden_size, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.classifier(x)

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

def plot_validation(train_losses, val_losses, val_accuracies, val_precisions, val_recalls):
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.bar(range(len(val_accuracies)), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Task')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.bar(range(len(val_precisions)), val_precisions, label='Validation Precision')
    plt.xlabel('Task')
    plt.ylabel('Precision')
    plt.title('Validation Precision')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.bar(range(len(val_recalls)), val_recalls, label='Validation Recall')
    plt.xlabel('Task')
    plt.ylabel('Recall')
    plt.title('Validation Recall')
    plt.legend()

    plt.tight_layout()
    plt.show()

criterion = torch.nn.CrossEntropyLoss()

def train(model, dataloaders, verbose=True, plots=True, n_epochs_stop = 10, lr=0.0001, num_epochs=5):
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
    if plots:
        plot_validation(train_losses, val_losses, val_accuracies, val_precisions, val_recalls)

    return model, val_accuracies, val_precisions, val_recalls

def train_level_model(level="A1", max_constructs=500):
    print(f"Level {level}")
    df_level = df[df['Level'] == level]
    num_classifiers = min(len(df_level), max_constructs)
    backbone_model = BertModel.from_pretrained('bert-base-uncased', cache_dir="/mnt/qb/work/meurers/mpb672/cache")
    task_heads = [NonlinearTaskHead(backbone_model.config.hidden_size, 2) for _ in range(num_classifiers)]
    multi_task_model = MultiTaskBERT(backbone_model, task_heads).to(device)
    datasets = [get_dataset(df_level.iloc[idx], tokenizer, max_len) for idx in tqdm(range(num_classifiers))]
    dataloaders = [get_loaders(dataset) for dataset in datasets]
    train(multi_task_model, dataloaders, verbose=False, num_epochs=5)
    torch.save(multi_task_model.state_dict(), '../models/bert/multi_task_model_state_dict_' + level + '.pth')

level = sys.argv[1]
train_level_model(level)