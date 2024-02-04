import torch
from transformers import BertTokenizer, BertModel
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import config

from tqdm import tqdm
import copy
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def forward_all(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        task_outputs = torch.stack(
            [torch.argmax(self.task_heads[task_id](pooled_output), dim=1) for task_id in range(len(self.task_heads))],
            dim=1
        )
        return task_outputs

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=config.CACHE_DIR)
backbone_model = BertModel.from_pretrained('bert-base-uncased', cache_dir=config.CACHE_DIR)

def load_model(level, egp_df):  
    df_level = egp_df[egp_df['Level'] == level]
    task_heads = [NonlinearTaskHead(backbone_model.config.hidden_size, 2) for _ in range(len(df_level))]
    multi_task_model = MultiTaskBERT(copy.deepcopy(backbone_model), task_heads).to(device)
    multi_task_model.load_state_dict(torch.load('../models/classifiers/multi_task_model_state_dict_' + level + '.pth'))
    return multi_task_model
    
def get_scores(level_model, candidates, max_len=128, batch_size=32, use_tqdm=False):
    input_ids = []
    attention_masks = []
    
    for candidate in candidates:
        encoding = bert_tokenizer.encode_plus(
            candidate,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids.append(encoding['input_ids'].squeeze(0))
        attention_masks.append(encoding['attention_mask'].squeeze(0))
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)

    dataset = TensorDataset(input_ids, attention_masks)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_outputs = []
    loader = tqdm(dataloader) if use_tqdm else dataloader
    for batch_input_ids, batch_attention_mask in loader:
        batch_input_ids = batch_input_ids.to(device)
        batch_attention_mask = batch_attention_mask.to(device)
        
        with torch.no_grad():
            outputs = level_model.forward_all(batch_input_ids, attention_mask=batch_attention_mask)
            all_outputs.append(outputs)
    
    return torch.cat(all_outputs, dim=0)