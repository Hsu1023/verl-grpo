import random
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from sklearn.metrics import f1_score as compute_f1
import numpy as np
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str, default='dapo17k', choices=['math5', 'dapo17k'])
args = parser.parse_args()

raw_data = torch.load(f'./probe_data_qwen3-4b_{args.data}_traces10_with_hidden.pt')
keys = raw_data.keys()
value_list = zip(*[raw_data[k] for k in keys])
data_set = [dict(zip(keys, v)) for v in value_list]

def balance_data_set(data_set):
    # drop some positive samples to balance the dataset 1:1
    positive_samples = [d for d in data_set if d['label']]
    negative_samples = [d for d in data_set if not d['label']]
    if len(positive_samples) > len(negative_samples):
        positive_samples = random.sample(positive_samples, len(negative_samples))
    else:
        negative_samples = random.sample(negative_samples, len(positive_samples))
    return positive_samples + negative_samples
# data_set = balance_data_set(data_set)
# random.shuffle(data_set)
# print(set(raw_data['index_data']))
idx_set = list(set([i.item() for i in raw_data['idx']]))
random.shuffle(idx_set)
# exit(0)
training_ratio = 0.6
validation_ratio = 0.2
training_idx = set(idx_set[:int(len(idx_set) * training_ratio)])
validation_idx = set(idx_set[int(len(idx_set) * training_ratio):int(len(idx_set) * (training_ratio + validation_ratio))])
testing_idx = set(idx_set[int(len(idx_set) * (training_ratio + validation_ratio)):])
# print(f'Training idx size: {len(training_idx)}, Validation idx size: {len(validation_idx)}, Testing idx size: {len(testing_idx)}')
# training_data = data_set[:int(len(data_set) * training_ratio)]
# testing_data = data_set[int(len(data_set) * training_ratio):]
training_data = [d for d in data_set if d['idx'].item() in training_idx]
validation_data = [d for d in data_set if d['idx'].item() in validation_idx]
testing_data = [d for d in data_set if d['idx'].item() in testing_idx]
random.shuffle(training_data)
random.shuffle(validation_data)
random.shuffle(testing_data)
print(f'Training data size: {len(training_data)}, Validation data size: {len(validation_data)}, Testing data size: {len(testing_data)}')



class ProbeModel(nn.Module):
    def __init__(self, input_size):
        super(ProbeModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.linear1 = nn.Linear(input_size, 128)
        self.linear2 = nn.Linear(128, 1)
        self.activation = nn.ReLU()
        self.initialize_weights()
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))
        # return torch.sigmoid(self.linear2(self.activation(self.linear1(x))))
    
    def initialize_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
    
probe_model = ProbeModel(input_size=data_set[0]['hidden_state'].shape[-1])
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(probe_model.parameters(), lr=0.001)
num_epochs = 100
batch_size = 16


def get_batch(batch, mode='train'):
    if mode == 'train':
        # inputs = torch.stack([item['hidden_state'][0] for item in batch] + [item['hidden_state'][3] for item in batch] + [item['hidden_state'][4] for item in batch] + [item['hidden_state'][5] for item in batch]+ [item['hidden_state'][6] for item in batch], dim=0)
        # labels = torch.tensor([1. if item['label'] else 0. for item in batch] * 5).unsqueeze(1)
        
        inputs = torch.stack([item['hidden_state'][0] for item in batch] + [item['hidden_state'][11] for item in batch], dim=0)
        labels = torch.tensor([1. if item['label'] else 0. for item in batch] * 2).unsqueeze(1)
        
        # inputs = torch.stack(, dim=0)
        # labels = torch.tensor([1. if item['label'] else 0. for item in batch]).unsqueeze(1)
        # inputs = torch.stack([item['hidden_state'][4] for item in batch], dim=0)
        # labels = torch.tensor([1. if item['label'] else -1. for item in batch]).unsqueeze(1)
        # labels *= torch.tensor([(item['hidden_state_pos'][4] - item['answer_start']) / item['answer_end'] for item in batch], dtype=torch.float).unsqueeze(1)
        # labels += torch.tensor([0. if item['label'] else 1. for item in batch]).unsqueeze(1)
    else:
        inputs = torch.stack([item['hidden_state'][10] for item in batch], dim=0)
        labels = torch.tensor([1. if item['label'] else 0. for item in batch]).unsqueeze(1)
    # print(inputs[0])
    return (inputs, labels)

def test_batch(data, epoch):
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            
            inputs, labels = get_batch(batch, 'test')
            outputs = probe_model(inputs)
            preds = (outputs.squeeze() >= 0.5).int().tolist()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            # print(preds, labels)
        accuracy = sum([1 if p == l else 0 for p, l in zip(all_preds, all_labels)]) / len(all_labels)
        f1_score = compute_f1(all_preds, all_labels)
        
        
    print(f'Epoch {epoch+1}/{num_epochs}, Test Accuracy: {accuracy:.4f}, Test F1: {f1_score:.4f} Loss: {loss.item():.4f}')
    return accuracy

best_validation_accuracy = 0.0
testing_accuracy_at_best = 0.0

for epoch in range(num_epochs):
    all_preds = []
    all_labels = []
        
    random.shuffle(training_data)
    probe_model.train()
    for i in range(0, len(training_data), batch_size):
        batch = training_data[i:i+batch_size]
        inputs, labels = get_batch(batch, 'train')
        
        optimizer.zero_grad()
        outputs = probe_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    probe_model.eval()
    validation_accuracy = test_batch(validation_data, epoch)
    if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = validation_accuracy
        testing_accuracy_at_best = test_batch(testing_data, epoch)
print(f'Best Validation Accuracy: {best_validation_accuracy:.4f}, Testing Accuracy at Best Validation: {testing_accuracy_at_best:.4f}')