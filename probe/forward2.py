import random
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from sklearn.metrics import f1_score as compute_f1, roc_auc_score
import numpy as np
random.seed(232)
torch.manual_seed(232)
np.random.seed(232)
parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str, default='math5', choices=['math5', 'dapo17k'])
args = parser.parse_args()

raw_data = torch.load(f'./probe_data_qwen3-4b_{args.data}_traces10_with_hidden.pt')
# raw_data2 = torch.load(f'./probe_data_qwen3-4b_math5_traces10_with_hidden.pt')

# for k in raw_data2.keys():
#     if type(raw_data[k], ) == list:
#         raw_data[k].extend(raw_data2[k])
#     else:
#         raw_data[k] = torch.cat([raw_data[k], raw_data2[k]], dim=0)

keys = raw_data.keys()
value_list = zip(*[raw_data[k] for k in keys])
data_set = [dict(zip(keys, v)) for v in value_list]

print('avg answer length:', sum([d['answer_end'] for d in data_set]) / len(data_set))
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
print(data_set[0]['label'])
# random.shuffle(data_set)
# print(set(raw_data['index_data']))
idx_set = list(set([i.item() for i in raw_data['idx']]))
random.shuffle(idx_set)
# calculate the pass@10 and avg@10
total = 0
correct = 0
avg = 0.0
# for idx in idx_set:
#     data_samples = [d for d in data_set if d['idx'].item() == idx]
#     total += 1
#     # check if any of the samples is correct
#     if any([d['label'] for d in data_samples]):
#         correct += 1
#     # calculate the avg score
#     scores = [1.0 if d['label'] else 0.0 for d in data_samples]
#     if len(scores) > 0:
#         avg += sum(scores) / len(scores)
# print(f'Pass@10: {correct/total:.4f}')
# print(f'Avg@10: {avg/total:.4f}')



# exit(0)
training_ratio = 0.6
validation_ratio = 0.2
if True:
    training_idx = set(idx_set[:int(len(idx_set) * training_ratio)])
    validation_idx = set(idx_set[int(len(idx_set) * training_ratio):int(len(idx_set) * (training_ratio + validation_ratio))])
    # testing_idx = set(idx_set[int(len(idx_set) * (training_ratio + validation_ratio)):])
    testing_idx = set(idx_set[int(len(idx_set) * (training_ratio + validation_ratio)):])
    # print(f'Training idx size: {len(training_idx)}, Validation idx size: {len(validation_idx)}, Testing idx size: {len(testing_idx)}')
    # training_data = data_set[:int(len(data_set) * training_ratio)]
    # testing_data = data_set[int(len(data_set) * training_ratio):]
    training_data = [d for d in data_set if d['idx'].item() in training_idx]
    validation_data = [d for d in data_set if d['idx'].item() in validation_idx]
    testing_data = [d for d in data_set if d['idx'].item() in testing_idx]
    # testing_data = [d for d in data_set if d['idx'].item() in list(testing_idx)[7:8]]
    random.shuffle(training_data)
    random.shuffle(validation_data)
    random.shuffle(testing_data)
# else:
#     random.shuffle(data_set)
#     training_data = data_set[:int(len(data_set) * training_ratio)]
#     validation_data = data_set[int(len(data_set) * training_ratio):int(len(data_set) * (training_ratio + validation_ratio))]
#     testing_data = data_set[int(len(data_set) * (training_ratio + validation_ratio)):]
    
print(f'Training data size: {len(training_data)}, Validation data size: {len(validation_data)}, Testing data size: {len(testing_data)}')



class ProbeModel(nn.Module):
    def __init__(self, input_size):
        super(ProbeModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.initialize_weights()
    
    def forward(self, x):
        # return torch.sigmoid(self.linear(x))
        return self.linear(x)
    
    def initialize_weights(self):
        # nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
# class ProbeModel(nn.Module):
#     def __init__(self, input_size):
#         super(ProbeModel, self).__init__()
#         self.linear1 = nn.Linear(input_size, 128)
#         self.linear2 = nn.Linear(128, 1)
#         self.initialize_weights()
    
#     def forward(self, x):
#         # return torch.sigmoid(self.linear(x))
#         return self.linear2(nn.ReLU()(self.linear1(x)))
    
#     def initialize_weights(self):
#         # nn.init.xavier_uniform_(self.linear.weight)
#         # nn.init.zeros_(self.linear1.weight)
#         nn.init.zeros_(self.linear1.bias)
#         # nn.init.zeros_(self.linear2.weight)
#         nn.init.zeros_(self.linear2.bias)
    
probe_model = ProbeModel(input_size=data_set[0]['hidden_state'].shape[-1])
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(probe_model.parameters(), lr=0.001)
num_epochs = 100
batch_size = 32


def get_batch(batch, mode='train'):
    if mode == 'train':
        
        inputs = torch.stack([item['hidden_state'][3] for item in batch], dim=0)
        labels = torch.tensor([1. if item['label'] else 0. for item in batch]).unsqueeze(1)
        
        
        # inputs = torch.stack([item['hidden_state'][3] for item in batch] + [item['hidden_state'][3] for item in batch], dim=0)
        # labels = torch.tensor([1. if item['label'] else 0. for item in batch] * 2).unsqueeze(1)
    else:
        inputs = torch.stack([item['hidden_state'][3] for item in batch], dim=0)
        labels = torch.tensor([1. if item['label'] else 0. for item in batch]).unsqueeze(1)
        # inputs = torch.stack([item['hidden_state'][0] for item in batch], dim=0)
        # labels = torch.tensor([1. if item['label'] else 0. for item in batch]).unsqueeze(1)
        
    return (inputs, labels)
from scipy.stats import spearmanr
def test_batch(data, epoch, verbose=False, training_data=None, validation_data=None):
    all_preds = []
    all_labels = []
    all_scores = []
    all_logits = []
    
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            
            inputs, labels = get_batch(batch, 'test')
            # print(inputs)
            outputs = probe_model(inputs)
            # print('o',outputs)
            logits = outputs.squeeze(-1)
            # print('o',logits)
            preds = (logits >= 0).int().tolist()
            all_scores.extend(logits.detach().cpu().tolist())
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_logits.extend(logits.tolist())
            
            # print(preds, labels)
        all_labels = [l.item() for l in all_labels]
        # if verbose:
        #     print('pos=', [s for s, l in zip(all_scores, all_labels) if l == 1])
        #     print('neg=', [s for s, l in zip(all_scores, all_labels) if l == 0])
        #     print(all_preds, all_labels)
        accuracy = sum([1 if p == l else 0 for p, l in zip(all_preds, all_labels)]) / len(all_labels)
        f1_score = compute_f1(all_labels, all_preds)
        try:
            auroc = roc_auc_score(all_labels, all_scores)
        except ValueError:
            auroc = float('nan')
        # if verbose:
        print(spearmanr(all_logits, all_labels).statistic)
        
        
    # print(f'Epoch {epoch+1}/{num_epochs}, Test Accuracy: {accuracy:.4f}, Test F1: {f1_score:.4f} Loss: {loss.item():.4f}')
    # return accuracy
    return (accuracy, f1_score, auroc)

best_validation_accuracy = 0.0
testing_accuracy_at_best = 0.0

for epoch in range(num_epochs):
    # all_preds = []
    # all_labels = []
        
    random.shuffle(training_data)
    probe_model.train()
    total_loss = 0.0
    for i in range(0, len(training_data), batch_size):
        batch = training_data[i:i+batch_size]
        inputs, labels = get_batch(batch, 'train')
        
        optimizer.zero_grad()
        outputs = probe_model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.detach().item()
        loss.backward()
        optimizer.step()
    # print(probe_model.linear.weight)
    
    probe_model.eval()
    acc, f1, auroc = test_batch(validation_data, epoch)
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {acc:.4f}, Validation F1: {f1:.4f}, Validation AUROC: {auroc:.4f}, Training Loss: {total_loss/(len(training_data)/batch_size):.4f}')
    
    if auroc >= best_validation_accuracy:
        best_validation_accuracy = auroc
        _, _, testing_accuracy_at_best = test_batch(testing_data + training_data, epoch, verbose=True)
        # _, _, testing_accuracy_at_best = test_batch(validation_data, epoch, verbose=True)
        # _, _, testing_accuracy_at_best = test_batch(training_data, epoch, verbose=True)
    # if acc >= best_validation_accuracy:
        # best_validation_accuracy = acc
        # testing_accuracy_at_best, _, _ = test_batch(testing_data, epoch, verbose=True)
print(f'Best Validation Accuracy: {best_validation_accuracy:.4f}, Testing Accuracy at Best Validation: {testing_accuracy_at_best:.4f}')
