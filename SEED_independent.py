# seed_independent.py
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

from pf_cada_model import MyModel

# ---------------- Config ----------------
seed = 3407
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
np.random.seed(seed); random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

num_classes = 3
batchsize = 256
img_rows, img_cols, num_chan = 8, 9, 4
t_steps = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- Load SEED ----------------
falx = np.load('D:/SEED/SEEDzyr/DE0.5s/t6x_89.npy')
y = np.load('D:/SEED/SEEDzyr/DE0.5s/t6y_89.npy')
one_y_1 = np.array([y[:1126]] * 3).reshape((-1,))
one_y_1 = OneHotEncoder(sparse_output=False).fit_transform(one_y_1.reshape(-1, 1))

all_data, all_labels = [], []
for nb in range(15):
    one_x = falx[nb * 3:nb * 3 + 3].reshape((-1, 6, img_rows, img_cols, 5))
    one_x = one_x[:, :, :, :, 1:5]  # 4 channels
    one_y = one_y_1
    all_data.extend(one_x)
    all_labels.extend(one_y)

all_data  = np.array(all_data)                          # (N, T, H, W, C)
all_data  = all_data.transpose((0, 1, 4, 2, 3))         # (N, T, C, H, W)
all_labels = np.array(all_labels)                       # (N, 3)

# ---------------- Subject Split ----------------
total_trials = all_data.shape[0]
num_subjects = 15
num_per_subj = total_trials // num_subjects
idx_per_subj = {s: range(s * num_per_subj, (s+1) * num_per_subj) for s in range(num_subjects)}

s0 = np.array(list(idx_per_subj[0]))
np.random.shuffle(s0)
split_point = int(len(s0) * 0.1)  # 10% finetune, rest test (adjust as needed)
test_indices    = s0[:split_point]
finetune_indices = s0[split_point:]

train_indices = np.hstack([np.array(list(idx_per_subj[i])) for i in range(1, num_subjects)])

# ---------------- DataLoaders ----------------
def build_loader_from_indices(x, y, indices, batch_size, shuffle=True):
    xs, ys = x[indices], y[indices]
    tensors = [torch.tensor(xs[:, i, :, :, :], dtype=torch.float32) for i in range(t_steps)]
    labels  = torch.tensor(np.argmax(ys, axis=1), dtype=torch.long)
    return DataLoader(TensorDataset(*tensors, labels), batch_size=batch_size, shuffle=shuffle)

finetune_loader = build_loader_from_indices(all_data, all_labels, finetune_indices, batchsize, True)
test_loader     = build_loader_from_indices(all_data, all_labels, test_indices, batchsize, False)

# ---------------- Train / Test helpers ----------------
def test(model, loader, device):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for batch in loader:
            inputs = [batch[i].to(device) for i in range(t_steps)]
            targets = batch[t_steps].to(device)
            class_logits, _ = model(*inputs, alpha=0.0)
            preds.extend(class_logits.argmax(dim=1).cpu().numpy())
            gts.extend(targets.cpu().numpy())
    preds, gts = np.array(preds), np.array(gts)
    return (preds == gts).mean(), preds, gts

def finetune(model, loader, criterion, optimizer, device, epochs=100):
    for ep in range(epochs):
        model.train()
        total = 0.0
        for batch in loader:
            inputs = [batch[i].to(device) for i in range(t_steps)]
            targets = batch[t_steps].to(device)
            optimizer.zero_grad()
            class_logits, _ = model(*inputs, alpha=0.0)
            loss = criterion(class_logits, targets)
            loss.backward()
            optimizer.step()
            total += loss.item()
        if (ep+1) % 10 == 0:
            print(f"[Finetune] ep {ep+1}/{epochs} loss={total:.4f}")

def train_adversarial(model, source_loader, target_loader, optimizer,
                      criterion_cls, criterion_dom, device,
                      epochs=5, lambda_d=0.1):
    import itertools
    model.train()
    for ep in range(epochs):
        tot_cls, tot_dom = 0.0, 0.0
        steps = max(len(source_loader), len(target_loader))
        src_it = itertools.cycle(source_loader)
        tgt_it = itertools.cycle(target_loader)
        for _ in range(steps):
            src = next(src_it); tgt = next(tgt_it)
            src_inputs = [src[i].to(device) for i in range(t_steps)]
            src_targets = src[t_steps].to(device)
            tgt_inputs = [tgt[i].to(device) for i in range(t_steps)]

            optimizer.zero_grad()
            cls_logits, dom_logits_src = model(*src_inputs, alpha=1.0)
            loss_cls = criterion_cls(cls_logits, src_targets)
            dom_targets_src = torch.zeros(dom_logits_src.size(0), dtype=torch.long, device=device)
            loss_dom_src = criterion_dom(dom_logits_src, dom_targets_src)
            _, dom_logits_tgt = model(*tgt_inputs, alpha=1.0)
            dom_targets_tgt = torch.ones(dom_logits_tgt.size(0), dtype=torch.long, device=device)
            loss_dom_tgt = criterion_dom(dom_logits_tgt, dom_targets_tgt)

            loss = loss_cls + lambda_d * (loss_dom_src + loss_dom_tgt)
            loss.backward()
            optimizer.step()

            tot_cls += loss_cls.item()
            tot_dom += (loss_dom_src.item() + loss_dom_tgt.item())
        print(f"[Adv] ep {ep+1}/{epochs} cls={tot_cls:.4f} dom={tot_dom:.4f}")

# ---------------- Training Flow ----------------
model = MyModel(num_classes=num_classes, t_steps=t_steps, img_hw=(img_rows, img_cols)).to(device)
criterion_cls = nn.CrossEntropyLoss()
criterion_dom = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# 5-fold adversarial training (source = other 14 subjects; target = subject 0 finetune set)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
y_train_all = np.argmax(all_labels[train_indices], axis=1)
for fold, (tr_idx, _) in enumerate(kf.split(train_indices, y_train_all)):
    print(f"Adversarial training fold {fold+1}/5 ...")
    tr_ids = train_indices[tr_idx]
    source_loader = build_loader_from_indices(all_data, all_labels, tr_ids, batchsize, True)
    train_adversarial(model, source_loader, finetune_loader, optimizer,
                      criterion_cls, criterion_dom, device,
                      epochs=5, lambda_d=0.1)

# Finetune (target domain)
finetune(model, finetune_loader, criterion_cls, optimizer, device, epochs=100)

# Test
te_acc, preds, gts = test(model, test_loader, device)
print(f"Test accuracy is: {te_acc:.4f}")

os.makedirs('exercise', exist_ok=True)
np.savetxt('exercise/SEED_InDependent_confusion_matrix.csv',
           confusion_matrix(gts, preds), delimiter=',', fmt='%d')
with open('exercise/SEED_InDependent_results.txt', 'w', encoding='utf-8') as f:
    f.write(f"Test Accuracy: {te_acc}\n")
