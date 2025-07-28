# seed_dependent.py
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

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
falx = np.load('D:/SEED/SEEDzyr/DE0.5s/t6x_89.npy')  # (15*3*..., 6, 8, 9, 5)
y = np.load('D:/SEED/SEEDzyr/DE0.5s/t6y_89.npy')
one_y_1 = np.array([y[:1126]] * 3).reshape((-1,))
one_y_1 = OneHotEncoder(sparse_output=False).fit_transform(one_y_1.reshape(-1, 1))

all_data, all_labels = [], []
for nb in range(15):
    one_x = falx[nb * 3:nb * 3 + 3].reshape((-1, 6, img_rows, img_cols, 5))
    one_x = one_x[:, :, :, :, 1:5]   # take 4 frequency-band channels
    one_y = one_y_1
    all_data.extend(one_x)
    all_labels.extend(one_y)

all_data = np.array(all_data)             # (N, T, H, W, C)
all_data = all_data.transpose((0, 1, 4, 2, 3))  # (N, T, C, H, W)
all_labels = np.array(all_labels)         # (N, 3)

# ---------------- Helpers ----------------
def build_loader(x, y, batch_size, shuffle=True):
    xs = [torch.tensor(x[:, i, :, :, :], dtype=torch.float32) for i in range(t_steps)]
    ys = torch.tensor(np.argmax(y, axis=1), dtype=torch.long)
    return DataLoader(TensorDataset(*xs, ys), batch_size=batch_size, shuffle=shuffle)

def train_epoch(model, loader, criterion, optimizer, device):
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
    return total / max(1, len(loader))

def validate(model, loader, criterion, device):
    model.eval()
    total, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            inputs = [batch[i].to(device) for i in range(t_steps)]
            targets = batch[t_steps].to(device)
            class_logits, _ = model(*inputs, alpha=0.0)
            loss = criterion(class_logits, targets)
            total += loss.item()
            pred = class_logits.argmax(dim=1)
            correct += (pred == targets).sum().item()
            n += targets.size(0)
    return total / max(1, len(loader)), correct / max(1, n)

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

# ---------------- Pretrain + 5-fold Finetune ----------------
model = MyModel(num_classes=num_classes, t_steps=t_steps, img_hw=(img_rows, img_cols)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

x_pre, x_ft, y_pre, y_ft = train_test_split(all_data, all_labels, test_size=0.2, random_state=seed, stratify=all_labels)
pre_loader = build_loader(x_pre, y_pre, batchsize, True)

print("Pretraining...")
for ep in range(60):
    loss = train_epoch(model, pre_loader, criterion, optimizer, device)
    if (ep+1) % 10 == 0:
        print(f"  [Pretrain] ep {ep+1}/60 loss={loss:.4f}")

acc_list, all_preds, all_gts = [], [], []
kf = KFold(n_splits=5, shuffle=True, random_state=seed)
for fold, (tr_idx, te_idx) in enumerate(kf.split(x_ft)):
    x_trv, y_trv = x_ft[tr_idx], y_ft[tr_idx]
    x_te, y_te = x_ft[te_idx], y_ft[te_idx]
    x_tr, x_val, y_tr, y_val = train_test_split(x_trv, y_trv, test_size=0.2, random_state=seed, stratify=y_trv)

    train_loader = build_loader(x_tr,  y_tr,  batchsize, True)
    val_loader   = build_loader(x_val, y_val, batchsize, False)
    test_loader  = build_loader(x_te,  y_te,  batchsize, False)

    print(f"Fold {fold+1}/5 Finetuning...")
    for ep in range(20):
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        vl, va = validate(model, val_loader, criterion, device)
        if (ep+1) % 5 == 0:
            print(f"  [Fold {fold+1}] ep {ep+1}/20 tr={tr_loss:.4f} val={vl:.4f}/{va:.4f}")

    te_acc, preds, gts = test(model, test_loader, device)
    acc_list.append(te_acc)
    all_preds.extend(preds)
    all_gts.extend(gts)
    print(f"Fold {fold+1} Test Acc: {te_acc:.4f}")

mean_acc, std_acc = np.mean(acc_list), np.std(acc_list)
global_acc = accuracy_score(all_gts, all_preds)
global_f1  = f1_score(all_gts, all_preds, average='weighted')
global_cm  = confusion_matrix(all_gts, all_preds)

print(f"Average Test Accuracy: {mean_acc:.4f}")
print(f"Std of Test Accuracies: {std_acc:.4f}")
print(f"Global Accuracy: {global_acc:.4f}")
print("Global F1 Score:", global_f1)
print("Global Confusion Matrix:\n", global_cm)

os.makedirs('exercise', exist_ok=True)
with open('exercise/SEED_dependent_results.txt', 'w', encoding='utf-8') as f:
    f.write(f"Average Test Accuracy: {mean_acc}\n")
    f.write(f"Std of Test Accuracies: {std_acc}\n")
    f.write(f"Global F1 Score: {global_f1}\n")
np.savetxt('exercise/SEED_dependent_confusion_matrix.csv', global_cm, delimiter=',', fmt='%d')
