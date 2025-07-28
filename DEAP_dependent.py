# deap_dependent.py
import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from pf_cada_model import MyModel

# ---------------- Config ----------------
img_rows, img_cols, num_chan = 8, 9, 4
t_steps = 6
batchsize = 256
flag = 'v'  # 'v' for Valence, 'a' for Arousal
seed = 3407
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- Load DEAP ----------------
name_index = [f"{i:02d}" for i in range(1, 33)]
dataset_dir = "D:/DEAP/with_base_0.5/"

all_data, all_labels = [], []
for sid in name_index:
    file_path = os.path.join(dataset_dir, 'DE_s' + sid)
    file = sio.loadmat(file_path)
    data = file['data']                                # (trial, band, H, W)
    y_v = file['valence_labels'][0]
    y_a = file['arousal_labels'][0]

    x = data.transpose([0, 2, 3, 1])                   # (trial, H, W, band)
    x = x.reshape((-1, t_steps, img_rows, img_cols, num_chan))  # (N, T, H, W, C)

    y_v_one, y_a_one = np.empty([0, 2]), np.empty([0, 2])
    for j in range(int(len(y_a) // t_steps)):
        v = np.array([1, 0]) if y_v[j * t_steps] == 1 else np.array([0, 1])
        a = np.array([1, 0]) if y_a[j * t_steps] == 1 else np.array([0, 1])
        y_v_one = np.vstack((y_v_one, v))
        y_a_one = np.vstack((y_a_one, a))
    y_one = y_v_one if flag == 'v' else y_a_one

    all_data.append(x)
    all_labels.append(y_one)

all_data  = np.concatenate(all_data, axis=0)                         # (N, T, H, W, C)
all_labels = np.concatenate(all_labels, axis=0)                      # (N, 2)
all_data  = np.transpose(all_data, (0, 1, 4, 2, 3))                  # (N, T, C, H, W)

# ---------------- Helper: build DataLoader ----------------
def build_loader(x, y, batch_size, shuffle=True):
    xs = [torch.tensor(x[:, i, :, :, :], dtype=torch.float32) for i in range(t_steps)]
    ys = torch.tensor(np.argmax(y, axis=1), dtype=torch.long)
    return DataLoader(TensorDataset(*xs, ys), batch_size=batch_size, shuffle=shuffle)

# ---------------- Train / Validate / Test ----------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    for batch in loader:
        inputs = [batch[i].to(device) for i in range(t_steps)]
        targets = batch[t_steps].to(device)
        optimizer.zero_grad()
        class_logits, _ = model(*inputs, alpha=0.0)  # subject-dependent: no domain alignment
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
            pred = class_logits.argmax(dim=1)
            preds.extend(pred.cpu().numpy())
            gts.extend(targets.cpu().numpy())
    preds, gts = np.array(preds), np.array(gts)
    return (preds == gts).mean(), preds, gts

# ---------------- Pretrain / 5-fold Finetune ----------------
model = MyModel(num_classes=2, t_steps=t_steps, img_hw=(img_rows, img_cols)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# Split all data: 20% for pretraining, 80% for 5-fold finetuning/testing
x_pretrain, x_finetune, y_pretrain, y_finetune = train_test_split(
    all_data, all_labels, test_size=0.8, random_state=seed, stratify=all_labels)

pre_loader = build_loader(x_pretrain, y_pretrain, batchsize, True)
print("Pretraining...")
for ep in range(60):
    loss = train_epoch(model, pre_loader, criterion, optimizer, device)
    if (ep+1) % 10 == 0:
        print(f"  [Pretrain] epoch {ep+1}/60 loss={loss:.4f}")

kf = KFold(n_splits=5, shuffle=True, random_state=seed)
acc_list, all_preds, all_gts = [], [], []
for fold, (tr_idx, te_idx) in enumerate(kf.split(x_finetune)):
    x_train_val, y_train_val = x_finetune[tr_idx], y_finetune[tr_idx]
    x_test, y_test = x_finetune[te_idx], y_finetune[te_idx]

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=0.1, random_state=seed, stratify=y_train_val)

    train_loader = build_loader(x_train, y_train, batchsize, True)
    val_loader   = build_loader(x_val,   y_val,   batchsize, False)
    test_loader  = build_loader(x_test,  y_test,  batchsize, False)

    print(f"Fold {fold+1}/5 Finetuning...")
    for ep in range(30):
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        if (ep+1) % 5 == 0:
            print(f"  [Fold {fold+1}] ep {ep+1}/30 tr={tr_loss:.4f} val={val_loss:.4f} acc={val_acc:.4f}")

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
with open('exercise/DEAP_dependent_v_results.txt', 'w', encoding='utf-8') as f:
    f.write(f"Average Test Accuracy: {mean_acc}\n")
    f.write(f"Std of Test Accuracies: {std_acc}\n")
    f.write(f"Global F1 Score: {global_f1}\n")
np.savetxt('exercise/DEAP_dependent_v_confusion_matrix.csv', global_cm, delimiter=',', fmt='%d')
