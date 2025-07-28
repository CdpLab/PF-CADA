# deap_independent.py
import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

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
    data = file['data']
    y_v = file['valence_labels'][0]
    y_a = file['arousal_labels'][0]

    x = data.transpose([0, 2, 3, 1])
    x = x.reshape((-1, t_steps, img_rows, img_cols, num_chan))

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

# ---------------- Subject Split ----------------
num_subjects = 32
num_per_subj = len(all_data) // num_subjects
indices_per_subject = {
    s: range(s * num_per_subj, (s + 1) * num_per_subj) for s in range(num_subjects)
}

# Subject 0: ~20% for finetune (target domain), rest for test
s0 = np.array(list(indices_per_subject[0]))
np.random.shuffle(s0)
split_point = int(len(s0) * 0.2)
finetune_indices = s0[:split_point]
test_indices     = s0[split_point:]

# Other 31 subjects: source domain for training/validation
others = np.hstack([np.array(list(indices_per_subject[i])) for i in range(1, num_subjects)])
np.random.shuffle(others)
val_split = int(len(others) * 0.1)
val_indices = others[:val_split]
train_indices = others[val_split:]

# ---------------- Helper: DataLoader ----------------
def build_loader_from_indices(x, y, indices, batch_size, shuffle=True):
    xs, ys = x[indices], y[indices]
    tensors = [torch.tensor(xs[:, i, :, :, :], dtype=torch.float32) for i in range(t_steps)]
    labels  = torch.tensor(np.argmax(ys, axis=1), dtype=torch.long)
    return DataLoader(TensorDataset(*tensors, labels), batch_size=batch_size, shuffle=shuffle)

val_loader = build_loader_from_indices(all_data, all_labels, val_indices, batchsize, True)
finetune_loader = build_loader_from_indices(all_data, all_labels, finetune_indices, batchsize, True)
test_loader = build_loader_from_indices(all_data, all_labels, test_indices, batchsize, False)

# ---------------- Train / Validate / Test ----------------
def validate(model, loader, criterion, device):
    model.eval()
    tot, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            inputs = [batch[i].to(device) for i in range(t_steps)]
            targets = batch[t_steps].to(device)
            class_logits, _ = model(*inputs, alpha=0.0)
            loss = criterion(class_logits, targets)
            tot += loss.item()
            pred = class_logits.argmax(dim=1)
            correct += (pred == targets).sum().item()
            n += targets.size(0)
    return tot / max(1, len(loader)), correct / max(1, n)

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

def finetune(model, loader, criterion, optimizer, device, val_loader=None, epochs=10):
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
        msg = f"[Finetune] ep {ep+1}/{epochs} loss={total:.4f}"
        if val_loader is not None:
            vl, va = validate(model, val_loader, criterion, device)
            msg += f" | val={vl:.4f}/{va:.4f}"
        print(msg)

# Adversarial domain adaptation (source labeled + target unlabeled)
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
            # Source: classification + domain
            cls_logits, dom_logits_src = model(*src_inputs, alpha=1.0)
            loss_cls = criterion_cls(cls_logits, src_targets)
            dom_targets_src = torch.zeros(dom_logits_src.size(0), dtype=torch.long, device=device)
            loss_dom_src = criterion_dom(dom_logits_src, dom_targets_src)
            # Target: domain only
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
model = MyModel(num_classes=2, t_steps=t_steps, img_hw=(img_rows, img_cols)).to(device)
criterion_cls = nn.CrossEntropyLoss()
criterion_dom = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# 5-fold adversarial training on source domain (31 subjects) w.r.t. one target finetune set
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
y_train_all = np.argmax(all_labels[train_indices], axis=1)
for fold, (tr_idx, _) in enumerate(kf.split(train_indices, y_train_all)):
    print(f"Adversarial training fold {fold+1}/5 ...")
    tr_ids = train_indices[tr_idx]
    src_loader = build_loader_from_indices(all_data, all_labels, tr_ids, batchsize, True)
    train_adversarial(model, src_loader, finetune_loader, optimizer,
                      criterion_cls, criterion_dom, device,
                      epochs=5, lambda_d=0.1)

# Finetune on target domain
finetune(model, finetune_loader, criterion_cls, optimizer, device, val_loader=val_loader, epochs=10)

# Validate + Test
vl, va = validate(model, val_loader, criterion_cls, device)
print(f"Validation: loss={vl:.4f} acc={va:.4f}")
te_acc, preds, gts = test(model, test_loader, device)
print(f"Test accuracy is: {te_acc:.4f}")

os.makedirs('exercise', exist_ok=True)
np.savetxt('exercise/DEAP_InDependent_v_confusion_matrix.csv',
           confusion_matrix(gts, preds), delimiter=',', fmt='%d')
with open('exercise/DEAP_InDependent_v_results.txt', 'w', encoding='utf-8') as f:
    f.write(f"Test Accuracy: {te_acc}\n")
