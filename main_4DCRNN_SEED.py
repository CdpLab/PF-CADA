import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
import numpy as np
import time
import torch.nn.functional as F
import torch.optim as optim


# 设定参数
num_classes = 3
batch_size = 128
img_rows, img_cols, num_chan = 8, 9, 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据
falx = np.load('D:/SEED/SEEDzyr/DE0.5s/t6x_89.npy')
y = np.load('D:/SEED/SEEDzyr/DE0.5s/t6y_89.npy')
y = torch.tensor(y, dtype=torch.long)

# 数据预处理
one_y_1 = np.array([y[:1126]] * 3).reshape((-1,))
one_y_1 = np.eye(num_classes)[one_y_1]  # 使用numpy进行独热编码

acc_list = []
std_list = []

class BaseNetwork(nn.Module):
    def __init__(self, input_dim):
        super(BaseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim[0], out_channels=64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(in_features=64 * (input_dim[1] // 2) * (input_dim[2] // 2), out_features=512)
        self.reshape = lambda x: x.view(-1, 1, 512)  # Reshape layer

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool1(x)
        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        x = self.reshape(x)
        return x

class MyModel(nn.Module):
    def __init__(self, input_dim):
        super(MyModel, self).__init__()
        self.base_network = BaseNetwork(input_dim)
        self.lstm = nn.LSTM(input_size=512, hidden_size=128, batch_first=True)
        self.out_layer = nn.Linear(128, 3)

    def forward(self, inputs):
        # inputs is a list of 6 tensors
        base_outputs = [self.base_network(input_tensor) for input_tensor in inputs]
        lstm_input = torch.cat(base_outputs, dim=1)  # Concatenate along the sequence dimension
        lstm_out, _ = self.lstm(lstm_input)
        # We take the output of the last LSTM cell
        output = self.out_layer(lstm_out[:, -1, :])
        return F.softmax(output, dim=1)

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        inputs = [batch[i].to(device) for i in range(6)]  # 移动数据到设备
        targets = batch[6].to(device)  # 移动标签到设备
        optimizer.zero_grad()
        outputs = model(*inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # 打印每个批次的信息
        if batch_idx % 10 == 0:  # 每10个批次打印一次
            print(f'Epoch {epoch}, Batch {batch_idx}, Current Batch Loss: {loss.item()}')

    return total_loss / len(train_loader)


def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = [batch[i].to(device) for i in range(6)]  # 移动数据到设备
            targets = batch[6].to(device)  # 移动标签到设备
            outputs = model(*inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets).sum().item()
    return total_loss / len(val_loader), correct / len(val_loader.dataset)


def test(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs = [batch[i].to(device) for i in range(6)]  # 移动数据到设备
            targets = batch[6].to(device)  # 移动标签到设备
            outputs = model(*inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets).sum().item()
    return correct / len(test_loader.dataset)

# 训练和评估模型
for nb in range(15):
    start = time.time()
    one_falx_1 = falx[nb * 3:nb * 3 + 3]
    one_falx_1 = one_falx_1.reshape((-1, 6, img_rows, img_cols, 5))
    one_y = torch.tensor(one_y_1, dtype=torch.float32)
    one_falx = one_falx_1[:,:,:,:,1:5]  # 使用四个频段

    seed = 3407
    np.random.seed(seed)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    all_acc = []

    for train_index, test_index in kfold.split(one_falx, one_y.argmax(1)):
        model = MyModel(input_dim=(num_chan, img_rows, img_cols))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())

        # 分割训练集和测试集
        train_data, test_data = one_falx[train_index], one_falx[test_index]
        train_labels, test_labels = one_y[train_index], one_y[test_index]

        validation_split = 0.2  # 使用20%的训练数据作为验证集
        split_index = int(len(train_data) * (1 - validation_split))
        train_data, validation_data = train_data[:split_index], train_data[split_index:]
        train_labels, validation_labels = train_labels[:split_index], train_labels[split_index:]

        # 数据加载
        train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32),
                                      torch.tensor(train_labels, dtype=torch.long))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        validation_dataset = TensorDataset(torch.tensor(validation_data, dtype=torch.float32),
                                           torch.tensor(validation_labels, dtype=torch.long))
        val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

        # 训练模型
        for epoch in range(10):
            train_loss = train(model, train_loader, criterion, optimizer, device)
            val_loss, val_accuracy = validate(model, val_loader, criterion, device)
            print(
                f"Epoch {epoch}, Train Loss: {train_loss}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

    acc_list.append(np.mean(all_acc))
    std_list.append(np.std(all_acc))
    end = time.time()
    print(f"Epoch {nb + 1}, Accuracy: {np.mean(all_acc):.2f}%, Std: {np.std(all_acc):.2f}, Time: {end - start:.2f}s")

print(f'Acc_all: {acc_list}')
print(f'Std_all: {std_list}')
print(f"Acc_mean: {np.mean(acc_list):.2f}%")
print(f"Std_mean: {np.std(std_list):.2f}%")