import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
import torch.nn.functional as F
import random
import torch

# 设置随机种子
seed = 3407
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

num_classes = 3
batch_size = 128
img_rows, img_cols, num_chan = 8, 9, 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化用于累积准确率的变量
total_accuracy = 0

falx = np.load('D:/SEED/SEEDzyr/DE0.5s/t6x_89.npy')
y = np.load('D:/SEED/SEEDzyr/DE0.5s/t6y_89.npy')

one_y_1 = np.array([y[:1126]] * 3).reshape((-1,))
one_y_1 = OneHotEncoder(sparse_output=False).fit_transform(one_y_1.reshape(-1, 1))

class SEEDDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 获取数据并调整形状
        x = self.X[idx]
        x = torch.tensor(x, dtype=torch.float32)
        x = x.permute(0, 3, 1, 2)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y

class BaseNetwork(nn.Module):
    def __init__(self, input_dim):
        super(BaseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim[0], out_channels=64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=4, padding=1)
        self.conv6 = nn.Conv2d(1024, 512, kernel_size=4, padding=1)
        self.conv7 = nn.Conv2d(512, 256, kernel_size=4, padding=1)
        self.conv8 = nn.Conv2d(256, 64, kernel_size=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        # 计算卷积层输出的特征数
        conv_output_size = self._get_conv_output((num_chan, img_rows, img_cols))
        self.dense1 = nn.Linear(conv_output_size, 512)
        self.dropout = nn.Dropout(p=0.5)  # 增加Dropout层
        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _get_conv_output(self, shape):
        input = torch.rand(2, *shape)  # 使用更大的批量大小
        output = self._forward_features(input)
        return int(np.prod(output.size()[1:]))  # 忽略批量维度

    def _forward_features(self, x):
        # 通过卷积层和池化层
        x = F.relu(self.conv1(x))  # 使用 F.relu 应用激活函数
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        return self.flatten(x)

    def forward(self, x):
        x = self._forward_features(x)
        x = F.relu(self.dense1(x))  # 在这里应用 ReLU 激活
        x = x.view(x.size(0), 1, -1)
        return x

class MyModel(nn.Module):
    def __init__(self, input_dim):
        super(MyModel, self).__init__()
        self.base_network = BaseNetwork(input_dim)
        self.lstm = nn.LSTM(input_size=512, hidden_size=128, batch_first=True)
        self.out_layer = nn.Linear(128, num_classes)  # 假设输出类别数为num_classes

    def forward(self, x1, x2, x3, x4, x5, x6):
        # 在传递给BaseNetwork之前移除大小为1的维度
        x1 = self.base_network(x1.squeeze(1))
        x2 = self.base_network(x2.squeeze(1))
        x3 = self.base_network(x3.squeeze(1))
        x4 = self.base_network(x4.squeeze(1))
        x5 = self.base_network(x5.squeeze(1))
        x6 = self.base_network(x6.squeeze(1))
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)  # 沿序列维度连接
        x, _ = self.lstm(x)  # 通过LSTM层
        x = self.out_layer(x[:, -1, :])  # 取最后一个时间步的输出
        return x

        # Training loop

def train(model, train_loader, criterion, optimizer, device, current_epoch):
    model.train()
    total_loss = 0
    for batch_idx, batch_data in enumerate(train_loader):
        # 解包批次数据，最后一个元素为目标标签，其余为输入
        *inputs, targets = [data.to(device) for data in batch_data]
        optimizer.zero_grad()
        # 使用*操作符将输入列表解包为独立的参数
        outputs = model(*inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {current_epoch+1}, Average Loss: {avg_loss:.4f}')


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_data in val_loader:
            # 解包批量数据，最后一个元素为目标标签，其余为输入
            *inputs, targets = [data.to(device) for data in batch_data]
            outputs = model(*inputs)
            loss = criterion(outputs, targets)  # 计算损失
            total_loss += loss.item()  # 累积损失
            _, predicted = torch.max(outputs.data, 1)
            if targets.dim() > 1:
                targets = torch.argmax(targets, dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(val_loader)  # 计算平均损失
    return avg_loss, accuracy  # 返回平均损失和准确率

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_data in test_loader:
            # 解包批量数据，最后一个元素为目标标签，其余为输入
            *inputs, targets = [data.to(device) for data in batch_data]
            # 使用*操作符将输入列表解包为独立的参数
            outputs = model(*inputs)
            _, predicted = torch.max(outputs.data, 1)
            # 如果targets是独热编码，需要将其转换为类别索引
            if targets.dim() > 1:
                targets = torch.argmax(targets, dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100 * correct / total
    return accuracy


def pretrain_model(model, train_loader, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch_data in enumerate(train_loader):
            # 将输入数据解包为一个包含所有输入张量的列表和一个标签张量
            *inputs, targets = [data.to(device) for data in batch_data]
            optimizer.zero_grad()
            outputs = model(*inputs)  # 使用 * 操作符将输入列表解包为单独的参数
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Pretrain Epoch: {epoch + 1}, Total Loss: {total_loss:.4f}')

def finetune_model(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    for epoch in range(epochs):
        model.train()  # 确保在每个epoch开始时设置为训练模式
        total_loss = 0
        for batch_idx, batch_data in enumerate(train_loader):
            # 将输入数据解包为一个包含所有输入张量的列表和一个标签张量
            *inputs, targets = [data.to(device) for data in batch_data]
            optimizer.zero_grad()
            outputs = model(*inputs)  # 使用 * 操作符将输入列表解包为单独的参数
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)  # validate函数可能会设置model.eval()
        model.train()  # 重新设置为训练模式，以防validate函数将其设置为评估模式
        print(f'Finetune Epoch: {epoch + 1}, Total Loss: {total_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

all_data = []
all_labels = []
for nb in range(15):
    one_falx_1 = falx[nb * 3:nb * 3 + 3]
    one_falx_1 = one_falx_1.reshape((-1, 6, img_rows, img_cols, 5))
    one_y = one_y_1
    one_falx = one_falx_1[:, :, :, :, 1:5]

    # 添加到总的数据集列表中
    all_data.extend(one_falx)
    all_labels.extend(one_y)

# 转换为numpy数组以便后续处理
all_data = np.array(all_data)
all_data=all_data.transpose((0, 1, 4, 2, 3))
all_labels = np.array(all_labels)

pretrain_data, finetune_data, pretrain_labels, finetune_labels = train_test_split(
    all_data, all_labels, test_size=0.5, random_state=seed
)
pretrain_data_tensor = [torch.tensor(pretrain_data[:, i, :, :, :], dtype=torch.float32) for i in range(6)]
pretrain_labels_tensor = torch.max(torch.tensor(pretrain_labels, dtype=torch.float32), 1)[1]
pretrain_dataset = TensorDataset(*pretrain_data_tensor, pretrain_labels_tensor)
pretrain_loader = DataLoader(pretrain_dataset, batch_size=128, shuffle=True)


# 初始化模型和优化器
model = MyModel(input_dim=(num_chan, img_rows, img_cols)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# 调用预训练函数
pretrain_model(model, pretrain_loader, criterion, optimizer, device, epochs=60)

# 保存预训练的模型参数
torch.save(model.state_dict(), 'SEED_pretrained_model.pth')

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
for fold, (train_index, test_index) in enumerate(kfold.split(finetune_data, finetune_labels.argmax(1))):
    print(f"Training on fold {fold+1}/5...")
    # 首先初始化模型
    model = MyModel(input_dim=(num_chan, img_rows, img_cols)).to(device)

    # 然后加载预训练的模型参数
    model.load_state_dict(torch.load('SEED_pretrained_model.pth'))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    # 使用 train_index 和 test_index 来切分数据集
    x_train, y_train = all_data[train_index], all_labels[train_index]
    x_test, y_test = all_data[test_index], all_labels[test_index]

    # 分割训练集和验证集
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=seed)

    # 转换数据为 torch.float32 类型的张量
    x_train_tensors = [torch.tensor(x_train[:, i, :, :, :], dtype=torch.float32) for i in range(6)]
    x_val_tensors = [torch.tensor(x_val[:, i, :, :, :], dtype=torch.float32) for i in range(6)]
    x_test_tensors = [torch.tensor(x_test[:, i, :, :, :], dtype=torch.float32) for i in range(6)]

    # 转换标签为 PyTorch 张量并使用 torch.max() 转换为类别索引
    _, y_train_indices = torch.max(torch.tensor(y_train, dtype=torch.float32), 1)
    _, y_val_indices = torch.max(torch.tensor(y_val, dtype=torch.float32), 1)
    _, y_test_indices = torch.max(torch.tensor(y_test, dtype=torch.float32), 1)

    # 创建 TensorDataset 和 DataLoader
    train_dataset = TensorDataset(*x_train_tensors, y_train_indices)
    val_dataset = TensorDataset(*x_val_tensors, y_val_indices)
    test_dataset = TensorDataset(*x_test_tensors, y_test_indices)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=128)

    # 调用微调函数
    finetune_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10)

    epoch = 5
    # 训练模型
    for current_epoch in range(epoch):
        train(model, train_loader, criterion, optimizer, device, current_epoch)
        # scheduler.step()
        # 在每个epoch后验证
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        print(f'Epoch {current_epoch + 1}/{epoch}, Validation Accuracy: {val_accuracy:.2f}%')
    # 评估模型
    test_accuracy = test(model, test_loader, device)
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    total_accuracy += test_accuracy

average_accuracy = total_accuracy / 5
print(f'Average Test Accuracy over 5 rounds: {average_accuracy:.2f}%')