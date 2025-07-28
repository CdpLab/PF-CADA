import numpy as np
import scipy
import scipy.io as sio
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch


img_size = (8, 9)
img_rows, img_cols, num_chan = 8, 9, 4
flag = 'a'
t = 6


acc_list = []
std_list = []
all_acc = []

all_data = []
all_valence_labels = []
all_arousal_labels = []
subject_labels = []

name_index = ['01', '02', '03', '04', '05', '06', '07', '08', '09',
              '10', '11', '12', '13', '14', '15', '16', '17', '18',
              '19', '20', '21', '22', '23', '24', '25', '26', '27',
              '28', '29', '30', '31', '32']


dataset_dir = "D:/DEAP/with_base_0.5/"

all_data = []
all_labels = []

seed = 3407
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for i in range(len(name_index)):
    file_path = os.path.join(dataset_dir, 'DE_s'+name_index[i])
    file = sio.loadmat(file_path)
    data = file['data']
    y_v = file['valence_labels'][0]  # 加载情感标签
    y_a = file['arousal_labels'][0]  # 加载唤醒度标签

    # 数据重构
    one_falx = data.transpose([0, 2, 3, 1])
    one_falx = one_falx.reshape((-1, t, img_rows, img_cols, num_chan))

    # 标签处理
    one_y_v = np.empty([0, 2])
    one_y_a = np.empty([0, 2])

    for j in range(int(len(y_a) // t)):
        # 调整标签形状
        label_v = np.array([1, 0]) if y_v[j * t] == 1 else np.array([0, 1])
        label_a = np.array([1, 0]) if y_a[j * t] == 1 else np.array([0, 1])

        # 堆叠标签
        one_y_v = np.vstack((one_y_v, label_v))
        one_y_a = np.vstack((one_y_a, label_a))

    if flag == 'v':
        one_y = one_y_v
    else:
        one_y = one_y_a

    # 将数据和标签添加到总集合
    all_data.append(one_falx)
    all_labels.append(one_y)

# 转换为 NumPy 数组
loso_data = np.concatenate(all_data, axis=0)
loso_labels = np.concatenate(all_labels, axis=0)
loso_data = np.transpose(loso_data, (0, 1, 4, 2, 3))
trials_per_subject = 800
data_by_subject = [loso_data[i*trials_per_subject:(i+1)*trials_per_subject] for i in range(32)]
labels_by_subject = [loso_labels[i*trials_per_subject:(i+1)*trials_per_subject] for i in range(32)]
# 前31个主体的数据和标签组合成一个数组
pretrainData = np.concatenate(data_by_subject[:-1], axis=0)
pretrainLabels = np.concatenate(labels_by_subject[:-1], axis=0)
# 最后一个主体的数据和标签组成一个数组
loso_test_data = data_by_subject[-1]
loso_test_labels = labels_by_subject[-1]


# 分层交叉验证
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
fold_acc = []
all_acc = []

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(4, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=4, padding=1)
        self.conv6 = nn.Conv2d(1024, 512, kernel_size=4, padding=1)
        self.conv7 = nn.Conv2d(512, 256, kernel_size=4, padding=1)
        self.conv8 = nn.Conv2d(256, 64, kernel_size=1, padding=0)
        self.flatten = nn.Flatten()

        # 特征金字塔
        self.feature_pyramid = nn.Conv2d(1152, 512, kernel_size=1)  # 用于合并特征的1x1卷积

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _forward_features(self, x):
        c1 = F.relu(self.conv1(x))
        c2 = F.relu(self.conv2(c1))
        c3 = F.relu(self.conv3(c2))
        c4 = F.relu(self.conv4(c3))
        c5 = F.relu(self.conv5(c4))
        c6 = F.relu(self.conv6(c5))
        c7 = F.relu(self.conv7(c6))
        c8 = F.relu(self.conv8(c7))

        # 金字塔特征合并
        up_c5 = F.interpolate(c5, size=c2.size()[2:], mode='bilinear', align_corners=False)
        merged = torch.cat([c2, up_c5], dim=1)
        pyramid_feature = F.relu(self.feature_pyramid(merged))

        return self.flatten(pyramid_feature)

    def forward(self, x):
        x = self._forward_features(x)
        return x


class MyModel(nn.Module):
    def __init__(self, img_size, num_chan):
        super(MyModel, self).__init__()
        self.base_network = BaseNetwork()
        self.lstm = nn.LSTM(input_size=28672, hidden_size=128, batch_first=True)
        self.out_layer = nn.Linear(128, 2)

    def forward(self, x1, x2, x3, x4, x5, x6):
        x1 = self.base_network(x1).unsqueeze(1)
        x2 = self.base_network(x2).unsqueeze(1)
        x3 = self.base_network(x3).unsqueeze(1)
        x4 = self.base_network(x4).unsqueeze(1)
        x5 = self.base_network(x5).unsqueeze(1)
        x6 = self.base_network(x6).unsqueeze(1)

        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)

        # # Debugging: Print the shape of 'x' to ensure it's correct
        # print("Shape of 'x' before LSTM:", x.shape)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)

        # Process the output of LSTM
        x = self.out_layer(x[:, -1, :])
        return x
    def get_lstm_features(self, x1, x2, x3, x4, x5, x6):
        x1 = self.base_network(x1)
        x2 = self.base_network(x2)
        x3 = self.base_network(x3)
        x4 = self.base_network(x4)
        x5 = self.base_network(x5)
        x6 = self.base_network(x6)

        # Concatenate along the sequence dimension
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        print(x.shape)

        # LSTM layer
        x, _ = self.lstm(x)
        return x[:, -1, :]  # 只返回LSTM的最后输出，不通过全连接层

def get_lstm_features(model, data_loader, device):
    model.eval()
    all_features = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            inputs = [batch[i].to(device) for i in range(6)]  # 将输入数据移至 GPU
            features = model.get_lstm_features(*inputs)
            all_features.append(features.cpu().numpy())  # 把特征从GPU转到CPU并转换为numpy数组
    return np.concatenate(all_features, axis=0)  # 合并所有特征

print("Using device:", device)

def pretrain_model(model, train_loader, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            inputs = [batch[i].to(device) for i in range(6)]
            targets = batch[6].to(device)
            optimizer.zero_grad()
            outputs = model(*inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # 打印每轮训练的提示信息
        print(f'Pretrain Epoch: {epoch + 1}, Total Loss: {total_loss:.4f}')

def finetune_model(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    for epoch in range(epochs):
        model.train()  # 确保在每个epoch开始时设置为训练模式
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            inputs = [batch[i].to(device) for i in range(6)]
            targets = batch[6].to(device)
            optimizer.zero_grad()
            outputs = model(*inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss, val_accuracy = validate(model, val_loader, criterion, device)  # validate函数可能会设置model.eval()
        model.train()  # 重新设置为训练模式，以防validate函数将其设置为评估模式
        print(f'Finetune Epoch: {epoch + 1}, Total Loss: {total_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        inputs = [batch[i].to(device) for i in range(6)]  # 将输入数据移至 GPU
        targets = batch[6].to(device)  # 将目标移至 GPU
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

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = [batch[i].to(device) for i in range(6)]  # 将输入数据移至 GPU
            targets = batch[6].to(device)  # 将目标移至 GPU
            outputs = model(*inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets).sum().item()
    return total_loss / len(val_loader), correct / len(val_loader.dataset)


def test(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs = [batch[i].to(device) for i in range(6)]  # 将输入数据移至 GPU
            targets = batch[6].to(device)  # 将目标移至 GPU
            outputs = model(*inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets).sum().item()
    return correct / len(test_loader.dataset)

def test_and_save_results(model, test_loader, device, result_file):
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch_data in test_loader:
            # 解包批量数据，最后一个元素为目标标签，其余为输入
            *inputs, targets = [data.to(device) for data in batch_data]
            outputs = model(*inputs)
            _, predicted = torch.max(outputs.data, 1)

            # 独热编码，需要将其转换为类别索引
            if targets.dim() > 1:
                targets = torch.argmax(targets, dim=1)

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # 转换为numpy数组
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)

    # 计算准确率
    accuracy = np.sum(all_targets == all_predictions) / len(all_targets)

    # 保存结果到文件
    np.savez(result_file, true_labels=all_targets, predicted_labels=all_predictions)

    return accuracy

pretrain_data, finetune_data, pretrain_labels, finetune_labels = train_test_split(
    loso_data, loso_labels, test_size=0.2, random_state=seed
)

pretrain_data_tensor = [torch.tensor(pretrain_data[:, i, :, :, :], dtype=torch.float32) for i in range(6)]
pretrain_labels_tensor = torch.max(torch.tensor(pretrain_labels, dtype=torch.float32), 1)[1]
pretrain_dataset = TensorDataset(*pretrain_data_tensor, pretrain_labels_tensor)
pretrain_loader = DataLoader(pretrain_dataset, batch_size=128, shuffle=True)


# 初始化模型和优化器
model = MyModel(img_size, num_chan).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# 调用预训练函数
pretrain_model(model, pretrain_loader, criterion, optimizer, device, epochs=80)

# 保存预训练的模型参数
torch.save(model.state_dict(), 'SI_DEAP_pretrained_model.pth')

best_accuracy = 0.0
best_model_state = None
fold_acc = []
acc_list = []
fold_features = []
for fold, (train_index, test_index) in enumerate(kfold.split(finetune_data, finetune_labels.argmax(1))):
    print(f"Training on fold {fold+1}/5...")

    img_size = (8, 9)
    num_chan = 4

    # 首先初始化模型
    model = MyModel(img_size, num_chan).to(device)

    # 然后加载预训练的模型参数
    model.load_state_dict(torch.load('SI_DEAP_pretrained_model.pth'))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    # 使用学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # 使用 train_index 和 test_index 来切分数据集
    x_train, y_train = loso_data[train_index], loso_labels[train_index]
    x_test, y_test = loso_data[test_index], loso_labels[test_index]

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
    finetune_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=80)

    # 在此处进行模型的训练和验证
    for epoch in range(20):
        # 训练模型
        train_loss = train(model, train_loader, criterion, optimizer, device)
        # 验证模型
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}, Train Loss: {train_loss}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
        # 更新最佳模型状态
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()

    test_accuracy = test(model, test_loader, device)
    fold_acc.append(test_accuracy)
    print(f"Fold {fold + 1} Test Accuracy: {test_accuracy}")
    acc = test(model, test_loader, device)
    acc_list.append(acc)

    # # 获取这个fold训练后的特征
    # fold_train_features = get_lstm_features(model, train_loader, device)
    # fold_features.append(fold_train_features)  # 存储特征

# np.save(f'fold_train_features_a.npy', fold_features)
torch.save(best_model_state, 'best_SI_DEAP_model.pth')

# 计算所有折的平均性能
mean_acc = sum(fold_acc) / len(fold_acc)
print('Average Test Accuracy across folds:', mean_acc)

# 最后，在所有折训练完成后
print('Acc_all:', acc_list)
print('Acc_avg:', np.mean(acc_list))

model.load_state_dict(best_model_state)

# 准备最后一个主体的数据
loso_test_data_tensors = [torch.tensor(loso_test_data[:, i, :, :, :], dtype=torch.float32) for i in range(6)]
_, loso_test_labels_tensor = torch.max(torch.tensor(loso_test_labels, dtype=torch.float32), 1)
loso_test_dataset = TensorDataset(*loso_test_data_tensors, loso_test_labels_tensor)
loso_test_loader = DataLoader(loso_test_dataset, batch_size=128)

# 测试最佳模型
# result_file_path = 'DEAP_result_file_a.npz'  # 修改为所需的路径
test_accuracy = test(model, loso_test_loader, device)
print(f"Test Accuracy for the last subject using the best model: {test_accuracy}")

# # 获取测试集的特征
# test_features = get_lstm_features(model, loso_test_loader, device)
# # 保存测试特征
# np.save('test_features_a.npy', test_features)


