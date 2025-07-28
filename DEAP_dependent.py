import numpy as np
import scipy.io as sio
import os
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader
import torch

img_size = (8, 9)
img_rows, img_cols, num_chan = 8, 9, 4
flag = 'v'
t = 4

all_data = []

name_index = ['01', '02', '03', '04', '05', '06', '07', '08', '09',
              '10', '11', '12', '13', '14', '15', '16', '17', '18',
              '19', '20', '21', '22', '23', '24', '25', '26', '27',
              '28', '29', '30', '31', '32']


dataset_dir = "D:/DEAP/with_base_0.5/"

trials_per_subject = 800
trial_size = 20  # 每个trial的大小
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
all_data = np.concatenate(all_data, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
all_data = np.transpose(all_data, (0, 1, 4, 2, 3))
trials_per_subject = 800

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

    def forward(self, x1, x2, x3, x4):
        x1 = self.base_network(x1).unsqueeze(1)
        x2 = self.base_network(x2).unsqueeze(1)
        x3 = self.base_network(x3).unsqueeze(1)
        x4 = self.base_network(x4).unsqueeze(1)
        # x5 = self.base_network(x5).unsqueeze(1)
        # x6 = self.base_network(x6).unsqueeze(1)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        # # Debugging: Print the shape of 'x' to ensure it's correct
        # print("Shape of 'x' before LSTM:", x.shape)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)

        # Process the output of LSTM
        x = self.out_layer(x[:, -1, :])
        return x


print("Using device:", device)


def pretrain_model(model, train_loader, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            inputs = [batch[i].to(device) for i in range(4)]
            targets = batch[4].to(device)
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
            inputs = [batch[i].to(device) for i in range(4)]
            targets = batch[4].to(device)
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
    return avg_loss


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = [batch[i].to(device) for i in range(4)]  # 将输入数据移至 GPU
            targets = batch[4].to(device)  # 将目标移至 GPU
            outputs = model(*inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets).sum().item()
    return total_loss / len(val_loader), correct / len(val_loader.dataset)


def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = [batch[i].to(device) for i in range(4)]
            targets = batch[6].to(device)
            outputs = model(*inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = correct / total
    return accuracy, all_preds, all_targets


num_samples = all_data.shape[0]
indices = np.random.permutation(num_samples)

# 五折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 初始化性能记录列表
acc_list = []
all_predictions = []
all_true_labels = []

for fold, (train_val_index, test_index) in enumerate(kf.split(indices)):
    print(f"Processing fold {fold + 1}/5...")
    # 分割训练验证集和测试集
    train_val_indices = indices[train_val_index]
    test_indices = indices[test_index]

    # 分割训练验证集为预训练、微调和验证
    num_train_val = len(train_val_indices)
    num_pretrain = int(num_train_val * 0.7)
    num_finetune = int(num_train_val * 0.2)
    num_val = int(num_train_val * 0.1)

    # 随机打乱训练验证索引
    np.random.shuffle(train_val_indices)
    pretrain_indices = train_val_indices[:num_pretrain]
    finetune_indices = train_val_indices[num_pretrain:num_pretrain + num_finetune]
    val_indices = train_val_indices[num_pretrain + num_finetune:num_pretrain + num_finetune + num_val]

    # 根据选择的索引划分数据集
    x_pretrain, y_pretrain = all_data[pretrain_indices], all_labels[pretrain_indices]
    x_finetune, y_finetune = all_data[finetune_indices], all_labels[finetune_indices]
    x_val, y_val = all_data[val_indices], all_labels[val_indices]
    x_test, y_test = all_data[test_indices], all_labels[test_indices]

    # 数据转换为 PyTorch 张量
    x_pretrain_tensors = [torch.tensor(x_pretrain[:, i, :, :, :], dtype=torch.float32) for i in range(4)]
    x_finetune_tensors = [torch.tensor(x_finetune[:, i, :, :, :], dtype=torch.float32) for i in range(4)]
    x_val_tensors = [torch.tensor(x_val[:, i, :, :, :], dtype=torch.float32) for i in range(4)]
    x_test_tensors = [torch.tensor(x_test[:, i, :, :, :], dtype=torch.float32) for i in range(4)]

    y_pretrain_tensor = torch.tensor(np.argmax(y_pretrain, axis=1), dtype=torch.long)
    y_finetune_tensor = torch.tensor(np.argmax(y_finetune, axis=1), dtype=torch.long)
    y_val_tensor = torch.tensor(np.argmax(y_val, axis=1), dtype=torch.long)
    y_test_tensor = torch.tensor(np.argmax(y_test, axis=1), dtype=torch.long)

    # 创建 DataLoader
    pretrain_loader = DataLoader(TensorDataset(*x_pretrain_tensors, y_pretrain_tensor), batch_size=128, shuffle=True)
    finetune_loader = DataLoader(TensorDataset(*x_finetune_tensors, y_finetune_tensor), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(*x_val_tensors, y_val_tensor), batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(*x_test_tensors, y_test_tensor), batch_size=128, shuffle=True)

    model = MyModel(img_size, num_chan).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    assert set(test_index).isdisjoint(set(train_val_index)), "Test indices overlap with training/validation indices."
    assert num_pretrain + num_finetune + num_val == num_train_val, "Sum of split parts does not equal total training/validation size."
    # 训练和测试模型
    for epoch in range(50):
        pretrain_model(model, pretrain_loader, criterion, optimizer, device, epochs=1)
        finetune_model(model, finetune_loader, val_loader, criterion, optimizer, device, epochs=1)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

    test_accuracy, predictions, targets = test(model, test_loader, device)
    acc_list.append(test_accuracy)
    all_predictions.extend(predictions)
    all_true_labels.extend(targets)
    print(f"Fold {fold + 1} Test Accuracy: {test_accuracy}")

# 计算和打印全局性能指标
global_accuracy = accuracy_score(all_true_labels, all_predictions)
global_f1_score = f1_score(all_true_labels, all_predictions, average='weighted')
global_cm = confusion_matrix(all_true_labels, all_predictions)
mean_accuracy = np.mean(acc_list)
std_accuracy = np.std(acc_list)
print(f"Average Test Accuracy: {mean_accuracy}")
print(f"Standard Deviation of Test Accuracies: {std_accuracy}")
print(f"Global Accuracy: {global_accuracy}")
print("Global F1 Score:", global_f1_score)
print("Global Confusion Matrix:\n", global_cm)

# 保存结果
with open('exercise/DEAP_dependent_v_results.txt', 'w') as f:
    f.write(f"Average Test Accuracy: {mean_accuracy}\n")
    f.write(f"Standard Deviation of Test Accuracies: {std_accuracy}\n")
    f.write(f"Global F1 Score: {global_f1_score}\n")
np.savetxt('exercise/DEAP_dependent_v_confusion_matrix.csv', global_cm, delimiter=',', fmt='%d')