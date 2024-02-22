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
import ODConv
from Generate.model import Generator, Discriminator

from ODConv import ODConv2d

input_dim = 4 * 8 * 9

num_classes = 2
batch_size = 128
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

# 加载和处理数据
def load_data(file_path):
    data = scipy.io.loadmat(file_path)['data']
    data = data.reshape(data.shape[0], -1)
    data_tensor = torch.tensor(data, dtype=torch.float)
    return data_tensor

def generate_data(generator, num_samples, num_chan=4, img_rows=8, img_cols=9):
    generator.eval()  # 确保模型处于评估模式
    # 为卷积生成器创建正确形状的噪声
    noise = torch.randn(num_samples, num_chan, img_rows, img_cols, device=device)
    with torch.no_grad():
        generated_data = generator(noise)
    # 不再需要调整形状，因为生成的数据应该已经是正确的形状
    generated_data = generated_data.cpu().numpy()  # 只需将数据移至CPU并转换为NumPy数组
    return generated_data


def assign_labels(num_samples, high_arousal_prob=0.5, high_valence_prob=0.5):
    arousal_labels = np.random.choice([0, 1], size=(num_samples,), p=[1-high_arousal_prob, high_arousal_prob])
    valence_labels = np.random.choice([0, 1], size=(num_samples,), p=[1-high_valence_prob, high_valence_prob])
    return arousal_labels, valence_labels

def pretrain_gan(generator, discriminator, epochs, batch_size, pretrain_data_paths, device):
    """
    预训练GAN模型
    """
    for path_index, path in enumerate(pretrain_data_paths, start=1):
        X_train = load_data(path)
        dataset = TensorDataset(X_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 为生成器和判别器分别定义优化器
        optimizer_G = torch.optim.SGD(generator.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.1)
        scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.1)
        criterion = nn.BCELoss()  # 二元交叉熵损失用于 GAN

        for epoch in range(epochs):
            for _, (real_data,) in enumerate(loader):
                real_data = real_data.to(device)
                batch_size = real_data.size(0)

                # 真实数据标签为1, 假数据标签为0
                real_labels = torch.ones(batch_size, 1, device=device)
                fake_labels = torch.zeros(batch_size, 1, device=device)

                # 训练判别器
                discriminator.zero_grad()
                outputs_real = discriminator(real_data)
                d_loss_real = criterion(outputs_real, real_labels)
                d_loss_real.backward()

                # 训练判别器时生成噪声数据
                noise = torch.randn(batch_size, num_chan, img_rows, img_cols, device=device)
                fake_data = generator(noise)
                outputs_fake = discriminator(fake_data.detach())
                d_loss_fake = criterion(outputs_fake, fake_labels)
                d_loss_fake.backward()
                optimizer_D.step()

                # 训练生成器
                generator.zero_grad()
                outputs = discriminator(fake_data)
                g_loss = criterion(outputs, real_labels)
                g_loss.backward()
                optimizer_G.step()

                # 更新学习率
                scheduler_G.step()
                scheduler_D.step()

                if epoch % 100 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{epochs}] D_loss: {d_loss_real.item() + d_loss_fake.item():.4f} G_loss: {g_loss.item():.4f}")

        # 每训练完一个数据集后的提示信息
        print(f"Completed pretraining with dataset {path_index}/{len(pretrain_data_paths)}: {path}")

def finetune_gan(generator, discriminator, epochs, batch_size, finetune_data_path, device):
    """
    微调GAN模型，使用微调数据路径
    """
    generator.train()
    discriminator.train()
    # 微调GAN的逻辑，与预训练类似，但是针对每个实验者的微调数据路径进行
    X_train = load_data(finetune_data_path)
    dataset = TensorDataset(X_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 为生成器和判别器分别定义优化器
    optimizer_G = torch.optim.SGD(generator.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.1)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.1)
    criterion = nn.BCELoss()  # 二元交叉熵损失用于 GAN

    for epoch in range(epochs):
        for _, (real_data,) in enumerate(loader):
            real_data = real_data.to(device)
            batch_size = real_data.size(0)

            # 真实数据标签为1, 假数据标签为0
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # 训练判别器
            discriminator.zero_grad()
            outputs_real = discriminator(real_data)
            d_loss_real = criterion(outputs_real, real_labels)
            d_loss_real.backward()

            # 训练判别器时生成噪声数据
            noise = torch.randn(batch_size, num_chan, img_rows, img_cols, device=device)
            fake_data = generator(noise)
            outputs_fake = discriminator(fake_data.detach())
            d_loss_fake = criterion(outputs_fake, fake_labels)
            d_loss_fake.backward()
            optimizer_D.step()

            # 训练生成器
            generator.zero_grad()
            outputs = discriminator(fake_data)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            optimizer_G.step()

            # 更新学习率
            scheduler_G.step()
            scheduler_D.step()

            if epoch % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}] D_loss: {d_loss_real.item() + d_loss_fake.item():.4f} G_loss: {g_loss.item():.4f}")

def split_data_for_pretraining_and_finetuning(data_dir, name_index, test_size=0.3):
    pretrain_data_paths = []
    finetune_data_paths = []

    for name in name_index:
        file_path = os.path.join(data_dir, f'DE_s{name}.mat')
        data = scipy.io.loadmat(file_path)['data']

        # 假设数据第一维是样本维
        num_samples = data.shape[0]
        indices = np.arange(num_samples)

        # 分割样本索引
        pretrain_indices, finetune_indices = train_test_split(indices, test_size=test_size, random_state=42)

        # 保存分割后的数据
        pretrain_data_path = os.path.join(data_dir, f'pretrain_DE_s{name}.mat')
        finetune_data_path = os.path.join(data_dir, f'finetune_DE_s{name}.mat')

        scipy.io.savemat(pretrain_data_path, {'data': data[pretrain_indices]})
        scipy.io.savemat(finetune_data_path, {'data': data[finetune_indices]})

        pretrain_data_paths.append(pretrain_data_path)
        finetune_data_paths.append(finetune_data_path)
        print(
            f"Splitting data for subject {name}: Pretrain data saved to {pretrain_data_path}, Finetune data saved to {finetune_data_path}")

    return pretrain_data_paths, finetune_data_paths

# 分割数据用于预训练和微调
pretrain_data_paths, finetune_data_paths = split_data_for_pretraining_and_finetuning(dataset_dir, name_index)

# 实例化GAN模型并移至适当的设备
generator = Generator(num_chan=4, img_rows=8, img_cols=9, input_dim=input_dim).to(device)
discriminator = Discriminator(input_dim).to(device)

# 使用预训练数据集路径预训练GAN
pretrain_gan(generator, discriminator, epochs=1000, batch_size=256, pretrain_data_paths=pretrain_data_paths, device=device)

# 微调每个实验者的GAN模型
for i, finetune_data_path in enumerate(finetune_data_paths):
    print(f"Finetuning GAN with data from {finetune_data_path}")

    # 微调GAN
    finetune_gan(generator, discriminator, epochs=100, batch_size=256, finetune_data_path=finetune_data_path, device=device)
    file_path = os.path.join(dataset_dir, 'DE_s' + name_index[i])
    file = sio.loadmat(file_path)
    data = file['data']

    generated_samples = generate_data(generator, 4800, num_chan, img_rows, img_cols)
    generated_samples = generated_samples.reshape(-1, *data.shape[1:])  # 确保生成数据的形状正确
    data = np.concatenate((data, generated_samples), axis=0)

    y_v = file['valence_labels'][0]  # 加载情感标签
    y_a = file['arousal_labels'][0]  # 加载唤醒度标签

    # 为生成的样本分配标签
    generated_arousal_labels, generated_valence_labels = assign_labels(len(generated_samples))

    # 合并愉悦度和激活度标签
    y_v_combined = np.concatenate((y_v, generated_valence_labels), axis=0)
    y_a_combined = np.concatenate((y_a, generated_arousal_labels), axis=0)

    # 数据重构
    one_falx = data.transpose([0, 2, 3, 1])
    one_falx = one_falx.reshape((-1, t, img_rows, img_cols, num_chan))

    # 标签处理
    one_y_v = np.empty([0, 2])
    one_y_a = np.empty([0, 2])

    for j in range(int(len(y_a_combined) // t)):
        # 调整标签形状，这里直接使用y_v_combined和y_a_combined
        label_v = np.array([1, 0]) if y_v_combined[j * t] == 1 else np.array([0, 1])
        label_a = np.array([1, 0]) if y_a_combined[j * t] == 1 else np.array([0, 1])

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

# 分层交叉验证
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
fold_acc = []
all_acc = []

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, padding=1)  # (4-1)/2 = 1.5，向上取整为 2，但由于PyTorch的限制，取 1
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=4, padding=1)
        self.conv6 = nn.Conv2d(1024, 512, kernel_size=4, padding=1)
        self.conv7 = nn.Conv2d(512, 256, kernel_size=4, padding=1)
        self.conv8 = nn.Conv2d(256, 64, kernel_size=1, padding=0)
        # self.conv9 = ODConv2d(64, 64, kernel_size=1, padding=0, stride=1, groups=1, reduction=0.0625, kernel_num=4)
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

    # def _get_conv_output(self, shape):
    #     # 临时创建一个输入张量来计算卷积层输出的大小
    #     input = torch.rand(1, *shape)
    #     output = self._forward_features(input)
    #     return int(np.prod(output.size()))

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
    def __init__(self, img_size, num_chan):
        super(MyModel, self).__init__()
        self.base_network = BaseNetwork()
        self.lstm = nn.LSTM(input_size=512, hidden_size=128, batch_first=True)
        self.out_layer = nn.Linear(128, 2)

    def forward(self, x1, x2, x3, x4, x5, x6):
        # Process each input through the base network
        x1 = self.base_network(x1)
        x2 = self.base_network(x2)
        x3 = self.base_network(x3)
        x4 = self.base_network(x4)
        x5 = self.base_network(x5)
        x6 = self.base_network(x6)

        # Concatenate along the sequence dimension
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)

        # LSTM layer
        x, _ = self.lstm(x)

        # Fully connected layer
        x = self.out_layer(x[:, -1, :])  # Take the last output of the sequence
        return x


all_data = np.transpose(all_data, (0, 1, 4, 2, 3))
print("Using device:", device)


for fold, (train_index, test_index) in enumerate(kfold.split(all_data, all_labels.argmax(1))):
    print(f"Training on fold {fold+1}/5...")

    img_size = (8, 9)
    num_chan = 4

    # 初始化模型和优化器
    model = MyModel(img_size, num_chan).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    # 使用学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

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


    def validate(model, val_loader, criterion):
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


    def test(model, test_loader):
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

        # 在此处进行模型的训练和验证


    for epoch in range(60):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion)
        print(
                f"Epoch {epoch}, Train Loss: {train_loss}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

    test_accuracy = test(model, test_loader)
    fold_acc.append(test_accuracy)
    print(f"Fold {fold + 1} Test Accuracy: {test_accuracy}")
    acc = test(model, test_loader)
    acc_list.append(acc)


# 计算所有折的平均性能
mean_acc = sum(fold_acc) / len(fold_acc)
print('Average Test Accuracy across folds:', mean_acc)

# 最后，在所有折训练完成后
print('Acc_all:', acc_list)
print('Acc_avg:', np.mean(acc_list))  # 如果acc_list不为空，这将正常工作