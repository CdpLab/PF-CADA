import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# class Generator(nn.Module):
#     def __init__(self, input_dim):
#         super(Generator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.LeakyReLU(0.2),
#             nn.BatchNorm1d(256, momentum=0.8),
#             nn.Linear(256, 512),
#             nn.LeakyReLU(0.2),
#             nn.BatchNorm1d(512, momentum=0.8),
#             nn.Linear(512, 1024),
#             nn.LeakyReLU(0.2),
#             nn.BatchNorm1d(1024, momentum=0.8),
#             nn.Linear(1024, input_dim),
#             nn.Tanh()
#         )
#
#     def forward(self, x):
#         return self.model(x)
class Generator(nn.Module):
    def __init__(self, num_chan, img_rows, img_cols, input_dim):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=4, padding=1)
        self.conv6 = nn.Conv2d(1024, 512, kernel_size=4, padding=1)
        self.conv7 = nn.Conv2d(512, 256, kernel_size=4, padding=1)
        self.conv8 = nn.Conv2d(256, 64, kernel_size=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        self.conv_output_size = self._get_conv_output((num_chan, img_rows, img_cols))
        self.dense1 = nn.Linear(self.conv_output_size, 512)
        self.dropout = nn.Dropout(p=0.5)  # 增加Dropout层
        self.lstm = nn.LSTM(input_size=512, hidden_size=128, batch_first=True)
        self.final_ffn = nn.Sequential(
            nn.Linear(128, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024, momentum=0.8),
            nn.Linear(1024, input_dim),
            nn.Tanh()
        )

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
        input = torch.rand(2, *shape)
        output = self._forward_features(input)
        return int(np.prod(output.size()[1:]))

    def _forward_features(self, x):
        x = F.relu(self.conv1(x))
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
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        x = x.view(x.size(0), 1, -1)  # LSTM层需要的输入维度为(batch_size, seq_len, features)
        x, (hn, cn) = self.lstm(x)
        # 取LSTM的最后一层输出
        x = x[:, -1, :]
        x = self.final_ffn(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# # 训练 GAN
# # 假定 generator 和 discriminator 已经定义并初始化
# def train_gan(generator, discriminator, epochs, batch_size, data_path):
#     X_train = load_data(data_path)  # 假定这个函数会返回正确格式的数据张量
#     dataset = TensorDataset(X_train)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
#     # 为生成器和判别器分别定义优化器
#     optimizer_G = torch.optim.SGD(generator.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
#     optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
#     scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.1)
#     scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.1)
#     criterion = nn.BCELoss()  # 二元交叉熵损失用于 GAN
#
#     for epoch in range(epochs):
#         for _, (real_data,) in enumerate(loader):
#             real_data = real_data.to(device)
#             batch_size = real_data.size(0)
#
#             # 真实数据标签为1, 假数据标签为0
#             real_labels = torch.ones(batch_size, 1, device=device)
#             fake_labels = torch.zeros(batch_size, 1, device=device)
#
#             # 训练判别器
#             discriminator.zero_grad()
#             outputs_real = discriminator(real_data)
#             d_loss_real = criterion(outputs_real, real_labels)
#             d_loss_real.backward()
#
#             noise = torch.randn(batch_size, input_dim, device=device)
#             fake_data = generator(noise)
#             outputs_fake = discriminator(fake_data.detach())
#             d_loss_fake = criterion(outputs_fake, fake_labels)
#             d_loss_fake.backward()
#             optimizer_D.step()
#
#             # 训练生成器
#             generator.zero_grad()
#             outputs = discriminator(fake_data)
#             g_loss = criterion(outputs, real_labels)
#             g_loss.backward()
#             optimizer_G.step()
#
#             # 更新学习率
#             scheduler_G.step()
#             scheduler_D.step()
#
#             if epoch % 100 == 0:
#                 print(f"Epoch [{epoch+1}/{epochs}] D_loss: {d_loss_real.item()+d_loss_fake.item():.4f} G_loss: {g_loss.item():.4f}")


# 训练所有文件
# def train_all_files(epochs, batch_size, file_index, generator, discriminator):
#         file_path = f'D:/DEAP/with_base_0.5/DE_s{file_index:02d}.mat'
#         print(f"Training on file: {file_path}")
#         train_gan(epochs=epochs, batch_size=batch_size, data_path=file_path, generator=generator, discriminator=discriminator)
#         print(f"Completed training on file: {file_path}")

# for i in range(len(name_index)):
#     file_path = os.path.join(dataset_dir, 'DE_s'+name_index[i])
#
#     file = sio.loadmat(file_path)
#     data = file['data']
#
#     generated_samples = generate_data(generator, 4800, input_dim)
#     generated_samples = generated_samples.reshape(-1, *data.shape[1:])  # 确保生成数据的形状正确
#     data = np.concatenate((data, generated_samples), axis=0)
#
#     y_v = file['valence_labels'][0]  # 加载情感标签
#     y_a = file['arousal_labels'][0]  # 加载唤醒度标签
#
#     # 为生成的样本分配标签
#     generated_arousal_labels, generated_valence_labels = assign_labels(len(generated_samples))
#
#     # 合并愉悦度和激活度标签
#     y_v_combined = np.concatenate((y_v, generated_valence_labels), axis=0)
#     y_a_combined = np.concatenate((y_a, generated_arousal_labels), axis=0)
#
#     # 数据重构
#     one_falx = data.transpose([0, 2, 3, 1])
#     one_falx = one_falx.reshape((-1, t, img_rows, img_cols, num_chan))
#
#     # 标签处理
#     one_y_v = np.empty([0, 2])
#     one_y_a = np.empty([0, 2])
#
#     for j in range(int(len(y_a_combined) // t)):
#         # 调整标签形状，这里直接使用y_v_combined和y_a_combined
#         label_v = np.array([1, 0]) if y_v_combined[j * t] == 1 else np.array([0, 1])
#         label_a = np.array([1, 0]) if y_a_combined[j * t] == 1 else np.array([0, 1])
#
#         # 堆叠标签
#         one_y_v = np.vstack((one_y_v, label_v))
#         one_y_a = np.vstack((one_y_a, label_a))
#
#     if flag == 'v':
#         one_y = one_y_v
#     else:
#         one_y = one_y_a
#
#         # 将数据和标签添加到总集合
#     all_data.append(one_falx)
#     all_labels.append(one_y)