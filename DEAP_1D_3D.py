import time
from sklearn import preprocessing
from scipy.signal import butter, lfilter
import scipy.io as sio
import numpy as np
import os
import math
import sys

# 这个 `read_file` 函数用于读取以MATLAB格式存储的文件，并提取其中的数据和标签信息。函数的主要步骤如下：
#
# 1. **加载MATLAB文件：**
#    - 使用 `scipy` 库中的 `sio.loadmat` 函数加载指定的MATLAB文件。
#
# 2. **提取数据和标签：**
#    - 从加载的MATLAB文件中提取 `trial_data`（试验数据）和 `base_data`（基础数据）。
#    - 提取情感标签，将其存储在 `arousal_labels` 和 `valence_labels` 中。
#
# 3. **返回结果：**
#    - 返回提取的 `trial_data`、`base_data`、`arousal_labels` 和 `valence_labels`。
#
# 这个函数的目的是方便地从MATLAB文件中获取需要的信息，以便在进一步的处理中使用。
def read_file(file):
    file = sio.loadmat(file)
    trial_data = file['data']
    base_data = file["base_data"]
    return trial_data, base_data, file["arousal_labels"], file["valence_labels"]


# 这个 `get_vector_deviation` 函数计算两个向量 `vector1` 和 `vector2` 的差异。函数返回的是这两个向量对应元素的差值。
#
# 具体来说，如果 `vector1` 和 `vector2` 是NumPy数组或类似的可运算结构，那么这个函数将返回一个新的数组，其中的每个元素都是 `vector1` 中相应位置元素减去 `vector2` 中相应位置元素的结果。这实际上是执行了向量的逐元素减法操作。
#
# 这个函数的目的可能是用于计算两个向量之间的差异，但在实际情况下，更详细的目的和上下文可能需要查看调用该函数的代码。
def get_vector_deviation(vector1, vector2):
    return vector1 - vector2

# 这个 `get_dataset_deviation` 函数用于计算试验数据集 `trial_data` 中每个数据点与对应的基准数据集 `base_data` 之间的差异。它返回一个新的数据集 `new_dataset`，其中每个记录都是试验数据与对应基准数据的差异。
#
# 具体来说，函数使用了一个循环，遍历试验数据集中的每个数据点（共有 4800 个数据点）。对于每个数据点，它找到对应的基准数据点（使用整除运算 `i // 120` 定位到 40 个基准数据点中的一个，注意最后一个数据点会与最后一个基准数据点对应）。然后，它调用之前定义的 `get_vector_deviation` 函数计算试验数据点与基准数据点的差异，并将结果添加到 `new_dataset` 中。
#
# 最终，函数返回了一个形状为 (4800, 128) 的新数据集，其中包含了每个试验数据点与对应基准数据点的差异。
def get_dataset_deviation(trial_data, base_data):
    new_dataset = np.empty([0, 128])
    for i in range(0, 4800):
        base_index = i // 120
        # print(base_index)
        base_index = 39 if base_index == 40 else base_index  # 最后一个值4800//120=40,这句代码意思是把4800这个点base_index的值也归到39
        new_record = get_vector_deviation(trial_data[i], base_data[base_index]).reshape(1, 128)
        # print(new_record.shape)
        new_dataset = np.vstack([new_dataset, new_record])
    # print("new shape:",new_dataset.shape)
    return new_dataset

# 这段代码实现了将一维数据映射到二维矩阵的过程，其中二维矩阵表示电极在头皮上的布局。具体而言，它将一维数据按照一定的规则填充到一个8x9的矩阵中。
#
# 这里有一个8x9的电极布局，其中0表示没有电极的位置，而其他数字表示相应位置上的电极。对应的一维数据中的元素通过规则填充到相应位置。
#
# 例如，`data[0]`表示电极1的数值，`data[1]`表示电极2的数值，以此类推。
#
# 这个函数返回一个8x9的二维矩阵，其中每个元素是根据一维数据中的值填充的。
def data_1Dto2D(data, Y=8, X=9):  # 转化成8*9的电极坐标并对应填入数据
    data_2D = np.zeros([Y, X])
    data_2D[0] = (0, 0, data[1], data[0], 0, data[16], data[17], 0, 0)
    data_2D[1] = (data[3], 0, data[2], 0, data[18], 0, data[19], 0, data[20])
    data_2D[2] = (0, data[4], 0, data[5], 0, data[22], 0, data[21], 0)
    data_2D[3] = (data[7], 0, data[6], 0, data[23], 0, data[24], 0, data[25])
    data_2D[4] = (0, data[8], 0, data[9], 0, data[27], 0, data[26], 0)
    data_2D[5] = (data[11], 0, data[10], 0, data[15], 0, data[28], 0, data[29])
    data_2D[6] = (0, 0, 0, data[12], 0, data[30], 0, 0, 0)
    data_2D[7] = (0, 0, 0, data[13], data[14], data[31], 0, 0, 0)
    # return shape:8*9
    return data_2D

# 这段代码实现了对 EEG 数据的预处理过程。具体而言，它包括以下步骤：
#
# 1. **读取数据：** 通过调用 `read_file` 函数，从指定路径 `path` 读取 EEG 数据，其中包括试验数据 (`trial_data`)、基准数据 (`base_data`) 以及情感标签 (`arousal_labels` 和 `valence_labels`)。
#
# 2. **数据标准化：** 如果 `y_n` 参数为 "yes"，则对试验数据和基准数据进行标准化处理。标准化是一种常见的数据预处理方法，旨在使数据在特征上具有相似的尺度，有助于提高模型训练的性能。这里使用 `preprocessing.scale` 函数对数据进行标准化。
#
# 3. **生成 3D 数据：** 将标准化后的数据按照一定的规则转换成一个 3D 数据集 (`data_3D`)，其中每个数据点对应一个 4x8x9 的三维矩阵。这个转换过程涉及将每个128维的向量转化为一个 4x9x9 的立方体矩阵。
#
# 4. **返回结果：** 返回最终的 3D 数据集 (`data_3D`)，以及对应的情感标签 (`arousal_labels` 和 `valence_labels`)。
#
# 整个函数的目的是为了将原始 EEG 数据进行预处理，使其适用于后续的深度学习模型。
def pre_process(path, y_n):
    # DE feature vector dimension of each band

    # 初始化一个空的3D数组，形状为(0, 8, 9)
    data_3D = np.empty([0, 8, 9])
    # 定义子向量的长度为32
    sub_vector_len = 32
    # 读取文件，获取试验数据、基准数据以及情感标签
    trial_data, base_data, arousal_labels, valence_labels = read_file(path)

    # 如果y_n为"yes"，对试验数据和基准数据进行标准化处理,调用 get_dataset_deviation 函数获取试验数据和基准数据之间的差异
    if y_n == "yes":
        data = get_dataset_deviation(trial_data, base_data)
        data = preprocessing.scale(data, axis=1, with_mean=True, with_std=True, copy=True)
    else:
        # 否则，对试验数据进行标准化处理
        data = preprocessing.scale(trial_data, axis=1, with_mean=True, with_std=True, copy=True)
    # convert 128 vector ---> 4*9*9 cube

    # 打印试验数据的形状
    print(data.shape) # 4800*128
    # 遍历每个试验数据的向量
    for vector in data:
        # 遍历每个频段（band）
        for band in range(0, 4):
            # 提取子向量，调用 data_1Dto2D 函数将子向量转化成二维数据（8*9的电极坐标）
            data_2D_temp = data_1Dto2D(vector[band * sub_vector_len:(band + 1) * sub_vector_len])
            # 将转化后的二维数据重新变形为 (1, 8, 9) 的形状
            data_2D_temp = data_2D_temp.reshape(1, 8, 9)
            # print("data_2d_temp shape:",data_2D_temp.shape)
            # 将二维数据纵向堆叠到 3D 数组 data_3D 中
            data_3D = np.vstack([data_3D, data_2D_temp])
    # 将最终得到的 3D 数据重新变形为 (-1, 4, 8, 9) 的形状
    data_3D = data_3D.reshape(-1, 4, 8, 9)
    # 打印最终处理后的数据形状
    print("final data shape:", data_3D.shape)  # 4800,4,8,9
    # 返回最终处理后的数据、arousal 标签和 valence 标签
    return data_3D, arousal_labels, valence_labels


if __name__ == '__main__':
    # 设置数据集目录路径为D: / DEAP / all_0.5 /。
    dataset_dir = "D:/DEAP/all_0.5/"
    # 设置是否使用基准数据的标志，这里设为"yes"表示使用基准数据
    use_baseline = "yes"

    # 根据是否使用基准数据，设置结果保存目录为D: / DEAP / with_base_0.5 / 或D: / DEAP / without_base_0.5 /。
    if use_baseline == "yes":
        result_dir = "D:/DEAP/with_base_0.5/"
        # 如果结果保存目录不存在，创建该目录。
        if os.path.isdir(result_dir) == False:
            os.makedirs(result_dir)
    else:
        result_dir = "D:/DEAP/without_base_0.5/"
        if os.path.isdir(result_dir) == False:
            os.makedirs(result_dir)

    # 遍历数据集目录下的每个文件。
    for file in os.listdir(dataset_dir):
        print("processing: ", file, "......")
        # 构建当前文件的完整路径。
        file_path = os.path.join(dataset_dir, file)

        # 调用pre_process函数，处理当前文件，得到处理后的数据、arousal标签和valence标签。
        data, arousal_labels, valence_labels = pre_process(file_path, use_baseline)
        print("final shape:", data.shape)

        # 使用sio.savemat将处理后的数据、arousal标签和valence标签保存为.mat文件，文件名与原文件相同，保存在结果目录下。
        sio.savemat(result_dir + file,
                    {"data": data, "valence_labels": valence_labels, "arousal_labels": arousal_labels})
        # break