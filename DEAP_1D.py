import os
import sys
import math
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn import preprocessing
from scipy.signal import butter, lfilter

# 巴特沃斯（Butterworth）滤波器是一种具有最大平坦幅度响应的低通滤波器，保证了信号的原始值，不会因为滤波被衰减。
# ｎ阶越大，其幅频响应就越逼近理想情况。（这个还未做，有兴趣的同学可以在我的基础上往下深入探究一下）
# lowcut 和 highcut：带通滤波器的截止频率范围（低频和高频）。
# fs：信号的采样频率。
# order：滤波器的阶数，代表滤波器的复杂度
def butter_bandpass(lowcut, highcut, fs, order=5):
    # nyq 是 Nyquist 频率的缩写
    # 在数字信号处理中，根据 Nyquist 定理，为了完整还原原始信号，采样频率必须至少是信号中最高频率的两倍。因此，Nyquist 频率（nyq）等于采样频率（fs）的一半
    # 这里的 fs 是信号的采样频率，0.5 * fs 就是 Nyquist 频率。在巴特沃斯滤波器的设计中，nyq 的值用于归一化截止频率（lowcut 和 highcut），以便更方便地在滤波器设计中使用。  
    nyq = 0.5 * fs
    # 在巴特沃斯滤波器设计中，low 和 high 表示的是截止频率的相对值，它们的取值范围在 0 到 1 之间。这些相对值会根据 Nyquist 频率进行归一化，以便适应不同采样频率的信号。
    # 具体而言：
    # low：低通滤波器的截止频率相对于 Nyquist 频率的相对位置。如果 low 设为 0.1，而采样频率为 100 Hz，那么低通滤波器的截止频率就是 0.1 * 0.5 * 100 = 5 Hz。
    # high：高通滤波器的截止频率相对于 Nyquist 频率的相对位置。如果 high 设为 0.2，而采样频率为 100 Hz，那么高通滤波器的截止频率就是 0.2 * 0.5 * 100 = 10 Hz。
    # 这种相对位置的设置允许在不同采样频率下使用相同的截止频率相对值，从而提高了滤波器设计的通用性。    
    low = lowcut / nyq
    high = highcut / nyq
    # 在巴特沃斯滤波器中，`b` 和 `a` 是滤波器的分子和分母系数。这些系数用于描述滤波器的传递函数。具体来说：
    # `b` 是分子系数，表示传递函数的分子部分的系数。
    # `a` 是分母系数，表示传递函数的分母部分的系数。
    # 这两者的值是根据滤波器的设计参数和类型计算得出的。在巴特沃斯滤波器的设计中，通常使用 `scipy.signal.butter` 函数来计算这些系数。`b` 和 `a` 的值会根据滤波器的阶数、截止频率等参数而变化。
    # 在你提供的代码中，`b, a = butter(order, [low, high], btype='band')` 使用了 `butter` 函数，其中 `order` 是滤波器的阶数，而 `[low, high]` 是截止频率的相对值。这个函数返回的 `b` 和 `a` 就是滤波器的系数。
    b, a = butter(order, [low, high], btype='band')
    return b, a

# 这段代码实现了一个带通滤波器的函数 `butter_bandpass_filter`，用于对输入的信号 `data` 进行带通滤波操作。函数调用了之前提到的 `butter_bandpass` 函数，该函数返回带通滤波器的系数 `b` 和 `a`。接着，使用 `scipy.signal.lfilter` 函数应用这些系数来对输入信号进行滤波操作。
#
# 具体步骤如下：
#
# 1. `b, a = butter_bandpass(lowcut, highcut, fs, order=order)`：调用 `butter_bandpass` 函数获取带通滤波器的系数 `b` 和 `a`。
#
# 2. `y = lfilter(b, a, data)`：使用 `scipy.signal.lfilter` 函数对输入信号 `data` 进行滤波。这一步会根据 `b` 和 `a` 的系数进行滤波操作，得到滤波后的信号 `y`。
#
# 3. `return y`：返回滤波后的信号。
#
# 这个函数的目的是对输入信号进行带通滤波，即保留位于指定频率范围内的信号分量，滤除其他频率的信号。
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# 这段代码定义了一个函数 `read_file`，该函数用于读取 MATLAB 格式的文件。具体解释如下：
#
# 1. `def read_file(file):`：定义了一个函数 `read_file`，该函数接受一个参数 `file`，即文件的路径。
#
# 2. `data = sio.loadmat(file)`：使用 SciPy 库中的 `loadmat` 函数加载 MATLAB 格式的文件。这个函数返回一个包含 MATLAB 文件内容的字典。
#
# 3. `data = data['data']`：从加载的字典中提取键为 `'data'` 的值，这假设 MATLAB 文件中有一个名为 `'data'` 的变量。
#
# 4. `return data`：返回提取的数据，即 MATLAB 文件中名为 `'data'` 的变量的值。
#
# 总体而言，这个函数的目的是读取 MATLAB 文件中名为 `'data'` 的变量的值，并将其作为函数的输出。
def read_file(file):
    data = sio.loadmat(file)
    data = data['data']
    # print(data.shape)
    return data

# 这段代码定义了一个函数 `compute_DE`，该函数用于计算信号的差分熵（DE）。下面是对代码的解释：
#
# 1. `def compute_DE(signal):`：定义了一个名为 `compute_DE` 的函数，接受一个参数 `signal`，即输入的信号。
#
# 2. `variance = np.var(signal, ddof=1)`：使用 NumPy 库中的 `np.var` 函数计算输入信号的方差。参数 `ddof=1` 表示使用样本方差而不是总体方差。
#
# 3. `return math.log(2 * math.pi * math.e * variance) / 2`：根据差分熵的计算公式，使用 `math.log` 函数计算差分熵，并将结果返回。公式中包含了 2πe 这些常数，以及之前计算的方差。最终结果被除以2。
#
# 这个函数的目的是提供一个计算输入信号差分熵的方法。
def compute_DE(signal):
    variance = np.var(signal, ddof=1)  # 计算方差
    return math.log(2 * math.pi * math.e * variance) / 2  # 这就是论文里那个计算de的公式，这里用的是样本方差而不是总体方差



# 这段代码实现了对输入信号的分解和特征提取过程，主要包括以下步骤：
#
# 1. 从文件中读取 EEG 数据，每个 trial 包含 32 个通道的信号，每个通道有 8064 个采样点。
#
# 2. 对每个 trial 进行信号分解和特征提取。其中，准备期间（3 秒）的信号用于计算基线特征，真正任务期间（60 秒）的信号用于计算任务相关特征。
#
# 3. 对于每个通道，分别进行以下处理：
#    - 对准备期间的信号进行带通滤波，分别提取 theta、alpha、beta、gamma 频段的信号。
#    - 计算每个频段内的基线差分熵（DE）。其中，将 3 秒准备期间的信号分为 6 段，每段计算一个 DE，并求平均值。
#    - 对真正任务期间的信号，将 60 秒分为 120 段，每段包含 64 个采样点，计算每个段内的 theta、alpha、beta、gamma 频段的差分熵。
#
# 4. 将计算得到的基线差分熵和任务相关差分熵保存到相应的数据结构中。
#
# 5. 最终输出基线差分熵和任务相关差分熵的数组，其中 `base_DE` 存储了每个 trial 的基线差分熵，`decomposed_de` 存储了每个 trial 的任务相关差分熵。
#
# 这个代码主要用于脑电信号处理，通过分解和特征提取，得到不同频段的差分熵，用于分析和刻画脑电信号的特征。

def decompose(file):
    # 这段代码主要用于设置一些变量和参数：
    #
    # - `# trial*channel*sample`: 这是一条注释，表示数据的维度为 trial（试验次数）×channel（通道数）×sample（采样点数）。
    # 即数据是一个三维数组，其中每个 trial 包含多个通道的脑电信号，每个通道有多个采样点。
    #
    # - `start_index = 384`: 这是设置起始索引的值，表示从第 384 个采样点开始（即 3 秒的预试验信号）。
    # 在之后的代码中，这个值用于截取真正任务期间的信号，以及用于计算基线特征的准备期间信号。
    #
    # - `data = read_file(file)`: 调用 `read_file` 函数从文件中读取脑电信号数据，并将读取到的数据存储在变量 `data` 中。
    #
    # - `shape = data.shape`: 获取数据的形状，即数据的维度信息。这里将数据的形状赋值给变量 `shape`。
    #
    # - `frequency = 128`: 设置采样频率为 128Hz，表示每秒对信号进行 128 次采样。这个值在后续的信号处理中可能会用到，用于设置滤波器等参数。
    # trial*channel*sample

    start_index = 384  # 3s pre-trial signals
    data = read_file(file)
    shape = data.shape
    frequency = 128

    # 这部分代码涉及到初始化两个 NumPy 数组：`decomposed_de` 和 `base_DE`。
    # 1. `decomposed_de = np.empty([0, 4, 120])`: 这行代码创建了一个空的三维数组 `decomposed_de`，其形状为 (0, 4, 120)。这个数组似乎用于存储处理后的脑电信号特征。
    # 在后续的代码中，会使用 `np.vstack` 方法将新计算得到的特征逐步堆叠到这个数组中。这种方式可以避免在循环过程中多次调整数组大小，提高代码的效率。
    # 2. `base_DE = np.empty([0, 128])`: 这行代码创建了另一个空的二维数组 `base_DE`，其形状为 (0, 128)。这个数组可能用于存储基线期间的特征。
    # 与上面的数组一样，也会在后续的代码中使用 `np.vstack` 逐步堆叠新的特征。

    decomposed_de = np.empty([0, 4, 120])
    # 其实就是用来初始化的
    base_DE = np.empty([0, 128])

    for trial in range(40):
        temp_base_DE = np.empty([0])
        temp_base_theta_DE = np.empty([0])
        temp_base_alpha_DE = np.empty([0])
        temp_base_beta_DE = np.empty([0])
        temp_base_gamma_DE = np.empty([0])

        temp_de = np.empty([0, 120])

        for channel in range(32):
            # 这两行代码用于从 `data`数组中选择信号的不同部分：
            # 1.
            # `trial_signal = data[trial, channel, 384:]`：这行代码选择了从第384个采样点到信号结束的部分，即试验期间的信号。这是因为索引从0开始，而
            # `384`表示从第385个采样点开始。
            # 2.
            # `base_signal = data[trial, channel, :384]`：这行代码选择了信号的前384个采样点，即基线期间的信号。冒号`: 384` 表示从开头到第384个采样点（不包括第384个采样点）。
            # 这样，`trial_signal`包含了试验期间的信号，而`base_signal`包含了基线期间的信号，这两个部分将用于后续的处理。

            trial_signal = data[trial, channel, 384:]
            base_signal = data[trial, channel, :384]
            # ****************compute base DE****************
            # 先经过每个频段的带通滤波器，然后算相应频段内的de，
            # 准备期间的3s，0.5s为一段分为了6段，每个频段求了平均
            # 而真正任务期间的60s  将8064-384=7680个采样点以0.5s为一段(64个为一段)分为了120个，就是下面的120个循环，不求平均


            # 这部分代码是应用带通滤波器对`base_signal`进行滤波，从而得到在不同频段的信号。具体来说：
            # - `base_theta`是对`base_signal`应用带通滤波器，保留频率在4到8赫兹之间的信号。
            # - `base_alpha`是对`base_signal`应用带通滤波器，保留频率在8到14赫兹之间的信号。
            # - `base_beta`是对`base_signal`应用带通滤波器，保留频率在14到31赫兹之间的信号。
            # - `base_gamma`是对`base_signal`应用带通滤波器，保留频率在31到45赫兹之间的信号。
            # 这样，通过这四个滤波操作，分别得到了在theta、alpha、beta和gamma频段内的信号。这些信号在后续的分析中可能会用于计算差分熵等特征。
            # 带通滤波器的作用是突出特定频段的信号，而抑制其他频段的信号。

            base_theta = butter_bandpass_filter(base_signal, 4, 8, frequency, order=3)
            base_alpha = butter_bandpass_filter(base_signal, 8, 14, frequency, order=3)
            base_beta = butter_bandpass_filter(base_signal, 14, 31, frequency, order=3)
            base_gamma = butter_bandpass_filter(base_signal, 31, 45, frequency, order=3)

            # 这部分代码计算了在准备阶段（base）内每个频段（theta、alpha、beta、gamma）的差分熵（DE），然后对每个频段分成的6段进行了平均。让我来解释一下：
            #
            # 1. ** 带通滤波器： ** 首先，通过`butter_bandpass_filter`函数，对`base_signal`进行带通滤波，得到了分别在theta、alpha、beta和gamma频段的信号
            # `base_theta`、`base_alpha`、`base_beta`、`base_gamma`。
            #
            # 2. ** 差分熵计算： ** 对于每个频段，代码使用`compute_DE`函数计算了不同段的差分熵。每个频段被分成了六个子段，分别是
            # `base_theta[:64]`、`base_theta[64:128]`、...、`base_theta[320:]`。对每个子段分别计算了差分熵。
            #
            # 3. ** 取平均： ** 计算了每个子段的差分熵之后，对这六个子段的差分熵取了平均。这是通过将这六个差分熵相加，然后除以6来实现的。这样就得到了在准备阶段内每个频段的平均差分熵值。
            #
            # 这个过程被分别应用于theta、alpha、beta和gamma四个频段，得到了`base_theta_DE`、`base_alpha_DE`、`base_beta_DE`和`base_gamma_DE`
            # 四个值，代表了在准备阶段内这四个频段的平均差分熵。这些值可能用于后续的分析或特征提取。

            base_theta_DE = (compute_DE(base_theta[:64]) + compute_DE(base_theta[64:128]) + compute_DE(
                base_theta[128:192]) + compute_DE(base_theta[192:256]) + compute_DE(base_theta[256:320]) + compute_DE(
                base_theta[320:])) / 6  # 6段每段都计算一个de，然后取平均
            base_alpha_DE = (compute_DE(base_alpha[:64]) + compute_DE(base_alpha[64:128]) + compute_DE(
                base_alpha[128:192]) + compute_DE(base_theta[192:256]) + compute_DE(base_theta[256:320]) + compute_DE(
                base_theta[320:])) / 6
            base_beta_DE = (compute_DE(base_beta[:64]) + compute_DE(base_beta[64:128]) + compute_DE(
                base_beta[128:192]) + compute_DE(base_theta[192:256]) + compute_DE(base_theta[256:320]) + compute_DE(
                base_theta[320:])) / 6
            base_gamma_DE = (compute_DE(base_gamma[:64]) + compute_DE(base_gamma[64:128]) + compute_DE(
                base_gamma[128:192]) + compute_DE(base_theta[192:256]) + compute_DE(base_theta[256:320]) + compute_DE(
                base_theta[320:])) / 6

            # 这部分代码是将计算得到的基准（准备阶段）的差分熵（DE）值存储在临时的数组中，以便后续的处理或分析。
            # 具体来说，每个频段（theta、gamma、beta、alpha）的差分熵值被追加到相应的临时数组中。
            # 例如，`temp_base_theta_DE`存储了基准阶段中theta频段的差分熵值，`temp_base_gamma_DE`存储了gamma频段的差分熵值，以此类推。
            # 这些数组可能在后续的处理中被用于进一步的分析、可视化或保存。

            temp_base_theta_DE = np.append(temp_base_theta_DE, base_theta_DE)
            temp_base_gamma_DE = np.append(temp_base_gamma_DE, base_gamma_DE)
            temp_base_beta_DE = np.append(temp_base_beta_DE, base_beta_DE)
            temp_base_alpha_DE = np.append(temp_base_alpha_DE, base_alpha_DE)

            # ****************compute task DE****************

            # 在这部分代码中，对试验信号`trial_signal`进行了带通滤波，分别提取了不同频段的信号。
            # 具体来说，使用了`butter_bandpass_filter`函数，该函数是一个带通滤波器，用于保留特定频段的信号，而滤除其他频段的信号。
            #
            # - `theta`是在频段4Hz到8Hz之间的信号。
            # - `alpha`是在频段8Hz到14Hz之间的信号。
            # - `beta`是在频段14Hz到31Hz之间的信号。
            # - `gamma`是在频段31Hz到45Hz之间的信号。
            #
            # 这样，通过对试验信号进行带通滤波，就得到了不同频段的信号，为后续计算这些频段的差分熵值做准备。带通滤波的目的是保留感兴趣的频段信息，而滤除其他频段的干扰信号。

            theta = butter_bandpass_filter(trial_signal, 4, 8, frequency, order=3)
            alpha = butter_bandpass_filter(trial_signal, 8, 14, frequency, order=3)
            beta = butter_bandpass_filter(trial_signal, 14, 31, frequency, order=3)
            gamma = butter_bandpass_filter(trial_signal, 31, 45, frequency, order=3)

            # 在这部分代码中，针对每个频段（theta、alpha、beta、gamma）都初始化了一个空的数组，分别为`DE_theta`、`DE_alpha`、`DE_beta`和`DE_gamma`。
            # 这些数组将用于存储每个小段内的差分熵值。
            #
            # - `DE_theta`将存储theta频段内每个小段的差分熵值。
            # - `DE_alpha`将存储alpha频段内每个小段的差分熵值。
            # - `DE_beta`将存储beta频段内每个小段的差分熵值。
            # - `DE_gamma`将存储gamma频段内每个小段的差分熵值。
            #
            # 这些数组的目的是在每个小段内计算差分熵值，并将这些值存储在相应的数组中，以备后续使用。

            DE_theta = np.zeros(shape=[0], dtype=float)
            DE_alpha = np.zeros(shape=[0], dtype=float)
            DE_beta = np.zeros(shape=[0], dtype=float)
            DE_gamma = np.zeros(shape=[0], dtype=float)

            # 在这段代码中，通过一个循环（`for index in range(120)`），针对每个频段（theta、alpha、beta、gamma），计算了每个小段内的差分熵值，并使用 `np.append` 函数将这些值添加到相应的数组中。
            #
            # 具体来说，对于每个频段，信号被分为了120个小段，每个小段包含64个数据点。在每个小段内，调用`compute_DE`函数计算该小段的差分熵值，并使用`np.append`
            # 将这个值添加到相应的数组（`DE_theta`、`DE_alpha`、`DE_beta`或`DE_gamma`）中。
            # 这样，最终这些数组中存储了每个频段内每个小段的差分熵值。这些值将被用于后续的分析或其他操作。

            for index in range(120):
                DE_theta = np.append(DE_theta, compute_DE(theta[index * 64:(index + 1) * 64]))
                DE_alpha = np.append(DE_alpha, compute_DE(alpha[index * 64:(index + 1) * 64]))
                DE_beta = np.append(DE_beta, compute_DE(beta[index * 64:(index + 1) * 64]))
                DE_gamma = np.append(DE_gamma, compute_DE(gamma[index * 64:(index + 1) * 64]))


            # 在这段代码中，`np.vstack`函数用于纵向堆叠数组，即将多个数组按垂直方向叠加起来。
            # 在这个上下文中，对于每个频段（theta、alpha、beta、gamma），`temp_de`存储了每个频段内的差分熵值。
            # 通过这些`np.vstack`操作，将每个频段的差分熵值垂直地堆叠在一起，形成一个大的数组`temp_de`。
            #
            # 这样，`temp_de`数组的每一行代表一个频段（theta、alpha、beta、gamma），而每一列代表相应小段的差分熵值。
            # 最终，`temp_de`将包含所有频段和小段的差分熵值，为后续的分析或其他处理步骤做准备。

            temp_de = np.vstack([temp_de, DE_theta])  # 纵向堆叠
            temp_de = np.vstack([temp_de, DE_alpha])
            temp_de = np.vstack([temp_de, DE_beta])
            temp_de = np.vstack([temp_de, DE_gamma])

            # 在这段代码中：
            #
            # 1.`temp_trial_de = temp_de.reshape(-1, 4, 120)`：这一行代码将`temp_de`重新塑造成一个三维数组，其中第一个维度为`-1`表示根据数组长度自动计算，
            # 第二个维度为`4`表示四个频段（theta、alpha、beta、gamma），第三个维度为`120`表示每个频段分为120段。
            #
            # 2.`decomposed_de = np.vstack([decomposed_de, temp_trial_de])`：这一行代码使用`np.vstack`将`temp_trial_de`沿垂直方向堆叠到`decomposed_de`
            # 数组中。这个数组将包含所有试验的所有频段和小段的差分熵值。
            #
            # 3.`temp_base_DE = np.append(temp_base_theta_DE, temp_base_alpha_DE)` 等：这一系列的 `np.append`操作
            # 将计算得到的基线（baseline）差分熵值（`temp_base_theta_DE`、`temp_base_alpha_DE`等）按顺序追加到`temp_base_DE`数组中。
            #
            # 4.`base_DE = np.vstack([base_DE, temp_base_DE])`：这一行代码使用`np.vstack`将`temp_base_DE`沿垂直方向堆叠到`base_DE`数组中。
            # 这个数组将包含所有试验的基线差分熵值。
            #
            # 整个过程重复了40次试验，最终得到了`decomposed_de`存储了所有试验的所有频段和小段的差分熵值，以及`base_DE`存储了所有试验的基线差分熵值。
            # 这些数据可以用于后续的分析或其他处理。

        temp_trial_de = temp_de.reshape(-1, 4, 120)
        decomposed_de = np.vstack([decomposed_de, temp_trial_de])
        print(decomposed_de.shape)
        temp_base_DE = np.append(temp_base_theta_DE, temp_base_alpha_DE)
        temp_base_DE = np.append(temp_base_DE, temp_base_beta_DE)
        temp_base_DE = np.append(temp_base_DE, temp_base_gamma_DE)
        base_DE = np.vstack([base_DE, temp_base_DE])

        # 这段代码主要对之前得到的`decomposed_de`进行一系列的形状调整。具体步骤如下：
        # 1.
        # `decomposed_de.reshape(-1, 32, 4, 120)`: 这一步将`decomposed_de`从一维数组重塑为四维数组，
        # 第一个维度为`-1`表示根据数组长度自动计算，第二个维度为`32`表示32个通道，第三个维度为`4`表示四个频段（theta、alpha、beta、gamma），
        # 第四个维度为`120`表示每个频段分为120段。
        #
        # 2.
        # `transpose([0, 3, 2, 1])`: 这一步进行了维度的转置，调整维度的顺序。
        #
        # 3.
        # `reshape(-1, 4, 32)`: 这一步将数组重新塑造为三维数组，其中第一个维度为`-1`表示根据数组长度自动计算，第二个维度为`4`表示四个频段（theta、alpha、beta、gamma），
        # 第三个维度为`32`表示32个通道。
        #
        # 4.
        # `reshape(-1, 128)`: 这一步再次将数组重新塑造为二维数组，其中第一个维度为`-1`表示根据数组长度自动计算，第二个维度为`128`表示32个通道乘以4个频段的差分熵值。
        #
        # 最终，打印了`base_DE`和`decomposed_de`的形状信息，其中`base_DE`是一个包含40个试验的基线差分熵值的数组，形状为`(40, 128)`，
        # 而`decomposed_de`是一个包含所有试验的所有频段和小段的差分熵值的数组，形状为`(4800, 128)`，其中`4800`是40次试验乘以120段小段。

    decomposed_de = decomposed_de.reshape(-1, 32, 4, 120).transpose([0, 3, 2, 1]).reshape(-1, 4, 32).reshape(-1, 128)
    print("base_DE shape:", base_DE.shape)  # 40*128
    print("trial_DE shape:", decomposed_de.shape)  # 4800*128,4800=40次trials*120段，64个点为一段
    return base_DE, decomposed_de

# `def get_labels(file)` 函数的主要目标是从情感标签文件中提取标签，并对其进行处理，具体过程如下：
#
# 1. 从情感标签文件中加载 valence 和 arousal 的标签，形状为 `(40, 4)`，其中每一行对应一个试验。
#
# 2. 对 valence 和 arousal 标签进行二值化处理，将标签值大于 5 的部分设为 `True`，表示高于中性；否则设为 `False`，表示低于等于中性。
#
# 3. 初始化空数组 `final_valence_labels` 和 `final_arousal_labels`，用于存储处理后的标签。
#
# 4. 对每个试验的 valence 和 arousal 标签进行循环，为每个试验构造包含 120 个标签值的数组。因为每个试验有 120 段，而且每个试验的标签在整个试验中是一样的。
#
# 5. 返回处理后的最终 valence 和 arousal 标签数组，形状为 `(40*120,)`。
#
# 总的来说，这个函数的目标是从情感标签文件中提取标签，对其进行二值化处理，并为每个试验构造一个包含 120 个标签值的数组，最终返回处理后的标签数组。

def get_labels(file):

    # 这部分代码从情感标签文件中加载了valence和arousal的标签，并进行了二值化处理：
    #
    # 1.
    # `valence_labels = sio.loadmat(file)["labels"][:, 0] > 5`：从情感标签文件中加载valence标签，
    # 并将标签值大于5的部分设为`True`，表示高于中性；否则设为`False`，表示低于等于中性。
    #
    # 2.
    # `arousal_labels = sio.loadmat(file)["labels"][:, 1] > 5`：从情感标签文件中加载arousal标签，
    # 并将标签值大于5的部分设为`True`，表示高于中性；否则设为`False`，表示低于等于中性。
    #
    # 3.
    # `final_valence_labels = np.empty([0])` 和 `final_arousal_labels = np.empty([0])`：初始化两个空数组，用于存储处理后的valence和arousal标签。
    #
    # 这样处理后，`valence_labels`和`arousal_labels`变成了布尔型数组，表示每个试验在相应的情感维度上是否高于中性。这些布尔数组被用于后续的标签构造。

    # 0 valence, 1 arousal, 2 dominance, 3 liking
    valence_labels = sio.loadmat(file)["labels"][:, 0] > 5  # valence labels
    arousal_labels = sio.loadmat(file)["labels"][:, 1] > 5  # arousal labels
    final_valence_labels = np.empty([0])
    final_arousal_labels = np.empty([0])

    # 这段代码用于构建最终的valence和arousal标签数组：
    #
    # 1.
    # `for i in range(len(valence_labels)): `：遍历每个试验。
    #
    # 2.
    # `for j in range(0, 120): `：对于每个试验，遍历其120个点。
    #
    # 3.
    # `final_valence_labels = np.append(final_valence_labels, valence_labels[i])` 和
    # `final_arousal_labels = np.append(final_arousal_labels, arousal_labels[i])`：将当前试验的valence和arousal标签值添加到相应的最终标签数组中。
    #
    # 通过这样的操作，最终的`final_valence_labels`和`final_arousal_labels`形成了一个形状为(40, 120)的数组，
    # 其中每个试验的标签值都是一样的，因为对于每个试验，其标签值被重复了120次。这符合数据的格式，其中每个试验包含了120个数据点。
    # 最后，通过`print("labels:", final_arousal_labels.shape)`输出了标签数组的形状，其中`final_arousal_labels.shape`的结果应该是(4800, )，
    # 因为有40次试验，每次试验有120个标签点。函数最后返回了构建好的valence和arousal标签数组。

    # print("labels.shape:",len(valence_labels)) 就是40，没错的
    for i in range(len(valence_labels)):
        # 这里的valence_labels的长度应该是40，下面的循环的意思是构造的标签shape为40*120，并且每个120个点标签值是一样的，因为40次trials每次的trial的标签值相同
        for j in range(0, 120):
            final_valence_labels = np.append(final_valence_labels, valence_labels[i])
            final_arousal_labels = np.append(final_arousal_labels, arousal_labels[i])
    print("labels:", final_arousal_labels.shape)
    return final_arousal_labels, final_valence_labels

# 这段代码定义了一个函数 `wgn(x, snr)`，用于在输入信号 `x` 上添加高斯白噪声（White Gaussian Noise）。这个函数接受两个参数：
#
# 1. `x`：输入信号，是一个一维的 NumPy 数组。
# 2. `snr`：信噪比（Signal-to-Noise Ratio），以分贝（dB）为单位。
#
# 函数的主要步骤如下：
#
# 1. `snr = 10 ** (snr / 10.0)`：将输入的信噪比（以分贝为单位）转换为倍数形式。
#
# 2. `xpower = np.sum(x ** 2) / len(x)`：计算输入信号 `x` 的功率。
#
# 3. `npower = xpower / snr`：根据信噪比计算噪声的功率。
#
# 4. `np.random.randn(len(x))`：生成与输入信号长度相同的高斯分布随机数。
#
# 5. `* np.sqrt(npower)`：通过噪声功率调整噪声的标准差。
#
# 最终，函数返回一个与输入信号相同长度的数组，其中包含了添加了高斯白噪声的信号。
# 在信号处理和通信系统中，高斯白噪声是一种常见的模型，用于表示随机性很强的背景噪声。它具有以下特性：
#
# 1. **高斯性质（Gaussian）：** 高斯白噪声是一种服从高斯分布的随机噪声，其统计性质完全由均值和方差描述。
#
# 2. **白噪声性质（White）：** 白噪声表示在所有频率上具有均匀分布的能量，即在频谱中平坦分布。这意味着它在各个频率上都有相似的能量。
#
# 为什么要定义高斯白噪声呢？
#
# - **模拟真实环境：** 许多自然和人造信号在传输和采集过程中会受到环境噪声的影响。高斯白噪声模型可以用来模拟这些噪声，使得算法和系统在更真实的情境下进行测试。
#
# - **通信系统：** 在通信系统中，信号通常会受到各种干扰和噪声，其中高斯白噪声是一个常见的模型，用于表示通信信道中的随机噪声。
#
# - **信号处理算法测试：** 在开发和测试信号处理算法时，向信号中添加高斯白噪声可以帮助评估算法的性能，尤其是在实际应用中可能面临的噪声环境下。
#
# 总的来说，定义高斯白噪声是为了更好地模拟真实场景中的噪声，并且它在信号处理和通信领域中具有广泛的应用。

def wgn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

# 这是一个特征归一化（Feature Normalization）的函数。特征归一化是一种常见的数据预处理步骤，其目的是将数据的特征进行缩放，使其具有相似的尺度。这有助于一些机器学习算法更快地收敛并提高模型的性能。这里使用的是零均值归一化（Zero-mean Normalization），也称为标准化。
#
# 具体步骤如下：
#
# 1. **计算均值和标准差：** 计算数据中非零元素的均值（mean）和标准差（sigma）。
#
# 2. **归一化：** 对数据中的非零元素进行归一化，公式为 X - mean / sigma。
#
# 3. **返回归一化后的数据：** 返回归一化后的数据。
#
# 这个函数接受一个数据数组 `data` 作为输入，对其进行归一化，并返回归一化后的数据。
#
# 请注意，这个函数使用的是数据中非零元素的均值和标准差，避免了零值对均值和标准差的影响。这在处理稀疏数据时比较常见。

def feature_normalize(data):
    mean = data[data.nonzero()].mean()
    sigma = data[data.nonzero()].std()
    data_normalized = data
    data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean) / sigma
    return data_normalized

# 这段代码主要是对一个文件夹里的DEAP数据集文件进行处理。下面是主要步骤的解释：
#
# 1. **导入必要的库：** 代码一开始导入了一些必要的库，包括`os`和`scipy.io`。
#
# 2. **设置文件路径：** 定义了DEAP数据集的原始文件夹路径 `dataset_dir` 和结果保存路径 `result_dir`。
#
# 3. **遍历文件夹：** 使用 `os.listdir` 遍历指定文件夹中的每个文件。
#
# 4. **处理每个文件：** 对每个文件进行处理，调用了 `decompose` 函数，该函数对DEAP数据进行了分解（decompose），生成了 `base_DE` 和 `trial_DE`。同时，调用了 `get_labels` 函数，获取了情感标签 `arousal_labels` 和 `valence_labels`。
#
# 5. **保存结果：** 使用 `scipy.io.savemat` 将处理后的数据保存到指定的结果文件夹中，文件名以 "DE_" 开头。
#
# 总体来说，这段代码的目的是对DEAP数据集进行一系列的处理，包括分解和标签提取，然后将处理后的结果保存到新的文件夹中。

if __name__ == '__main__':
    dataset_dir = "D:/DEAP/data_preprocessed_matlab/"

    result_dir = "D:/DEAP/all_0.5/"

    # 这部分代码检查指定的结果目录是否存在，如果不存在，则使用`os.makedirs(result_dir)`创建该目录。
    # 这是为了确保在保存处理后的结果之前，目标目录是存在的。如果目录不存在，就会创建该目录，否则，就什么都不做。这是一种保证结果目录存在的常见做法。

    if os.path.isdir(result_dir) == False:
        os.makedirs(result_dir)

    # 这里通过循环遍历指定的dataset_dir目录下的每个文件。打印一条消息表示正在处理当前文件。
    for file in os.listdir(dataset_dir):  # 文件夹里的每一个文件 进行decompose处理
        print("processing: ", file, "......")

        # 该行通过使用os.path.join构建了当前文件的完整路径，连接了目录路径(dataset_dir)和文件名(file)。
        file_path = os.path.join(dataset_dir, file)

        # 调用decompose函数处理当前文件(file_path)，并将返回的两个值分别赋给base_DE和trial_DE。
        base_DE, trial_DE = decompose(file_path)

        # 调用get_labels函数获取当前文件(file_path)的唤醒和愉悦标签。获取到的标签分别赋给 arousal_labels和valence_labels。
        arousal_labels, valence_labels = get_labels(file_path)

        # 使用scipy库中的sio.savemat函数保存结果为MATLAB(.mat) 格式。文件名以"DE_"为前缀，后接原始文件名(file)。
        # 要保存的数据包括base_DE、trial_DE、valence_labels和arousal_labels。文件保存在指定的结果目录(result_dir)中。
        sio.savemat(result_dir + "DE_" + file,
                    {"base_data": base_DE, "data": trial_DE, "valence_labels": valence_labels,
                     "arousal_labels": arousal_labels})