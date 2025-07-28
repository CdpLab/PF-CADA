import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 打开图像
img = Image.open("1.png")  # 替换为你的图像路径
img = img.convert("RGB")  # 确保图像为 RGB 格式
img_array = np.array(img)

# 显示图像大小
print("Image shape:", img_array.shape)  # (height, width, 3)

# 定义颜色条的数值范围
value_min, value_max = 0.2, 0.6

# 获取颜色条的 RGB 值范围，假设最低值为蓝色，最高值为红色
# 这里假设 RGB(0, 0, 255) 对应最低值，RGB(255, 0, 0) 对应最高值
color_min = np.array([0, 0, 255])   # 蓝色
color_max = np.array([255, 0, 0])   # 红色

# 计算每个像素的数值（线性插值）
def rgb_to_value(rgb):
    # 计算 RGB 与 color_min 的距离和 color_max 的距离
    dist_to_min = np.linalg.norm(rgb - color_min)
    dist_to_max = np.linalg.norm(rgb - color_max)
    # 计算在数值范围内的比例
    ratio = dist_to_min / (dist_to_min + dist_to_max)
    # 将比例映射到数值范围并反转颜色
    value = value_max - (value_max - value_min) * (1 - ratio)
    return value

# 将 RGB 转换为数值矩阵
value_matrix = np.zeros((img_array.shape[0], img_array.shape[1]))
for i in range(img_array.shape[0]):
    for j in range(img_array.shape[1]):
        value_matrix[i, j] = rgb_to_value(img_array[i, j])

# 显示数值矩阵（反转颜色）
plt.imshow(value_matrix, cmap="coolwarm", vmin=value_min, vmax=value_max)
plt.colorbar(label="Value")
plt.title("Mapped Value Matrix from RGB Heatmap (Reversed Colors)")
plt.show()
