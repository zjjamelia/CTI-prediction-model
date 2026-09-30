import numpy as np


def resize_matrix(matrix, target_shape=(511, 511)):
    """
    将矩阵压缩或裁剪为指定的形状。

    参数：
    matrix (numpy.ndarray): 输入矩阵。
    target_shape (tuple): 目标形状，默认为 (511, 511)。

    返回：
    numpy.ndarray: 压缩或裁剪后的矩阵。
    """
    # 获取目标形状
    target_rows, target_cols = target_shape

    # 获取输入矩阵的当前形状
    current_rows, current_cols = matrix.shape

    # 裁剪矩阵以适应目标形状
    resized_matrix = matrix[:target_rows, :target_cols]

    return resized_matrix


# 读取文件中的矩阵
input_file = '../network/Sim_mat_protein_protein.txt'
matrix = np.loadtxt(input_file)

# 将矩阵裁剪为 (511, 511)
resized_matrix = resize_matrix(matrix, (511, 511))

# 保存裁剪后的矩阵到新文件
output_file = '../network/Resized_Sim_mat_protein_protein.txt'
np.savetxt(output_file, resized_matrix)

print("原始矩阵形状:", matrix.shape)
print("压缩后的矩阵形状:", resized_matrix.shape)
