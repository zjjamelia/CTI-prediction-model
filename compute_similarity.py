import numpy as np
import os
from scipy.spatial.distance import pdist, squareform
import time

# 定义网络文件名
nets = [
    'mat_drug_matrix', 'mat_drug_protein', 'mat_drug_structure',
    'mat_protein_matrix','mat_protein_drug','mat_protein_protein'
]

# 处理每个网络文件
for net in nets:
    start_time = time.time()

    # 构建输入文件路径
    input_id = os.path.join('../data', f'{net}.txt')

    # 读取数据
    M = np.loadtxt(input_id)

    # 计算Jaccard相似性
    Sim = 1 - pdist(M, 'jaccard')
    Sim = squareform(Sim)

    # 添加单位矩阵并处理NaN值
    Sim = Sim + np.eye(M.shape[0])
    Sim = np.nan_to_num(Sim)

    # 构建输出文件路径
    output_id = os.path.join('../network', f'Sim_{net}.txt')

    # 保存相似性矩阵
    np.savetxt(output_id, Sim, delimiter='\t')

    end_time = time.time()
    print(f'Processed {net} in {end_time - start_time:.2f} seconds')

# 处理化学相似性
#M_drugs = np.loadtxt('../data/Similarity_Matrix_Drugs.txt')
#np.savetxt('../network/Sim_mat_Drugs.txt', M_drugs, delimiter='\t')
#
## 处理序列相似性
#M_proteins = np.loadtxt('../data/Similarity_Matrix_Proteins.txt')
#np.savetxt('../network/Sim_mat_Proteins.txt', M_proteins, delimiter='\t')
