import numpy as np

def find_intersection(matrix1, matrix2):
    """
    找到两个矩阵的交集（基于列向量）。
    
    参数:
        matrix1 (numpy.ndarray): 第一个矩阵，形状为 (3, N1)
        matrix2 (numpy.ndarray): 第二个矩阵，形状为 (3, N2)
    
    返回:
        numpy.ndarray: 交集矩阵，形状为 (3, M)，其中 M 是交集的列数
    """
    # 将矩阵的列向量转换为行向量，方便比较
    matrix1_rows = matrix1.T  # 转置后形状为 (N1, 3)
    matrix2_rows = matrix2.T  # 转置后形状为 (N2, 3)
    
    # 使用 set 操作找到交集
    intersection = np.array([row for row in matrix1_rows if row.tolist() in matrix2_rows.tolist()])
    
    # 如果没有交集，返回空矩阵
    if intersection.size == 0:
        return np.empty((3, 0))
    
    # 将交集矩阵转置回原来的形状 (3, M)
    return intersection.T

# 示例矩阵
matrix1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matrix2 = np.array([[1, 2, 10], [4, 5, 11], [7, 8, 12], [13, 14, 15]])

A = np.array([[1, 2], [4, 5], [7, 8]])
B = np.array([[1, 2], [3, 5], [7, 8], [13, 14]])
C = np.array([x for x in set(tuple(x) for x in A) & set(tuple(x) for x in B)])
print(C)
print(matrix1[:,[1,2]])
# 找到交集
intersection_matrix = find_intersection(matrix1, matrix2)

print("矩阵1:")
print(matrix1)
print("矩阵2:")
print(matrix2)
print("交集矩阵:")
print(intersection_matrix)