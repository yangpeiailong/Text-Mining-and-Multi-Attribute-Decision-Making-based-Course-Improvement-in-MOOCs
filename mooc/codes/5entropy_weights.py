import numpy as np
import xlrd

def entropy_weight(decision_matrix, benefit_or_cost):
    # 数据预处理：对于成本指标，取其倒数
    for i, is_cost in enumerate(benefit_or_cost):
        if is_cost:  # 成本指标
            decision_matrix[:, i] = 1 / decision_matrix[:, i]

            # 数据标准化处理
    decision_matrix = decision_matrix / decision_matrix.sum(axis=0)

    # 计算每个属性的熵值
    m, n = decision_matrix.shape
    entropy = np.zeros((n,))
    for j in range(n):
        entropy[j] = -np.sum(decision_matrix[:, j] * np.log2(decision_matrix[:, j] + np.finfo(float).eps))

        # 计算每个属性的权重
    weight = (1 - entropy) / (1 - entropy).sum()

    return weight


dataname = 'C_language_programming'
wb = xlrd.open_workbook("D:/pythonwork/paper2024/mooc/5决策矩阵/" + dataname + ".xls")
sheet = wb.sheet_by_index(0)  # 通过索引的方式获取到某一个sheet，现在是获取的第一个sheet页，也可以通过sheet的名称进行获取，sheet_by_name('sheet名称')
rows = sheet.nrows  # 获取sheet页的行数，一共有几行
columns = sheet.ncols

dec_mat = []
for i in range(1, rows):
    tmp_list = []
    for j in range(1, columns):
        tmp_list.append(float(sheet.cell(i,j).value))
    dec_mat.append(tmp_list)

dec_mat = np.array(dec_mat)

b_or_c = [True, False, False, False, False, True]
print(entropy_weight(dec_mat, b_or_c))
