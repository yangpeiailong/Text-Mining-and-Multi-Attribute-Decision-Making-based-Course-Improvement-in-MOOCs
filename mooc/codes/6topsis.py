import numpy as np
import xlrd
import math


def calculate_comprehensive_scores(evaluation_criteria, weights, b_or_c):
    # 验证输入参数长度是否一致
    num_criteria = len(weights)
    if len(evaluation_criteria[0]) != num_criteria or len(b_or_c) != num_criteria:
        raise ValueError("Number of criteria must match the number of evaluation scores and weights")

        # 初始化综合评分列表
    comprehensive_scores = []

    # 遍历每个备选方案
    for alternative in evaluation_criteria:
        # 初始化当前备选方案的加权分数
        weighted_score = 0

        # 遍历每个指标
        for criterion_index in range(num_criteria):
            # 获取当前指标的权重和类型
            weight = weights[criterion_index]
            criterion_type = b_or_c[criterion_index]

            # 获取当前备选方案在该指标上的评分
            score = alternative[criterion_index]

            # 根据指标类型进行加权
            if criterion_type == 'False':  # 收益指标
                weighted_score += weight * score
            else:  # 成本指标
                # 成本指标通常用其倒数或减去一个常数来转换
                weighted_score += weight / (score + 1e-6)  # 避免除以零的错误，加入一个非常小的数


                # 将当前备选方案的综合评分添加到列表中
        comprehensive_scores.append(weighted_score)

    return comprehensive_scores

def topsis(data, w, borc):
    # step1: 归一化每个指标
    data_new = np.zeros((data.shape[0], data.shape[1]), np.float32)
    # temp = np.sum([])
    # data = data / [math.sqrt(sum([data[jj][ii] ** 2 for jj in range(data.shape[0])])) for ii in range(data.shape[1])]
    for jj in range(data.shape[1]):
        temp = math.sqrt(sum([iii ** 2 for iii in data[:, jj]]))
        for ii in range(data.shape[0]):
            data_new[ii][jj] = data[ii][jj] / temp
    # step2: 计算加权后的决策矩阵
    v_mat = np.zeros((data.shape[0], data.shape[1]), np.float32)
    for ii in range(v_mat.shape[0]):
        for jj in range(v_mat.shape[1]):
            v_mat[ii][jj] = data_new[ii][jj] * w[jj]
    # step3: 获得正理想解和负理想解值集合
    splus = np.zeros(data.shape[1], np.float32)
    sminus = np.zeros(data.shape[1], np.float32)
    for jj in range(v_mat.shape[1]):
        if borc[jj]:
            splus[jj] = min(v_mat[:, jj])
            sminus[jj] = max(v_mat[:, jj])
        else:
            splus[jj] = max(v_mat[:, jj])
            sminus[jj] = min(v_mat[:, jj])

    # step4: 计算备选方案与正负理想解的欧式距离
    distance = np.zeros((data.shape[0], 2), np.float32)  # 每行存放与正负理想解的欧氏距离，
    # 如distance[0][0]表示第一个备选方案与正理想解的距离，distance[0][1]表示第一个备选方案与负理想解的距离
    for ii in range(distance.shape[0]):
        distance[ii][0] = np.sqrt(np.sum(np.square(v_mat[ii] - splus)))
        distance[ii][1] = np.sqrt(np.sum(np.square(v_mat[ii] - sminus)))

    # step5: 计算R值
    rset = np.zeros(data.shape[0], np.float32)
    for ii in range(len(rset)):
        rset[ii] = distance[ii][1] / (distance[ii][0] + distance[ii][1])
    sorted_score_index = np.argsort(rset)[:: -1]
    rset_temp = rset[sorted_score_index]
    rset_temp = rset_temp.tolist()
    new_sort = []
    for ii in range(len(rset)):
        new_sort.append(rset_temp.index(rset[ii]) + 1)
    return rset, new_sort

dataname = 'Java'
wb = xlrd.open_workbook("D:/pythonwork/paper2024/mooc/data/5decision matrix/" + dataname + ".xls")
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
weights = [0.17744637, 0.17476294, 0.14933752, 0.17247386, 0.15582149, 0.17015782] # Java
# weights = [0.17848966, 0.17256881, 0.15578431, 0.16887738, 0.15572494, 0.16855491] # C

print(topsis(dec_mat, weights, b_or_c))
# print(calculate_comprehensive_scores(dec_mat, weights, b_or_c))