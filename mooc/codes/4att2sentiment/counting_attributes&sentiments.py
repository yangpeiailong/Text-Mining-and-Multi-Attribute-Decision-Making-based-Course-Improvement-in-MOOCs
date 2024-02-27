# encoding=utf-8
import jieba
from tqdm import tqdm
import xlwt
import os
import xlrd

dataname = 'C_language_programming'
wb = xlrd.open_workbook("D:/pythonwork/paper2024/mooc/属性/" + dataname + "_关键词归类2.xls")
sheet = wb.sheet_by_index(0)  # 通过索引的方式获取到某一个sheet，现在是获取的第一个sheet页，也可以通过sheet的名称进行获取，sheet_by_name('sheet名称')
rows = sheet.nrows  # 获取sheet页的行数，一共有几行
# columns = sheet.ncols  # 获取sheet页的列数，一共有几列
# mainarea2areas = {
#     '形象价值':['满意度', '忠诚度', '美誉度', '知名度'],
#     '情感体验':['场景感', '刺激感', '新奇感', '独特感', '丰富感', '趣味性', '愉悦感', '闲适感'],
#     '吸引物感知':['景观', '环境感受', '自然状态', '人文氛围', '地方特色', '物质文化'],
#     '经营管理感知':['服务质量', '旅游服务', '娱乐产品', '艺术表演', '景区管理', '设施设备'],
#     '功能感知':['宣教功能','社交功能', '健身功能', '感官感受(视觉)', '感官感受(听觉)', '感官感受(味觉)', '感官感受(嗅觉)'],
#     '成本感知':['经济感知','时间成本','体力成本']
# }
# row_for_mainarea = [1, 5, 13, 19, 25, 32]
area_initialization = {str(sheet.cell(i, 0).value): {j: {'总': 0, '正': 0, '中': 0, '负': 0, '无': 0, '正比例':0, '中比例':0, '负比例':0, '无比例':0} for j in
                                                      str(sheet.cell(i, 1).value).split()} for i in range(1, rows)}


readname = "D:/pythonwork/paper2024/mooc/3获取每个句子的属性词/{}_句子、标签和属性词.xls".format(dataname)
# print(readname)
area2concept = area_initialization
if os.path.exists(readname):
    wb2 = xlrd.open_workbook(readname, encoding_override='gbk')
    sheet2 = wb2.sheet_by_index(0)  # 通过索引的方式获取到某一个sheet，现在是获取的第一个sheet页，也可以通过sheet的名称进行获取，sheet_by_name('sheet名称')
    rows2 = sheet2.nrows  # 获取sheet页的行数，一共有几行
    samples = [str(sheet2.cell(i, 0).value) for i in range(1, rows2)]
    # print(readname)
    labels = [int(float(str(sheet2.cell(i, 1).value))) for i in range(1, rows2)]

    samples = [jieba.lcut(samples[i]) for i in range(len(samples))]
    # attention = {i:0 for i in area2concept.keys()}
    for i in tqdm(area2concept.keys()):
        for j in area2concept[i].keys():
            for k in range(len(samples)):
                if j in samples[k]:
                    area2concept[i][j]['总'] += 1
                    # attention[i] += 1
                    if labels[k] == 1:
                        area2concept[i][j]['正'] += 1
                    elif labels[k] == 0:
                        area2concept[i][j]['中'] += 1
                    elif labels[k] == -1:
                        area2concept[i][j]['负'] += 1
                    elif labels[k] == 2:
                        area2concept[i][j]['无'] += 1
            if area2concept[i][j]['总'] != 0:
                area2concept[i][j]['正比例'] = area2concept[i][j]['正']/ area2concept[i][j]['总']
                area2concept[i][j]['中比例'] = area2concept[i][j]['中'] / area2concept[i][j]['总']
                area2concept[i][j]['负比例'] = area2concept[i][j]['负'] / area2concept[i][j]['总']
                area2concept[i][j]['无比例'] = area2concept[i][j]['无'] / area2concept[i][j]['总']
            else:
                area2concept[i][j]['正比例'] = 0
                area2concept[i][j]['中比例'] = 0
                area2concept[i][j]['负比例'] = 0
                area2concept[i][j]['无比例'] = 0
    # print(area2concept)
    area2total_sentiments = {i:{'总':0, '正':0, '中':0, '负':0, '无':0, '词平均文档频率':0, '正比例':0, '中比例':0, '负比例':0, '无比例':0} for i in area2concept.keys()}
    for i in tqdm(area2concept.keys()):
        for k in range(len(samples)):
            if set(area2concept[i].keys()) & set(samples[k]):
                area2total_sentiments[i]['总'] += 1
                if labels[k] == 1:
                    area2total_sentiments[i]['正'] += 1
                elif labels[k] == 0:
                    area2total_sentiments[i]['中'] += 1
                elif labels[k] == -1:
                    area2total_sentiments[i]['负'] += 1
                elif labels[k] == 2:
                    area2total_sentiments[i]['无'] += 1
        area2total_sentiments[i]['词平均文档频率'] = (sum([area2concept[i][j]['总'] for j in area2concept[i].keys()])
                                                      / len(area2concept[i].keys()))
        area2total_sentiments[i]['正比例'] = area2total_sentiments[i]['正'] / area2total_sentiments[i]['总']
        area2total_sentiments[i]['中比例'] = area2total_sentiments[i]['中'] / area2total_sentiments[i]['总']
        area2total_sentiments[i]['负比例'] = area2total_sentiments[i]['负'] / area2total_sentiments[i]['总']
        area2total_sentiments[i]['无比例'] = area2total_sentiments[i]['无'] / area2total_sentiments[i]['总']

    book = xlwt.Workbook(encoding='utf-8')
    sheet3 = book.add_sheet('sheet1', cell_overwrite_ok=True)
    sheet3.write(0, 0, 'Attribute')
    sheet3.write(0, 1, 'Total number')
    sheet3.write(0, 2, 'Positive number')
    sheet3.write(0, 3, 'Neutral number')
    sheet3.write(0, 4, 'Negative number')
    sheet3.write(0, 5, 'Non-emotional number')
    sheet3.write(0, 6, 'Average DF')
    sheet3.write(0, 7, 'Positive proportion')
    sheet3.write(0, 8, 'Neutral proportion')
    sheet3.write(0, 9, 'Negative proportion')
    sheet3.write(0, 10, 'Non-emotional proportion')
    rows3 = 1
    for i in tqdm(area2concept.keys()):
        sheet3.write(rows3, 0, i)
        sheet3.write(rows3, 1, area2total_sentiments[i]['总'])
        sheet3.write(rows3, 2, area2total_sentiments[i]['正'])
        sheet3.write(rows3, 3, area2total_sentiments[i]['中'])
        sheet3.write(rows3, 4, area2total_sentiments[i]['负'])
        sheet3.write(rows3, 5, area2total_sentiments[i]['无'])
        sheet3.write(rows3, 6, area2total_sentiments[i]['词平均文档频率'])
        sheet3.write(rows3, 7, area2total_sentiments[i]['正比例'])
        sheet3.write(rows3, 8, area2total_sentiments[i]['中比例'])
        sheet3.write(rows3, 9, area2total_sentiments[i]['负比例'])
        sheet3.write(rows3, 10, area2total_sentiments[i]['无比例'])
        rows3 += 1
    book.save('D:/pythonwork/paper2024/mooc/3获取每个句子的属性词/{}_属性词情感统计.xls'.format(dataname))
