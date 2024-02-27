import xlrd
import re
import xlwt
import jieba
from tqdm import tqdm
import jieba.posseg as psg

wb = xlrd.open_workbook("D:/pythonwork/paper2024/mooc/data/Java_切分句子后.xls")
sheet = wb.sheet_by_index(0)  # 通过索引的方式获取到某一个sheet，现在是获取的第一个sheet页，也可以通过sheet的名称进行获取，sheet_by_name('sheet名称')
rows = sheet.nrows  # 获取sheet页的行数，一共有几行

sentences = []

for i in range(1, rows):
    sentences.append(str(sheet.cell(i, 1).value).strip())


noun = []
sample_listwithlabel = []
sample_listwithoutlabel = []
for i in tqdm(sentences, ncols=100):
    # temp_list = [x.word for x in psg.cut(i) if x.flag.startswith('n')]
    temp_list1 = [(x.word, x.flag) for x in psg.cut(i)]
    temp_list2 = [x.word for x in psg.cut(i)]
    sample_listwithlabel.append(temp_list1) # 存放有词性标注的分词结果，格式为
    # [
    #   [(“词11”,“词性标注11”), (“词12”,“词性标注12”), ...],
    #   [(“词21”,“词性标注21”), (“词22”,“词性标注22”), ...],
    #   ...
    # ]
    sample_listwithoutlabel.append(temp_list2)  # 存放无词性标注的分词结果，格式为
    # [
    #   [“词11”, “词12”, ...],
    #   [“词21”, “词22”, ...],
    #   ...
    # ]
    for j in temp_list1:
        if j[1].startswith('n'):
            if [j[0], j[1], 0] not in noun:
                noun.append([j[0], j[1], 0])
            # 存放名词 每个元素都是一个列表，为["名词", "标注", "名词的文档频率(初始化为0,后面在0上面加)"]

for i in tqdm(range(len(noun)), ncols=100):
    for j in range(len(sample_listwithoutlabel)):
        if noun[i][0] in sample_listwithoutlabel[j]:
            noun[i][2] += 1

nounwithnumber = sorted(noun, key=(lambda x: x[2]), reverse=True)
nounwithnumber2 = []
# 过滤掉文档频率小于5的名词
for i in nounwithnumber:
    if i[2] >= 5 and len(i[0]) > 1:
        nounwithnumber2.append(i)
# print(nounwithnumber2)
# nouns&numbers.xls存放名词、名词的标签和名词的数量
book = xlwt.Workbook(encoding='utf-8')
sheet1 = book.add_sheet('test', cell_overwrite_ok=True)
sheet1.write(0, 0, '名词')
sheet1.write(0, 1, '词性标注')
sheet1.write(0, 2, '文档频数')
row = 1
for i in range(len(nounwithnumber2)):
    sheet1.write(row, 0, nounwithnumber2[i][0])
    sheet1.write(row, 1, nounwithnumber2[i][1])
    sheet1.write(row, 2, nounwithnumber2[i][2])
    row += 1
book.save('D:/pythonwork/paper2024/mooc/data/Java_nouns&numbers.xls')