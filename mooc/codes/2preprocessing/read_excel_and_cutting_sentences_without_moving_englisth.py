import xlrd
import re
import xlwt
import jieba
from tqdm import tqdm
import jieba.posseg as psg

wb = xlrd.open_workbook("D:/pythonwork/paper2024/mooc/data/C_language_programming.xls")
sheet = wb.sheet_by_index(0)  # 通过索引的方式获取到某一个sheet，现在是获取的第一个sheet页，也可以通过sheet的名称进行获取，sheet_by_name('sheet名称')
rows = sheet.nrows  # 获取sheet页的行数，一共有几行
columns = sheet.ncols  # 获取sheet页的列数，一共有几列

# 判断是否含中文


names = []
sentences = []
times = []
likes = []
class_order = []
rankings = []
stopwords_file = 'D:/pythonwork/paper2024/mooc/data/chinesestopwords.txt'
with open(stopwords_file, 'r') as f:
    stopword_list = [word.strip() for word in f.readlines()]



for i in tqdm(range(1, rows)):
    # 段落拆分为句子，除了句子结束符（。，？之类）外还有一些特定的转折词，表示句子前后情感有转折的，尽量确保句子情感前后统一
    list1 = re.split('<br />|而且|但是|但|\s+要是|,要是|，要是|\s+就是|,就是|，就是|只是|可惜|不过|。|！|!|；', str(sheet.cell(i, 1).value))
    while '' in list1:
        list1.remove('')
    list1 = [jieba.lcut(j) for j in list1]
    list1 = [[word for word in j if word not in stopword_list] for j in list1]
    list1 = [''.join(j) for j in list1]
    for j in list1:
        if len(j) >= 5:  # 过滤掉不包含中文及字数少于5的句子
            names.append(str(sheet.cell(i,0).value))
            sentences.append(j.strip().replace('\n', ''))
            times.append(str(sheet.cell(i, 2).value))
            likes.append(str(sheet.cell(i, 3).value))
            class_order.append(sheet.cell(i, 4).value)
            rankings.append(int(sheet.cell(i, 5).value))
# print(sentences, times)
book = xlwt.Workbook(encoding='utf-8')
sheet1 = book.add_sheet('sheet1', cell_overwrite_ok=True)
sheet1.write(0, 0, '用户昵称')
sheet1.write(0, 1, '评论内容')
sheet1.write(0, 2, '评论时间')
sheet1.write(0, 3, '点赞数')
sheet1.write(0, 4, '第几次课程')
sheet1.write(0, 5, '评分')

row = 1
for i in range(len(sentences)):
    sheet1.write(row, 0, names[i])
    sheet1.write(row, 1, sentences[i])
    sheet1.write(row, 2, times[i])
    sheet1.write(row, 3, likes[i])
    sheet1.write(row, 4, class_order[i])
    sheet1.write(row, 5, rankings[i])
    row += 1
book.save("D:/pythonwork/paper2024/mooc/data/C_language_programming_切分句子后_特殊.xls")