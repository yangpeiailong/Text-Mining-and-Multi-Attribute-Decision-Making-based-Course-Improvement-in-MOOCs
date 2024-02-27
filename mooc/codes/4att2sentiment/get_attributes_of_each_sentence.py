import xlrd
import xlwt
from tqdm import tqdm
import jieba

dataname = 'C_language_programming'
wb = xlrd.open_workbook("D:/pythonwork/paper2024/mooc/2句子和标签/" + dataname + "_句子和标签.xls")
sheet = wb.sheet_by_index(0)  # 通过索引的方式获取到某一个sheet，现在是获取的第一个sheet页，也可以通过sheet的名称进行获取，sheet_by_name('sheet名称')
rows = sheet.nrows  # 获取sheet页的行数，一共有几行
# columns = sheet.ncols  # 获取sheet页的列数，一共有几列
samples = [str(sheet.cell(i, 0).value) for i in range(1, rows)]
labels = [str(sheet.cell(i, 1).value) for i in range(1, rows)]
samples = [jieba.lcut(samples[i]) for i in range(len(samples))]
wb2 = xlrd.open_workbook("D:/pythonwork/paper2024/mooc/data/" + dataname + "_nouns&numbers.xls")
sheet2 = wb2.sheet_by_index(0)
rows = sheet2.nrows
attributes = [str(sheet2.cell(i, 0).value).strip() for i in range(1, rows)]
samples_attributes = []
for i in tqdm(range(len(samples))):
    sample_attributes = []
    for j in range(len(attributes)):
        if attributes[j] in samples[i]:
            sample_attributes.append(attributes[j])
    samples_attributes.append(sample_attributes)

book = xlwt.Workbook(encoding='utf-8')
sheet2 = book.add_sheet('sheet1', cell_overwrite_ok=True)
sheet2.write(0, 0, '评论')
sheet2.write(0, 1, '标签')
sheet2.write(0, 2, '关键词')
row = 1
for i in range(len(samples)):
    sheet2.write(row, 0, samples[i])
    sheet2.write(row, 1, labels[i])
    if len(samples_attributes[i]) != 0:
        sheet2.write(row, 2, '/'.join(samples_attributes[i]))
    else:
        sheet2.write(row, 2, "NONE")
    row += 1
book.save('D:/pythonwork/paper2024/mooc/3获取每个句子的属性词/' + dataname + '_句子、标签和属性词.xls')
