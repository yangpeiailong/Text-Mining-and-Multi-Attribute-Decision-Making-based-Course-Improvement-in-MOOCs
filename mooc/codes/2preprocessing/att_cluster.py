from fasttext import load_model
import xlrd
from sklearn.cluster import AffinityPropagation
import numpy as np

dataname = "C_language_programming"
ft = load_model('D:/fasttext/cc.zh.300.bin')
wb = xlrd.open_workbook("D:/pythonwork/paper2024/mooc/data/1attributes/" + dataname + "_nouns&numbers.xls")
sheet = wb.sheet_by_index(0)  # 通过索引的方式获取到某一个sheet，现在是获取的第一个sheet页，也可以通过sheet的名称进行获取，sheet_by_name('sheet名称')
rows = sheet.nrows  # 获取sheet页的行数，一共有几行

nouns = []
for i in range(1, rows):
    nouns.append(str(sheet.cell(i, 0).value).strip())

nouns2vec = []
for i in nouns:
    nouns2vec.append(ft.get_word_vector(i))

nouns2vec = np.array(nouns2vec)
ap = AffinityPropagation()
ap.fit(nouns2vec)

labels = ap.labels_
nouns_clusters = {}
for i in range(len(nouns)):
    if labels[i] in nouns_clusters.keys():
        nouns_clusters[labels[i]].append(nouns[i])
    else:
        nouns_clusters[labels[i]] = [nouns[i]]

print(nouns_clusters)
