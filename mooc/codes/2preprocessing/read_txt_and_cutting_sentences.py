
import re
import xlwt
import jieba
from tqdm import tqdm
import jieba.posseg as psg

file = 'D:/pythonwork/paper2024/mooc/data/test.txt'
with open(file, 'r', encoding='utf-8') as f:
    sentences = [i.strip() for i in f.readlines()]


s = []
for i in tqdm(range(len(sentences))):
    # 段落拆分为句子，除了句子结束符（。，？之类）外还有一些特定的转折词，表示句子前后情感有转折的，尽量确保句子情感前后统一
    list1 = re.split('<br />|而且|但是|但|\s+要是|,要是|，要是|\s+就是|,就是|，就是|只是|可惜|不过|。|！|!|；', sentences[i])
    while '' in list1:
        list1.remove('')
    for j in list1:
        if len(j) >= 5:
            s.append(j)

file_out = 'D:/pythonwork/paper2024/mooc/data/test_句子切分后.txt'
with open(file_out, 'w', encoding='utf-8') as f:
    for i in s:
        f.write(i + '\n')
