import xlrd

wb = xlrd.open_workbook("D:/pythonwork/paper2024/mooc/属性/C_language_programming_关键词归类2.xls")
sheet = wb.sheet_by_index(0)  # 通过索引的方式获取到某一个sheet，现在是获取的第一个sheet页，也可以通过sheet的名称进行获取，sheet_by_name('sheet名称')
rows = sheet.nrows  # 获取sheet页的行数，一共有几行

attribute_words = {}
for i in range(1, rows):
    attribute = str(sheet.cell(i, 0).value).strip()
    words = str(sheet.cell(i, 1).value).strip().split()
    attribute_words[attribute] = words


# 1.判断有没有一个词出现在多个属性，输出这个词
# for i in attribute_words.keys():
#     for j in attribute_words.keys():
#         if i != j:
#             for l in attribute_words[i]:
#                 if l in attribute_words[j]:
#                     print(l)
#                     print(i, j)


# 2.判断一个词是否重复出现在一个属性中
def find_duplicates(lst):
    # 使用字典记录元素出现的次数
    count_dict = {}
    duplicates = []

    # 遍历列表，记录每个元素出现的次数
    for item in lst:
        if item in count_dict:
            count_dict[item] += 1
        else:
            count_dict[item] = 1

            # 找出重复的元素
    for item, count in count_dict.items():
        if count > 1:
            duplicates.append(item)

    return duplicates

for i in attribute_words.keys():
    a_list = attribute_words[i]
    b_list = list(set(a_list))
    if len(a_list) != len(b_list):
        print(i)
        print(find_duplicates(a_list))