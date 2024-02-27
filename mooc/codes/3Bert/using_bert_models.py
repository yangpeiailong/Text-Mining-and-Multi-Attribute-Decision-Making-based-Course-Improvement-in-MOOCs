# -*- coding:utf-8 -*-
# bert文本分类模型
# model: bert
# date: 2022.3.29 10:36

import numpy as np
import jieba
import torch
import torch.nn as nn
import torch.utils.data as Data
from transformers import BertModel, BertTokenizer
import xlwt
from tqdm import tqdm
import re
import xlrd


def get_words(samples_temp):
    words_temp = []
    for line_temp in samples_temp:
        tmp_list = list(line_temp.split(' '))
        for word in tmp_list:
            if str(word) not in words_temp:
                words_temp.append(word)
    return words_temp


class MyDataset(Data.Dataset):
    def __init__(self, sentences, labels=None, with_labels=True, ):
        self.sentences = sentences
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.with_labels = with_labels
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        # Selecting sentence1 and sentence2 at the specified index in the data frame
        sent = self.sentences[index]

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(sent,
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,  # Truncate to max_length
                                      max_length=maxlen,
                                      return_tensors='pt')  # Return torch.Tensor objects
        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(
            0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(
            0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens
        if self.with_labels:  # True if the dataset has labels
            label = self.labels[index]
            return token_ids, attn_masks, token_type_ids, label
        else:
            return token_ids, attn_masks, token_type_ids


# model1
class BaseBert(nn.Module):
    def __init__(self):
        super(BaseBert, self).__init__()
        self.bert = BertModel.from_pretrained(model, output_hidden_states=True, return_dict=True)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(hidden_size, n_class)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X):
        input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # 返回一个output字典
        # 用最后一层cls向量做分类
        # outputs.pooler_output: [batch_size, hidden_size]
        logits = self.softmax(self.linear(self.dropout(outputs.pooler_output)))
        # logits = self.linear(self.dropout(outputs.pooler_output))
        return logits


# model2
class BertCNN(nn.Module):
    def __init__(self):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained(model, output_hidden_states=True, return_dict=True)
        self.dropout = nn.Dropout(0.5)
        # self.linear = nn.Linear(hidden_size, n_class)
        self.convs = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(in_channels=hidden_size, out_channels=num_of_filters, kernel_size=h),
                           nn.ReLU(),
                           nn.MaxPool1d(maxlen - h + 1)
                           ) for h in window_sizes]
        )
        self.linear = nn.Linear(num_of_filters * len(window_sizes), n_class)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X):
        input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # 返回一个output字典
        # 用最后一层cls向量做分类
        # outputs.pooler_output: [batch_size, hidden_size]
        outputs = self.dropout(outputs.last_hidden_state)
        outputs = outputs.permute(0, 2, 1)
        outputs = [conv(outputs) for conv in self.convs]
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.view(-1, outputs.size(1))
        outputs = self.linear(outputs)
        return self.softmax(outputs)


# model3
class BertLSTM(nn.Module):
    def __init__(self):
        super(BertLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(model, output_hidden_states=True, return_dict=True)
        self.dropout = nn.Dropout(0.5)
        # self.linear = nn.Linear(hidden_size, n_class)
        self.lstm = nn.LSTM(hidden_size, hidden_size_lstm, bidirectional=True)

        self.f1 = nn.Sequential(nn.Linear(hidden_size_lstm * 2, 128),
                                nn.Dropout(0.8),
                                nn.ReLU(),
                                nn.Linear(128, n_class),
                                nn.Softmax(dim=-1)
                                )

    def forward(self, X):
        input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # 返回一个output字典
        # 用最后一层cls向量做分类
        # outputs.pooler_output: [batch_size, hidden_size]
        x = self.dropout(outputs.last_hidden_state)
        x = x.permute(1, 0, 2)
        h0 = torch.randn(2, x.size(1), hidden_size_lstm).cuda()
        c0 = torch.randn(2, x.size(1), hidden_size_lstm).cuda()
        output, (hn, cn) = self.lstm(x, (h0, c0))
        # output, (hn, cn) = self.lstm(x)
        # print(hn.shape())
        # hn = torch.cat((hn[0], hn[1]), 1)
        # x = self.f1(hn)
        x = self.f1(output[-1])
        return x


# model4
class BertLSTMAttention(nn.Module):
    def __init__(self):
        super(BertLSTMAttention, self).__init__()
        self.bert = BertModel.from_pretrained(model, output_hidden_states=True, return_dict=True)
        self.dropout = nn.Dropout(0.5)
        # self.linear = nn.Linear(hidden_size, n_class)
        self.lstm = nn.LSTM(hidden_size, hidden_size_lstm, bidirectional=True)

        # 定义计算注意力权重的内容
        self.linear1 = nn.Linear(hidden_size_lstm * 2, 128)
        self.tanh = nn.Tanh()
        self.u_w = nn.Linear(128, 1)
        self.softmax1 = nn.Softmax(dim=-1)

        # 定义输出
        self.f1 = nn.Sequential(nn.Linear(hidden_size_lstm * 2, 128),
                                nn.Dropout(0.8),
                                nn.ReLU(),
                                nn.Linear(128, n_class),
                                nn.Softmax(dim=-1)
                                )

    def forward(self, X):
        input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # 返回一个output字典
        # 用最后一层cls向量做分类
        # outputs.pooler_output: [batch_size, hidden_size]
        x = self.dropout(outputs.last_hidden_state)
        x = x.permute(1, 0, 2)
        h0 = torch.randn(2, x.size(1), hidden_size_lstm).cuda()
        c0 = torch.randn(2, x.size(1), hidden_size_lstm).cuda()
        output, (hn, cn) = self.lstm(x, (h0, c0))
        output = output.permute(1, 0, 2)
        # (seq_len, batchsize, hidden_size*num_directions) => (batchsize, seq_len, hidden_size*num_directions)

        attention_u = self.tanh(self.linear1(output))
        # (batchsize, seq_len, hidden_size*num_directions) => (batchsize, seq_len, u_size)
        attention_a = self.softmax1(self.u_w(attention_u))
        # print(attention_a.shape, output.shape)
        # (batchsize, seq_len, u_size) * (u_size, 1) => (batchsize, seq_len, 1)
        output = torch.matmul(attention_a.permute(0, 2, 1), output).squeeze()
        # (batchsize, 1, seq_len) * (batchsize, seq_len, hidden_size * num_directions) =>(batchsize, hidden_size * num_directions)
        return self.f1(output)


class BertLSTMCNN(nn.Module):
    def __init__(self):
        super(BertLSTMCNN, self).__init__()
        self.bert = BertModel.from_pretrained(model, output_hidden_states=True, return_dict=True)
        self.dropout = nn.Dropout(0.5)
        # self.linear = nn.Linear(hidden_size, n_class)
        self.lstm = nn.LSTM(hidden_size, hidden_size_lstm, bidirectional=True)

        self.convs = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(in_channels=hidden_size + 2 * hidden_size_lstm, out_channels=num_of_filters, kernel_size=h),
                           nn.ReLU(),
                           nn.MaxPool1d(maxlen - h + 1)
                           ) for h in window_sizes]
        )
        self.linear = nn.Linear(num_of_filters * len(window_sizes), n_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # 返回一个output字典
        # 用最后一层cls向量做分类
        # outputs.pooler_output: [batch_size, hidden_size]
        x = self.dropout(outputs.last_hidden_state)
        input = x.permute(1, 0, 2)
        h0 = torch.randn(2, input.size(1), hidden_size_lstm).cuda()
        c0 = torch.randn(2, input.size(1), hidden_size_lstm).cuda()
        output, (hn, cn) = self.lstm(input, (h0, c0))
        output = output.permute(1, 0, 2)
        # print(output.shape)
        outputsplit1, outputsplit2 = output.chunk(2, dim=2)
        # print(x.shape)
        outputcat = torch.cat((outputsplit1, x, outputsplit2), dim=2)
        outputcat = outputcat.permute(0, 2, 1)
        x = [conv(outputcat) for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = x.view(-1, x.size(1))
        x = self.linear(x)
        return self.softmax(x)


if __name__ == '__main__':
    train_curve = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size_training = 32
    batch_size_test = 100
    epoches = 40
    model = "bert-base-chinese"
    hidden_size = 768


    # data
    # sentences = ["我喜欢打篮球", "这个相机很好看", "今天玩的特别开心", "我不喜欢你", "太糟糕了", "真是件令人伤心的事情"]
    # labels = [1, 1, 1, 0, 0, 0]  # 1积极, 0消极.
    dataname = '大学英语_without_labels'
    # 加载数据集
    with open("E:/2022慕课评论文本挖掘/" + dataname + ".txt", 'r', encoding='utf-8') as f:
        samples = f.readlines()
    # wb = xlrd.open_workbook("E:/2022慕课评论文本挖掘/C语言_without_labels.txt")
    # sheet = wb.sheet_by_index(0)  # 通过索引的方式获取到某一个sheet，现在是获取的第一个sheet页，也可以通过sheet的名称进行获取，sheet_by_name('sheet名称')
    # rows = sheet.nrows  # 获取sheet页的行数，一共有几行
    # columns = sheet.ncols  # 获取sheet页的列数，一共有几列
    # sample_data_file = 'E:/数据/' + dataname + '.txt'
    # label_data_file = 'E:/数据/' + dataname + '_label.txt'
    # samples = [str(sheet.cell(i, 1).value) for i in range(1, rows)]
    # times = [str(sheet.cell(i, 0).value) for i in range(1, rows)]
    samples = [s.strip() for s in samples]
    # samples = [jieba.lcut(sample) for sample in samples]

    maxlen = min([max([len(sample) for sample in samples]), 200])
    # maxlen = 50
    label_to_idx = {}
    with open('E:/2022慕课评论文本挖掘/model/label_to_index_bert_for_' + 'C语言' + '.txt', 'r', encoding='utf-8') as f:
        for i in f.readlines():
            temp = i.strip().split(':')
            key = temp[0]
            value = temp[1]
            label_to_idx[key] = int(value)
        f.close()

    idx_to_label = {label_to_idx[key]: key for key in label_to_idx.keys()}
    n_class = len(set(label_to_idx.keys()))




    num_of_filters = 100  # 卷积核个数
    window_sizes = [2, 3, 4]  # 卷积核尺寸
    hidden_size_lstm = 100  # lstm的隐藏层尺寸
    classifier = 'basebert'
    bc = 0
    if classifier == 'basebert':
        bc = BaseBert().to(device)
    elif classifier == 'bert+cnn':
        bc = BertCNN().to(device)
    elif classifier == 'bert+lstm':
        bc = BertLSTM().to(device)
    elif classifier == 'bert+lstm+att':
        bc = BertLSTMAttention().to(device)
    elif classifier == 'bert+lstm+cnn':
        bc = BertLSTMCNN().to(device)

    bc.load_state_dict(torch.load('E:/2022慕课评论文本挖掘/model/model_params_best_C语言_basebert_3.pkl'))
    using = MyDataset(samples, labels=None, with_labels=False)
    results = []
    with torch.no_grad():
        for i in tqdm(range(len(samples))):
            x = using.__getitem__(i)
            x = tuple(p.unsqueeze(0).to(device) for p in x)
            pred = bc([x[0], x[1], x[2]])
            pred = pred.cpu().numpy().tolist()
            pred = np.array(pred)
            y_pred = np.argmax(pred, 1)
            results.append(idx_to_label[y_pred[0]])

    # book = xlwt.Workbook(encoding='utf-8')
    # sheet1 = book.add_sheet('sheet1', cell_overwrite_ok=True)
    # sheet1.write(0, 0, '时间')
    # sheet1.write(0, 1, '评论')
    # sheet1.write(0, 2, '标签')
    # row = 1
    # for i in range(len(samples)):
    #     sheet1.write(row, 0, times[i])
    #     sheet1.write(row, 1, samples[i])
    #     sheet1.write(row, 2, results[i])
    #     row += 1
    # book.save('E:/成都理工大学重要文件夹/韩冬老师项目/3模型打标签后的数据/携程_时间、句子和标签.xls')
    with open('E:/2022慕课评论文本挖掘/classification_results/labels_bert_for_' + dataname + '.txt', 'w',
              encoding='utf-8') as f:
        for i in results:
            f.writelines(str(i) + '\n')
        f.close()



