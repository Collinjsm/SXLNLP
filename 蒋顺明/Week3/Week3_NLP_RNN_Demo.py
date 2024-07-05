import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
class TorchModel(nn.Module):  # 创建一个继承nn.Module的类
    def __init__(self, vector_dim, sentence_length, vocab):  #vector_dim,每个字的向量维度；sentence_lenght表示样本中的字数；vocab表示字符集
        super(TorchModel, self).__init__()  #调用父类的构造函数
        self.embedding = nn.Embedding(len(vocab)+1, vector_dim)  #Embedding层，将字符映射为向量 6* 20
        self.pool = nn.AvgPool1d(sentence_length) # 池化层，压缩特征 20*6
        # 使用RNN神经网络
        self.rnn = nn.RNN(vector_dim, 5, batch_first=True) # RNN层，输入维度为vector_dim，输出维度为5
        # 分类层，将RNN的输出转为分类结果
        self.classify = nn.Linear(5, len(vocab))  # 线性层，将RNN的输出转为分类结果 5*2

        # 激活函数，使用sigmoid函数
        #self.activation = torch.sigmoid
        # 损失函数，使用交叉熵
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):  #定义前向传播函数
        x = self.embedding(x)
        x = x.transpose(1, 2)
        #x = x.squeeze(0)
        x = self.pool(x)
        x = x.squeeze()
        x, _ = self.rnn(x)
        x = self.classify(x)
        #x = x.squeeze()
        #y_pred = self.activation(x)

        if y is not None:
            return self.loss(x, y)
        else:
            return x
def build_vocab():  #构建字符集
    # 从python路径下读取vocab.txt文件，将文件中的内容读取出来
    with open("vocab.txt", "r") as f:
        # 读取文件中的内容，不是json格式，是字符串格式
        vocab = f.read()
        #将读取出来的字符串进行重复筛选，去掉空格，换行符等
        vocab = list(set(vocab))
        #将读取出来的字符集输入到一个字典中，每个字符对应一个序号，从1开始
        vocab = {char: index+1 for index, char in enumerate(vocab)}
        #将未知字符unk加入到字典中
        vocab["unk"] = len(vocab)+1
        #将字典保存到vocab.json文件中
        # with open("vocab.json", "w") as f:
        #     json.dump(vocab, f)
    return vocab
# 生成一个样本
def build_sample(vocab, sentence_length):
    # 随机从字表选取sentence_length个字，可以重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # 判断是否有“喜欢”、“愤怒”、“平静”、“悲伤”其中的1个字
    if set("喜欢") & set(x):
        y = 0
    elif set("悲伤") & set(x):
        y = 1
    elif set("愤怒") & set(x):
        y = 2
    elif set("平静") & set(x):
        y = 3
    else:
        y = 4
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y

def build_dataset(vocab, total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample(vocab, 6)
        X.append(x)
        Y.append(y)
    return torch.LongTensor(X), torch.LongTensor(Y)

# 生成模型函数
def build_model(vocab, vector_dim, sentence_length): # 
    model = TorchModel(vector_dim, sentence_length, vocab)
    return model

# 测试模型的准确率
def evaluate(model, vocab, sample_length):
    # 进行评估模式
    model.eval()
    # 生成数据
    X, Y = build_dataset(vocab, 100)
    X = torch.LongTensor(X)
    Y = torch.LongTensor(Y)
    print("本次样本有%s个是喜欢的" % (Y == 0).sum().item())
    print("本次样本有%s个多少是悲伤的" % (Y == 1).sum().item())
    print("本次样本有%s个多少是愤怒的" % (Y == 2).sum().item())
    print("本次样本有%s个多少是平静的" % (Y == 3).sum().item())
    print("本次样本有%s个多少是未知的" % (Y == 4).sum().item())
    correct_num ,wrong_num = 0, 0
    with torch.no_grad():
        Y_pred = model(X)
        for y_p, y_t in zip(Y_pred, Y):
            if torch.argmax(y_p) == y_t:
                correct_num += 1
            else:
                wrong_num += 1
    print("正确预测个数：%d, 正确率：%f"%(correct_num, correct_num/(correct_num+wrong_num)))
    return correct_num/(correct_num+wrong_num)

# 主函数
def main():
    # 配置参数
    epoch_num = 20 # 训练轮数
    batch_size = 20 # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.005  # 学习率
    # 构建字符集
    vocab = build_vocab()
    # 构建模型
    model = build_model(vocab, char_dim, sentence_length)
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = [] # 记录准确率
    # 训练过程
    for epoch in range(epoch_num): # 训练轮数
        model.train() # 训练模式
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(vocab, batch_size)
            optimizer.zero_grad()
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, np.mean(watch_loss)])
        print("第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()


    torch.save(model.state_dict(), "model1.pth")
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
# 使用训练好的模型预测
# def predict(model_path, vocab_path, input_strings):
#     # 加载词汇表
#     vocab = json.load(open(vocab_path, "r", encoding="utf8"))
#
#     # 建立模型
#     model = TorchModel(20, 6, vocab)
#
#     # 加载训练好的权重
#     model.load_state_dict(torch.load(model_path))
#
#     # 测试模式
#     model.eval()
#
#     with torch.no_grad():  # 不计算梯度
#         for input_string in input_strings:
#             # 将输入序列化
#             x = [vocab.get(char, vocab['unk']) for char in input_string]
#             x = torch.LongTensor([x]).unsqueeze(0)
#             x= x.expand(-1, -1,20)
#
#             # 模型预测
#             y_pred = model(x)
#
#             # 打印结果
#             print("输入：%s, 预测类别：%s" % (
#             input_string, ["喜欢", "悲伤", "愤怒", "平静", "未知"][torch.argmax(y_pred).item()]))





if __name__ == "__main__":
    main()
    #predict("model1.pth", "vocab.json", 6)
    # string_test = ["我喜欢你啊笨","我很悲伤啊啊","我很愤怒啊啊","我很平静啊啊","我很开心啊啊"]
    # predict("model1.pth", "vocab.json", string_test)
    #

