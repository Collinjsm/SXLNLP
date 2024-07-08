import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
class TorchModel(nn.Module):  # 创建一个继承nn.Module的类
    def __init__(self, hidden_size, output_size, vocab, vector_dim):  #vector_dim,每个字的向量维度；sentence_lenght表示样本中的字数；vocab表示字符集
        super(TorchModel, self).__init__()  #调用父类的构造函数
        self.embedding = nn.Embedding(len(vocab)+1, vector_dim)  #Embedding层，将字符映射为向量 6* 20
        #self.pool = nn.AvgPool1d(sentence_length) # 池化层，压缩特征 20*6
        # 使用RNN神经网络
        self.rnn = nn.RNN(vector_dim, hidden_size, batch_first=True) # RNN层，将向量映射为隐藏层 20*5*15-》20*5*5
        self.layer = nn.Linear(hidden_size, output_size)  # 线性层，将RNN的输出转化为类别 5*5
        # self.activation = torch.sigmoid
        # 损失函数，使用交叉熵
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):  #定义前向传播函数
        x = self.embedding(x)
        # x = x.transpose(1, 2)
        # x = self.pool(x)
        # x = x.squeeze()
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        x = self.layer(x)
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
        with open("vocab.json", "w") as f:
            json.dump(vocab, f)
    return vocab
# 生成一个样本
def build_sample(vocab, sentence_length):
    #  从字符集中随机生成一句6个字的句子
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # 设置类别，喜欢，悲伤，愤怒，平静，未知
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
    # 将字符转化为对应的序号
    x = [vocab.get(word, vocab["unk"]) for word in x]
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
def build_model(vocab, vector_dim): #
    model = TorchModel(10,5, vocab, vector_dim)
    return model

# 测试模型的准确率
def evaluate(model, vocab):
    # 进行评估模式
    model.eval()
    # 生成数据
    x, y = build_dataset(vocab, 100)
    # X = torch.LongTensor(X)
    # Y = torch.LongTensor(Y)
    print("本次样本有多少是喜欢的%s" % (y == 0).sum().item())
    print("本次样本有多少是悲伤的%s" % (y == 1).sum().item())
    print("本次样本有多少是愤怒的%s" % (y == 2).sum().item())
    print("本次样本有多少是平静的%s" % (y == 3).sum().item())
    print("本次样本有多少是未知的%s" % (y == 4).sum().item())
    correct_num, wrong_num = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if torch.argmax(y_p) == y_t:
                correct_num += 1
            else:
                wrong_num += 1
    print("正确预测个数：%d, 正确率：%f" % (correct_num, correct_num/(correct_num+wrong_num)))
    return correct_num/(correct_num+wrong_num)

# 主函数
def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    # batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 15  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.05  # 学习率
    # 构建字符集
    vocab = build_vocab()
    # 构建模型
    model = build_model(vocab, char_dim)
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = [] # 记录准确率
    # 训练过程
    for epoch in range(epoch_num): # 训练轮数
        model.train() # 训练模式
        # 生成数据
        x, y = build_dataset(vocab, train_sample) # 生成数据
        #X = torch.LongTensor(X) # 转化为tensor
        #Y = torch.LongTensor(Y) #
        # 训练
        optimizer.zero_grad() # 梯度归零
        loss = model(x, y) # 计算损失
        loss.backward() # 计算梯度
        optimizer.step() # 更新权重
        #log.append(loss.item())
        print("epoch:%d, loss:%f" % (epoch, loss.item()))
        acc = evaluate(model, vocab)
        print("准确率为：%f" % acc)
        #loss和acc的变化放在log中
        log.append((loss.item(), acc))
    # 画图
    log = np.array(log)
    plt.plot(log[:, 0], label="loss")
    plt.plot(log[:, 1], label="acc")
    plt.legend()
    plt.show()

    # 保存模型
    torch.save(model.state_dict(), "model1.pth")
    # 保存字符集
    with open("vocab.json", "w") as f:
        json.dump(vocab, f)

# 使用训练好的模型预测
def predict(model_path, vocab_path,vocab_dim, test_data):
    # 读取字符集
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    # 读取模型
    model = build_model(vocab, vocab_dim)
    model.load_state_dict(torch.load(model_path))

    # 将输入的字符串转化为序号
    x = []
    for input_string in test_data:
        x.append([vocab.get(char, vocab["unk"]) for char in input_string])
    x = torch.LongTensor(x)
    # 预测
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        for y in y_pred:
            pred_class = torch.argmax(y).item()
            pred_prob = torch.max(torch.nn.functional.softmax(y, dim=0)).item()
            print(f"预测类别：{pred_class}, 预测概率：{pred_prob}")





if __name__ == "__main__":
     # main()
    # predict("model1.pth", "vocab.json", 6, 15)
    # 生成数据
    # vocab = build_vocab()
    # build_dataset(vocab, 10)
    # 请帮我生成测试数据，符合rnn的输入要求
    test_data = ["喜欢喜欢喜欢", "我很悲伤难过", "喜欢喜欢喜欢"]
    predict("model1.pth", "vocab.json", 15, test_data)

