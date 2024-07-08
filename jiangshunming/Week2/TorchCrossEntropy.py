import torch
import torch.nn as nn
import numpy as np


# 定义一个2层的神经网络模型，激活函数使用softmax，损失函数使用交叉熵
class  TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(TorchModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1) # 第一层线性层
        self.layer2 = nn.Linear(hidden_size1, hidden_size2) # 第二层线性层
        self.activation = torch.softmax # 激活函数
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.layer1(x)
        x = self.layer2(x)
        y_pred = self.activation(x, dim=1)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            # 返回预测结果是1|2|3|4|5|6
            return y_pred


# 定义样本生成函数，生成一个样本，包含6个数，如果[1,0,0,0,0,0]则为数字1，[0,1,0,0,0,0]则为数字2，以此类推
def build_sample():
    x = np.random.random(6) # 生成一个6维向量
    y = np.zeros(6) # 生成一个6维向量
    y[np.argmax(x)] = 1 # 将最大值的位置设置为1
    return x, y

# 批量生成样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)

# 定义评估函数
def evaluate(model):
    # 生成数据
    X, Y = build_dataset(100)
    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(Y)
    # 模型进入评估模式
    model.eval()
    # 输出每个样本的个数，数字1的样本个数，数字2的样本个数，数字3的样本个数，数字4的样本个数，数字5的样本个数，数字6的样本个数
    print("样本总数：", len(Y))
    print("数字1的样本个数：", len(Y[Y[:, 0] == 1]))
    print("数字2的样本个数：", len(Y[Y[:, 1] == 1]))
    print("数字3的样本个数：", len(Y[Y[:, 2] == 1]))
    print("数字4的样本个数：", len(Y[Y[:, 3] == 1]))
    print("数字5的样本个数：", len(Y[Y[:, 4] == 1]))
    print("数字6的样本个数：", len(Y[Y[:, 5] == 1]))
    # 计算预测值
    y_pred = model(X)
    # 计算损失
    loss = model(X, Y)
    print("损失值：", loss.item())
    # 计算准确率
    correct = 0
    wrong = 0
    for y_pred, y in zip (y_pred, Y):
        if torch.argmax(y_pred) == torch.argmax(y): # 计算正确个数
            correct += 1 # 计算正确个数
        else:
            wrong += 1 # 计算错误个数
    print("正确个数：", correct)
    print("错误个数：", wrong)
    print("准确率：", correct / (correct + wrong))
    return correct / (correct + wrong)


# 训练模型
def train(model, optimizer, total_sample_num):
    X, Y = build_dataset(total_sample_num) # 生成数据
    X = torch.FloatTensor(X) # 转换数据类型
    Y = torch.FloatTensor(Y) # 转换数据类型
    model.train() # 模型进入训练模式
    log = [] # 保存日志
    for epoch in range(1000):
        optimizer.zero_grad() # 梯度清零
        loss = model(X, Y) # 计算损失
        loss.backward() # 反向传播：计算梯度
        optimizer.step() # 更新参数：梯度下降
        if epoch % 100 == 0: # 每100轮打印一次
            print("epoch:", epoch, "loss:", loss.item()) # 打印损失值
            evaluate(model) # 打印准确率
            log.append([evaluate(model), loss.item()]) # 保存准确率和损失值


    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 使用图画出损失值和准确率
    import matplotlib.pyplot as plt
    log = np.array(log)
    plt.plot(range(len(log)), log[:, 0], label="acc")
    plt.plot(range(len(log)), log[:, 1], label="loss")
    plt.legend()
    plt.show()


def main():
    # 配置参数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 6  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size, 10, 6)
    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 训练模型
    train(model, optimizer, train_sample)
    return

# 定义函数用于预测y_pred
def predict(model_path, input_vector):
    input_size = 6
    hidden_size1 = 10
    hidden_size2 = 6
    model = TorchModel(input_size, hidden_size1, hidden_size2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # 不计算梯度
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vector))
    # 循环输入的向量和预测结果是分类的类别，预测概率是多少
    for i, r in enumerate(result):
        # argmax打印出最大值的索引，索引0代表类别1，索引1代表类别2，以此类推
        # 将torch.argmax转换为对应的类别
        print("输入向量：", input_vector[i], "预测类别：", torch.argmax(r).item() + 1, "预测概率：", r[torch.argmax(r)].item())




# 运行主函数
if __name__ == '__main__':
    # 选择是训练模型或者预测模型
    # 输入选择1，训练模型；2，预测模型
    while True:
        print("1.训练模型")
        print("2.预测模型")
        print("3.退出")
        choice = int(input("请输入选择："))
        if choice == 1:
            main()
        elif choice == 2:
            # 自定义一个样本, 生成10个样本，包含6个数，如果[1,0,0,0,0,0]则为数字1，[0,1,0,0,0,0]则为数字2，以此类推
            input_vector = [[1,0,0,0,0,0],
                            [0,1,0,0,0,0],
                            [0,0,1,0,0,0],
                            [0,1,0,0,0,0],
                            [0,0,0,0,1,0],
                            [0,1,0,1,0,0],]
            predict("model.pth", input_vector)

    # main()










