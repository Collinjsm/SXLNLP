#week3作业

#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}

#待切分文本
sentence = "经常有意见分歧"

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    target = [] # 用来存储切分结果
    if sentence == "": # 如果句子为空，直接返回空列表
        return [] # 返回空列表
    for i in range(1, len(sentence)+1): # 遍历句子的每个位置
        if sentence[:i] in Dict: # 如果当前位置的词在词典中
            if i == len(sentence): # 如果当前位置是句子的末尾
                target.append(sentence) # 将整个句子作为一个词加入结果
                return target # 返回结果
            for j in all_cut(sentence[i:], Dict): # 递归调用函数，切分剩余部分
                target.append(sentence[:i] + " " + j) # 将当前位置的词和剩余部分的切分结果拼接
    return target # 返回结果

result = all_cut(sentence, Dict)
# 每个值输出一行
for i in result:
    print(i)

