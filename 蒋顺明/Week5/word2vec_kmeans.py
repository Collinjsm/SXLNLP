#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model): # 输出要处理的句子和模型
    vectors = [] #创建一个列表存储文本向量
    for sentence in sentences: #循环输入的句子集合
        words = sentence.split()  #句子集合分词
        vector = np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word] #使用model的wv属性获取词向量
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))  #加和平均
    return np.array(vectors) #将输入的句子转化为了向量


def main():
    model = load_word2vec_model("model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list) #定义一个字典存储标签和句子
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
    #这里的距离计算采用了欧式距离
    #计算类内平均距离
    mean_distance = {}
    for label, sentences in sentence_label_dict.items():
        #vectors = sentences_to_vectors(sentences, model)
        mean_distance[label] = np.mean([np.linalg.norm(vector - kmeans.cluster_centers_[label]) for vector in vectors])
    # 舍弃类内平均距离较远的类别
    for label, distance in mean_distance.items():
        if distance > np.percentile(list(mean_distance.values()), 75):
            print("舍弃类别：%s" % label)
            sentence_label_dict.pop(label)


    for label, sentences in sentence_label_dict.items():
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()

