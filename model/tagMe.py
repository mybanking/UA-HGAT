#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import json
import requests
from tqdm import tqdm
from multiprocessing import Pool as ProcessPool
# from py2neo import Graph, Node, Relationship,NodeMatcher ,RelationshipMatcher
# import json
# import jieba

time0 = time.time()

dataset = 'example'
path2 = './data/stopwords_en.txt'

def getEntityList(text):
    url = 'https://tagme.d4science.org/tagme/tag'
    # token = '14a339a1-6913-4e8a-bff3-da6ae4459381-843339462'
    token = 'fe4df7bf-ab75-4efb-aa1c-551afaa65cd3-843339462'
    para = {'lang': 'en',
           'gcube-token': token,
           'text': text,
    }
    headers = {
        'Host': 'tagme.d4science.org',
        'Connection': 'keep-alive',
        'Cache-Control': 'max-age=0',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Cookie': '_ga=GA1.2.827175290.1544765315; _gid=GA1.2.121830695.1544765315',
    }
    response = requests.get(url+'?lang=en&gcube-token='+token+'&text='+text, headers=headers, timeout=10)
    try:
        return json.loads(response.text)['annotations']
    except:
        print('.'*50)
        print(text)
        print(response.text)
        print('.'*50)

def sentence2Link(sentence):
    return json.dumps(getEntityList(sentence))


def run(para, times = 0):
    ind = para[0]
    sentence = para[1]
    try:
        l = sentence2Link(sentence)
    except Exception as e:
        print(ind, e)
        print("Content: ", sentence)
        if times < 5:
            return run(para, times + 1)
        else:
            with open("error_info.txt", 'w+') as f:
                f.write("{}\t{}\n".format(ind, sentence))
            return None
    print(ind)
    return str(ind)+'\t'+l


def process_pool(data):
    cnt = 0
    p = ProcessPool(32)
    chunkSize = 128
    res = []
    i = 0
    while i < int(len(data)/chunkSize):
        try:
            res += list(p.map(run, data[i*chunkSize: (i+1)*chunkSize]))
            print(str(round((i+1)*chunkSize/len(data)*100, 2))+'%', round(time.time()-time0, 2))
            cnt += 1
            # fout = open("cache" + str(cnt).zfill(3) + '.txt', 'w', encoding='utf8')
            # fout.write('\n'.join(res))
            # fout.close()
            i += 1
            time.sleep(1.0)
        except:
            for i in range(60):
                time.sleep(10)
                print("\t{} / 600s".format(i*10))
        
    res += list(p.map(run, data[(i)*chunkSize:]))
    p.close()
    p.join()
    return res


def getEntityFromNeo4j(data):
    """建立连接"""
    graph = Graph("http://10.10.2.102:7474/", auth=("neo4j", "root"))

    # 停用词
    stopwords = [line.strip() for line in open(path2, encoding="utf-8").readlines()]  # 加载停用词

    counts = {}  # 计数{word，frequency}
    for index,seg in data:
       word_list = jieba.cut(seg, cut_all=False)
       for word in word_list:
           if word not in stopwords:
               if len(word) == 1:
                   continue
               else:
                   counts[word] = counts.get(word,0) + 1
    items = list(counts.items())
    items.sort(key=lambda x: x[1], reverse=True)

    for i in range(30): # 输出词频统计前30项
        word, count = items[i]
        print("{:<10}{:>7}".format(word, count))


    # matcher = NodeMatcher(graph)
    #
    # a = matcher.match(
    #     name='London'
    # )
    #
    #
    # for b in a:
    #     relationship_matcher = RelationshipMatcher(graph)
    #
    #     relationship = list(relationship_matcher.match([b], r_type=None))
    #
    #     print(relationship)
    #
    #     relationship = list(relationship_matcher.match((None, b), r_type=None))
    #
    #     print(relationship[0])



if __name__ == "__main__":
    print('reading...')
    data = []
    with open("./data/{0}/{0}.txt".format(dataset), 'r', encoding='utf8') as fin:
        for line in fin:
            ind, cate, content = line.split('\t')
            if int(ind) > -1 and int(ind) < 9e19:
                data.append([ind, content])

    print('read done. Tagging...')
    outdata = process_pool(data)
    fout = open("./data/{0}/{0}2entity.txt".format(dataset), 'w', encoding='utf8')
    fout.write('\n'.join(outdata))
    fout.close()
