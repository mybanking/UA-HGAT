## 导入psycopg2包
import psycopg2
from tqdm import tqdm
import networkx
from utils import sample
from nltk.tokenize import WordPunctTokenizer
import matplotlib.pyplot as plt
from utils import sample, preprocess_corpus_notDropEntity, load_stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import collections
import pickle as pkl
import json, os
from bert_serving.client import BertClient

dataset = 'agnews'
path = './data/' + dataset + '/'
rootpath = './'
outpath = rootpath + 'model/data/{}/'.format(dataset)
datapath = rootpath + 'data/{}/'.format(dataset)
map_path = './model/data/' + dataset + '/'

NumOfTrainTextPerClass = 50
TOPK = 10
TopK_for_Topics = 2
# 构造图 实体图 从知识图谱中构造
conn = psycopg2.connect(database="cobweb-kdy", user="postgres",
                        password="postgres", host="10.10.2.52", port="5432")
stopwords = load_stopwords()
def init():


    g = networkx.Graph()

    ###打乱数据集 改成True
    train, vali, test, alltext = sample(datapath=datapath, DATASETS=dataset, resample=False,
                                        trainNumPerClass=NumOfTrainTextPerClass)
    # load text-entity
    entitySet = set()
    with open(path + dataset + ".txt") as f:
        for line in tqdm(f):
            cache = line.strip().split('\t')
            indx = cache[0]
            label = cache[1]
            text = cache[2]
            text = readEntityFromPg(text)

            #文本 和实体
            g.add_edges_from([(indx, e[0])
                                  for e in text])

            # 知识图谱
            g.add_edges_from([(e[0], e[2],{"r":e[1]})
                                  for e in text])

            # fig, ax = plt.subplots()
            # networkx.draw(g, ax=ax, with_labels=True)
            # plt.show()

    lda_feature ,lda,idxlist= build_topic_feature_sklearn(datapath, dataset, stopwords=stopwords, train=True)

    topic_num = lda.components_.shape[0]

    topics = []
    for i in range(topic_num):
        topicName = 'topic_' + str(i)
        topics.append(topicName)
    topK_topics = naive_arg_topK(lda_feature, TopK_for_Topics, axis=1)

    for i in range(topK_topics.shape[0]):
        for j in range(TopK_for_Topics):
            g.add_edge(idxlist[i], topics[topK_topics[i, j]])

    text_nodes, entity_nodes, topic_nodes = cnt_nodes(g)
    print("gnodes:", len(g.nodes()), "gedges:", len(g.edges()))

    # build Edges data
    cnt = 0
    nodes = g.nodes()
    graphdict = collections.defaultdict(list)
    for node in tqdm(nodes):
        try:
             cache = [j for j in g[node]]
             if len(cache) != 0:
                 graphdict[node] = cache
             cnt += len(cache)

        except:
            print(g[node])
            break
    print('edges: ', cnt)

    text_nodes, entity_nodes, topic_nodes = cnt_nodes(g)

    mapindex = dict()
    cnt = 0
    for i in text_nodes | entity_nodes | topic_nodes:
        mapindex[i] = cnt
        cnt += 1
    print(len(g.nodes()), len(mapindex))

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # build feature data
    gnodes = set(g.nodes())
    print(gnodes, mapindex)
    with open(outpath + 'train.map', 'w') as f:
        f.write('\n'.join([str(mapindex[i]) for i in train if i in gnodes]))
    with open(outpath + 'vali.map', 'w') as f:
        f.write('\n'.join([str(mapindex[i]) for i in vali if i in gnodes]))
    with open(outpath + 'test.map', 'w') as f:
        f.write('\n'.join([str(mapindex[i]) for i in test if i in gnodes]))


    # topic node
    flag_zero = False
    content = dict()
    for i in range(topic_num):
        #         zero_num = textShape[1] + entityFlen - topic_num
        topicName = topics[i]
        if topicName not in topic_nodes:
            continue
        one_hot = [0] * topic_num
        one_hot[i] = 1
        content[topicName] = one_hot
        content[topicName] = lda.components_[i].tolist() + ['topic']


    with open(outpath + '{}.content.topic'.format(dataset), 'w') as f:
        for ind in tqdm(content):
            f.write(str(mapindex[ind]) + '\t' + '\t'.join(map(str, content[ind])) + '\n')

    print("共{}个主题".format(len(content)))

    # save mappings
    with open(outpath+'mapindex.txt', 'w') as f:
        for i in mapindex:
            f.write("{}\t{}\n".format(i, mapindex[i]))
    # save adj matrix
    with open(outpath + '{}.cites'.format(dataset), 'w') as f:
        doneSet = set()
        nodeSet = set()
        for node in graphdict:
            for i in graphdict[node]:
                if (node, i) not in doneSet:
                    f.write(str(mapindex[node]) + '\t' + str(mapindex[i]) + '\n')
                    doneSet.add((i, node))
                    doneSet.add((node, i))
                    nodeSet.add(node)
                    nodeSet.add(i)
        for i in tqdm(range(len(mapindex))):
            f.write(str(i) + '\t' + str(i) + '\n')

    print('Num of nodes with edges: ', len(nodeSet))
    text_featureG()
    entity_feature(entity_nodes)

def naive_arg_topK(matrix, K, axis=0):
    """
    perform topK based on np.argsort
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: dimension to be sorted.
    :return:
    """
    full_sort = np.argsort(-matrix, axis=axis)
    return full_sort.take(np.arange(K), axis=axis)




def cnt_nodes(g):
    text_nodes, entity_nodes, topic_nodes = set(), set(), set()
    for i in g.nodes():
        if i.isdigit():
            text_nodes.add(i)
        elif i[:6] == 'topic_':
            topic_nodes.add(i)
        else:
            entity_nodes.add(i)
    print("# text_nodes: {}     # entity_nodes: {}     # topic_nodes: {}".format(
            len(text_nodes), len(entity_nodes), len(topic_nodes)))
    return text_nodes, entity_nodes, topic_nodes


# 从知识图谱上获取实体
def readEntityFromPg(text):

    text_list = text.lower().split(" ")


    ## 建立游标，用来执行数据库操作
    cursor = conn.cursor()

    sql = "select su,re,ob from kg where su in "

    value ="("
    for i in  range(0,len(text_list)):
        if i == len(text_list)-1:
            value = value +"%s"
        else:
            value = value +"%s,"

    value = value +")"

    sql = sql+value
    #
    # sql = sql+"or ob in"+value

    text_list = tuple(text_list)
    # text_list = text_list*2
    ## 执行SQL SELECT命令
    cursor.execute(sql,text_list)

    ## 获取SELECT返回的元组
    rows =set(cursor.fetchall())





    ## 关闭游标
    cursor.close()
    return  rows




# 从知识图谱上获取
def readFromPg(text):

    text_list = text.lower().split(" ")


    ## 建立游标，用来执行数据库操作
    cursor = conn.cursor()

    sql = "select su,re,ob from kg where su in "

    value ="("
    for i in  range(0,len(text_list)):
        if i == len(text_list)-1:
            value = value +"%s"
        else:
            value = value +"%s,"

    value = value +")"

    sql = sql+value
    #
    # sql = sql+"or ob in"+value

    text_list = tuple(text_list)
    # text_list = text_list*2
    ## 执行SQL SELECT命令
    cursor.execute(sql,text_list)

    ## 获取SELECT返回的元组
    rows =set(cursor.fetchall())



    result = ""
    for row in rows:
        result = result + " "+row[0]+" "+row[1]+" "+row[2]

    ## 关闭游标
    cursor.close()


    return result

def text_featureG():

    mapIndex = dict()
    with open(map_path + "mapindex.txt") as map:

        for line in tqdm(map):
            cache = line.strip().split('\t')
            mapIndex[cache[0]] = cache[1]

    map.close()

    bc = BertClient()

    with open(path + dataset + ".txt") as f, open(outpath + '{}.content.text'.format(dataset), 'w') as fw:
        for line in tqdm(f):
            cache = line.strip().split('\t')
            indx = cache[0]
            label = cache[1]
            bert_feature = bc.encode([cache[2]])[0].tolist()
            bert_feature = ('\t').join(str(x) for x in bert_feature)
            fw.write(str(mapIndex[indx]) + "\t" + bert_feature + "\t" + label + "\n")

def entity_feature(entity_nodes):
    mapIndex = dict()
    with open(map_path + "mapindex.txt") as map:

        for line in tqdm(map):
            cache = line.strip().split('\t')
            mapIndex[cache[0]] = cache[1]

    map.close()

    bc = BertClient()

    with  open(outpath + '{}.content.entity'.format(dataset), 'w') as fw:
        for node in tqdm(entity_nodes):
            bert_feature = bc.encode([node])[0].tolist()
            bert_feature = ('\t').join(str(x) for x in bert_feature)
            fw.write(str(mapIndex[node]) + "\t" + bert_feature + "\t" + "entity\n")


def tokenize(sen):
    return WordPunctTokenizer().tokenize(sen)
    # return jieba.cut(sen)


def build_topic_feature_sklearn(datapath, DATASETS, TopicNum=20, stopwords=list(), train=False):
    # sklearn-lda

    idxlist = []
    corpus = []
    catelist = []
    with open('{}{}.txt'.format(datapath, DATASETS), 'r', encoding='utf8') as f:
        for line in f:
            ind, cate, content = line.strip().split('\t')
            idxlist.append(ind)
            corpus.append(content)
            catelist.append(cate)

    # with open(datapath + 'doc_index_LDA.pkl', 'wb') as f:
    #     pkl.dump(idxlist, f)

    print("text feature transforming...")
    corpus = preprocess_corpus_notDropEntity(corpus,stopwords=stopwords, involved_entity=set())

    vectorizer = CountVectorizer(min_df=10 if DATASETS != "example" else 0, stop_words=stopwords)
    X = vectorizer.fit_transform(corpus)
    # # vocabulary_ 的对照关系，读上面那个bow的模型就可以了


    if train:
        alpha, beta = 0.1, 0.1
        lda = LatentDirichletAllocation(n_components=TopicNum, max_iter=1200,
                                        learning_method='batch', n_jobs=-1,
                                        doc_topic_prior=alpha, topic_word_prior=beta,
                                        verbose=1,
                                        )
        lda_feature = lda.fit_transform(X)
        return lda_feature,lda,idxlist



init()


conn.close()