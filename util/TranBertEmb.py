from bert_serving.client import BertClient
bc = BertClient()
from tqdm import tqdm
import numpy as np

dataset = 'example'
path = './data/' + dataset + '/'
map_path = './model/data/' + dataset + '/'

mapIndex = dict()
with open(map_path+"mapindex.txt") as map:

    for line in tqdm(map):
        cache = line.strip().split('\t')
        mapIndex[cache[0]] = cache[1]

map.close()

with open(path+dataset+".txt") as f ,open(path+"bert"+dataset+".txt", 'w') as fw:
    for line in tqdm(f):
        cache = line.strip().split('\t')
        indx = cache[0]
        label = cache[1]
        text = cache[2]
        bert_feature = bc.encode([cache[2]])[0].tolist()
        bert_feature= ('\t').join(str(x) for x in bert_feature)


        fw.write(str(mapIndex[indx])+"\t"+bert_feature+"\t"+label+"\n")
f.close()
fw.close()

#
# indexes =[]
# features =[]
# cache =[]
# labels =[]
# with open(path+dataset+".txt") as f ,open(path+"bert"+dataset+".txt") as f:
#     for line in tqdm(f):
#         cache = line.strip().split('\t')
#         indexes.append(np.array(cache[0], dtype=int))
#         features.append(np.array(cache[1:-1], dtype=np.float32))
#         labels.append(np.array([cache[-1]], dtype=str))


