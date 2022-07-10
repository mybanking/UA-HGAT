from itertools import groupby

filenames = ['train']
ans = []
path = './model/data/twitter'

nums = 10

idx_map = dict()
with open(path +'/mapindex.txt', 'r',encoding='utf-8') as f:
    for line in f:
        line = line.strip().split('\t')
        idx_map[line[1]] = line[0]

f.close()
textmap =  dict()
with open(path +'/twitter.txt', 'r',encoding="utf-8") as f:
    for line in f:
        line =line.strip().split('\t')
        textmap[line[0]] = line[1]

f.close()
cache = []
lists = []
with open(path +'/train_inductive.map', 'r') as f:
    for line in f:
        line = line.strip().split('\t')
        x = idx_map.get(line[0])
        a = dict()
        a["label"] = textmap[x]
        a["index"] = x
        a["tran"] = line[0]
        lists.append(a)

user_sort = sorted(lists, key=lambda x: (x["label"]))
user_group = groupby(user_sort, key=lambda x: (x["label"]))

re =[]
for key, group in user_group:
    # print(key, list(group)[0:50])
    for a in  list(group)[0:nums]:
        re.append(a["tran"])
f.close()

with open(path +'/'+str(nums)+'train_inductive.map', 'w') as f:
    for text in re:
        f.write(text+"\n")
f.close()
