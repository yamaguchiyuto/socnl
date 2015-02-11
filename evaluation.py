import sys
import numpy as np

def read_answer(filepath):
    answer = []
    for line in open(filepath):
        nid,label = line.rstrip().split('\t')
        answer.append([int(label)])
    return np.array(answer)

def read_inferred(filepath):
    inferred = []
    confs = []
    for line in open(filepath):
        label,conf = line.rstrip().split('\t')
        inferred.append([int(label)])
        confs.append([float(conf)])
    return (np.array(inferred),np.array(confs))


answer_file = sys.argv[1]
inferred_file = sys.argv[2]

answer = read_answer(answer_file)
inferred,conf = read_inferred(inferred_file)
v = np.hstack([answer,inferred,conf]).tolist()
sorted_v = np.array(sorted(v, key=lambda x:x[2],reverse=True))

for i in range(1,21):
    r = i/20.0 # recall
    n = int(len(answer) * r) # number of not rejected
    if n == 0: n += 1
    correct = sum(sorted_v[:n,0]==sorted_v[:n,1])

    p = correct / float(n)
    print p,r
