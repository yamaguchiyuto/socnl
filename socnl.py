import numpy as np
import scipy as sp
from scipy.sparse import diags

def solve(y,A,a,t=10):
    """ iter """
    labeled_nodes = y.shape[0]
    Au = A[labeled_nodes:,:]
    if a.sum() == 0:
        Du = diags(np.array(1.0/Au.sum(1).T)[0],0) # (Du + (a-1)I)^{-1}
        r = 0
    else:
        Du = diags(np.array(1.0/(Au.sum(1)+a.sum()).T)[0],0) # Du^{-1}
        r = Du.dot(np.outer(np.ones(A.shape[0]-labeled_nodes),a)) # Du^{-1}*1a^T
    f = np.zeros((Au.shape[0],y.shape[1])) / y.shape[1]
    Pu = Du.dot(Au) # (Du + (a-1)I)^{-1} * Au
    Puu = Pu[:,labeled_nodes:]
    Pul = Pu[:,:labeled_nodes]

    i = 0
    while True:
        f = Puu.dot(f) + Pul.dot(y) + r
        if i > t: break
        i += 1
    return f

def make_decision(f):
    return np.array(f.argmax(1))

def confidence(f):
    return np.array(f.max(1))

def prior(lamb,y):
    return (y.sum(0) / y.shape[0] ) * lamb

def fit(labels,graph,lamb):
    y = labels
    A = graph
    a = prior(lamb,y)
    f = solve(y,A,a)
    labels = make_decision(f)
    p = confidence(f)
    return zip(labels,p)

if __name__ == '__main__':
    """ test """
    import sys
    from scipy.sparse import lil_matrix

    def read_labels(filepath,n_of_labeled_nodes,n_of_labels):
        labels = lil_matrix((n_of_labeled_nodes,n_of_labels))
        for line in open(filepath,'r'):
            nid,label = map(int,line.rstrip().split('\t'))
            if label != -1:
                labels[nid,label] = 1
        return labels.tocsr()

    def read_graph(filepath, n_of_nodes):
        graph = lil_matrix((n_of_nodes, n_of_nodes))
        for line in open(filepath, 'r'):
            src,dsts = line.rstrip().split('\t')
            src = int(src)
            for dst in map(int, dsts.split(' ')):
                graph[src,dst] = 1
        return graph.tocsr()

    label_file = sys.argv[1]
    graph_file = sys.argv[2]
    n_of_nodes = int(sys.argv[3])
    n_of_labeled_nodes = int(sys.argv[4])
    n_of_labels = int(sys.argv[5])
    lamb = float(sys.argv[6])

    """ READ DATA """
    labels = read_labels(label_file,n_of_labeled_nodes,n_of_labels)
    graph = read_graph(graph_file, n_of_nodes)

    """ LP """
    inferred_labels = fit(labels,graph,lamb)

    for l,p in inferred_labels:
        print "%s\t%s" % (int(l),float(p))
