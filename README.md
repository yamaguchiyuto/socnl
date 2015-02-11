# Demo

``
sh demo.sh
``

* Results in `toy.inferred` and `toy.result`

# Usage

``
python socnl.py [training filepath] [graph filepath] [# of nodes] [# of labeled nodes] [# of labels] [lambda]  > [inferred filepath]
``

``
python evaluation.py [test filepath] [inferred filepath] > [result filepath]
``

* socnl.py requires numpy/scipy

# File format

Make sure to start all node ids from 0.

## Graph file format

``
[src node id] \t [dst node id]
``

* See toy data file

## Training file format

``
[node id] \t [label id]
``

* Make sure to list labeled nodes on top, and unlabeled nodes on bottom.
* For unlabeled nodes, let [label id] = -1.
* See toy data file

## Test file format

``
[node id] \t [label id]
``

* Make sure to list only test nodes.
* See toy data file

## Inferred file format

``
[inferred label id] \t [confidence value]
``

## Result file format

``
[precision@p] [p]
``
