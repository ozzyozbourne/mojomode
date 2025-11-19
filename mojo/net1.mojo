from memory import alloc
from layout import Layout, LayoutTensor
from random import random_float64, seed
from time import monotonic

alias R, C = 5, 2
alias layout = Layout.row_major(R, C)

fn cost(read train: LayoutTensor[DType.float64, layout], read w: SIMD[DType.float64, 1], out cost: SIMD[DType.float64, 1]):
    cost = 0.0
    for i in range(R): cost += pow(w*train[i, 0][0] - train[i, 1][0], 2)

fn main():
    # seed(Int(monotonic()))
    var train = LayoutTensor[DType.float64, layout](alloc[Float64](R * C))
    for r in range(R): 
        train[r, 0][0] = r
        train[r, 1][0] = r * 2 
    print(train)
    w = 1
    eps = 1e-3
    print(cost(train, w), " ", cost(train, w - eps), " ", cost(train, w + eps))
    del1 = (cost(train, w + eps) - cost(train, w))/eps
    del2 = (cost(train, w - eps) - cost(train, w))/eps

    move1 = -2*del1
    move2 = -2*del1

    print(del1, "   ",  del2)
    print(move1, "   ", move2)



