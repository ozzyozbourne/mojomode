from memory import alloc
from layout import Layout, LayoutTensor, print_layout
from collections import InlineArray
from time import perf_counter_ns

alias M = 2
alias K = 2
alias N = 2

alias layout_A = Layout.col_major(M, K)
alias layout_B = Layout.row_major(K, N)
alias layout_C = Layout.row_major(M, N)

fn matmul_bad(
    A: LayoutTensor[DType.float32, layout_A],
    B: LayoutTensor[DType.float32, layout_B],
    mut C: LayoutTensor[DType.float32, layout_C],
):
    for i in range(M):
        for j in range(N):
            var sum: Float32 = 0.0
            for k in range(K):
                # A[i,k] in col-major: offset = i + k*M (stride M between elements!)
                # B[k,j] in row-major: offset = k*N + j (stride N between elements!)
                sum += A[i, k][0] * B[k, j][0]
            C[i, j] = sum

fn main():
    
    var storage_A = alloc[Float32](M * K)
    var storage_B = alloc[Float32](K * N)
    var storage_C = alloc[Float32](M * N)
    
    var A = LayoutTensor[DType.float32, layout_A](storage_A)
    var B = LayoutTensor[DType.float32, layout_B](storage_B)
    var C = LayoutTensor[DType.float32, layout_C](storage_C)
    
    for i in range(M):
        for j in range(K):
            A[i, j][0] = 2*j

    print_layout(layout_A)
    
    for i in range(K):
        for j in range(N):
            B[i, j][0] = 3*j

    print_layout(layout_B)

    for i in range(M):
        for j in range(N):
            C[i, j] = 0.0

    print_layout(layout_C)
    
    matmul_bad(A, B, C)

    print(A)
    print()

    print(B)
    print()

    print(C)
    
