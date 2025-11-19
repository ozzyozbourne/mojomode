from memory import alloc
from layout import Layout, LayoutTensor, print_layout
from memory import UnsafePointer
from collections import InlineArray
from time import perf_counter_ns

alias M = 2048
alias K = 2048
alias N = 2048

alias layout_A = Layout.row_major(M, K)
alias layout_B = Layout.col_major(K, N)
alias layout_C = Layout.row_major(M, N)


fn matmul_good(
    A: LayoutTensor[DType.float32, layout_A],
    B: LayoutTensor[DType.float32, layout_B],
    mut C: LayoutTensor[DType.float32, layout_C],
):
    for i in range(M):
        for j in range(N):
            var sum: Float32 = 0.0
            for k in range(K):
                # A[i,k] in row-major: offset = i*K + k (stride 1 between elements!)
                # B[k,j] in col-major: offset = k + j*K (stride 1 between elements!)
                sum += A[i, k][0] * B[k, j][0]
            C[i, j] = sum


fn main():
    # Allocate storage
    var storage_A = alloc[Float32](M * K)
    var storage_B = alloc[Float32](K * N)
    var storage_C = alloc[Float32](M * N)
    
    # Create tensors
    var A = LayoutTensor[DType.float32, layout_A](storage_A)
    var B = LayoutTensor[DType.float32, layout_B](storage_B)
    var C = LayoutTensor[DType.float32, layout_C](storage_C)
    
    # Initialize with simple values
    for i in range(M):
        for j in range(K):
            A[i, j] = 1.0
    
    for i in range(K):
        for j in range(N):
            B[i, j] = 1.0
    
    for i in range(M):
        for j in range(N):
            C[i, j] = 0.0
    
    start = perf_counter_ns()
    matmul_good(A, B, C)
    end = perf_counter_ns()
    
    var elapsed_ms = (end - start) / 1_000_000
    
    print("Time: ", elapsed_ms, " ms")

    storage_A.free()
    storage_B.free()
    storage_C.free()
