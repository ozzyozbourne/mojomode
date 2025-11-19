from memory import alloc
from time import perf_counter_ns
from layout import Layout, LayoutTensor
from collections import InlineArray
from math import fma
from sys import simd_width_of

alias M = 2048
alias K = 2048
alias N = 2048

alias layout_A = Layout.row_major(M, K)
alias layout_B = Layout.col_major(K, N)
alias layout_C = Layout.row_major(M, N)

alias simd_width = simd_width_of[DType.float32]()

fn matmul_fma(
    A: LayoutTensor[DType.float32, layout_A],
    B: LayoutTensor[DType.float32, layout_B],
    mut C: LayoutTensor[DType.float32, layout_C],
):
    for i in range(M):
        for j in range(N):
            var sum_vec = SIMD[DType.float32, simd_width](0.0)
            
            # Vectorized inner loop - process simd_width elements at a time
            for k in range(0, K - simd_width + 1, simd_width):
                # Load SIMD vectors from A and B
                a_vec = A.load[simd_width](i, k)
                b_vec = B.load[simd_width](k, j)
                
                # FMA: sum_vec = (a_vec * b_vec) + sum_vec
                sum_vec = fma(a_vec, b_vec, sum_vec)
            
            # Horizontal sum: add all elements in the vector
            var sum: Float32 = 0.0
            for s in range(simd_width):
                sum += sum_vec[s]
            
            # Handle remaining elements (if K not divisible by simd_width)
            for k in range((K // simd_width) * simd_width, K):
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
            A[i, j] = 1.0
    
    for i in range(K):
        for j in range(N):
            B[i, j] = 1.0
    
    for i in range(M):
        for j in range(N):
            C[i, j] = 0.0
    
    
    start = perf_counter_ns()
    matmul_fma(A, B, C)
    end = perf_counter_ns()
    
    elapsed_ms = (end - start) / 1_000_000
    print("Time: ", elapsed_ms, " ms")

    print("Sample result C[0,0] =", C[0, 0][0], "(expected:", Float32(K), ")")

