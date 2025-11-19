from math import fma
from layout import LayoutTensor, Layout
from collections import InlineArray

fn main():
    print("=== Understanding Element Width ===\n")
    demo_basic_width()
    demonstrate_load_vs_indexing()
    demonstrate_performance_difference()
    print("=== Summary ===")
    print("1. tensor[i,j] uses element width 1 by default (scalar)")
    print("2. load[N]() lets you specify any width N (SIMD)")
    print("3. Larger N = fewer iterations = faster performance!")

fn demo_basic_width():
    alias rows = 4
    alias cols = 8
    alias size = rows * cols
    alias layout = Layout.row_major(rows, cols)
    
    var storage = InlineArray[Float32, size](uninitialized=True)
    var tensor = LayoutTensor[DType.float32, layout](storage)
    
    print("--- Regular Tensor (element width = 1) ---")
    
    tensor[0, 0] = Float32(5.0)
    
    var value = tensor[0, 0]  
    print("Value:", value[0])
    print()
    
    print("--- Using load() to get different widths ---")
    _ = tensor.fill(1.0)  # Fill with 1.0
    
    var vec1 = tensor.load[1](0, 0)
    print("load[1] returns width 1, value:", vec1[0])
    
    var vec4 = tensor.load[4](0, 0)
    print("load[4] returns width 4")
    print("  values:", vec4[0], vec4[1], vec4[2], vec4[3])
    
    var vec8 = tensor.load[8](0, 0)
    print("load[8] returns width 8")
    print("  first 4:", vec8[0], vec8[1], vec8[2], vec8[3])
    print()


fn demonstrate_load_vs_indexing():
    print("=== tensor[i,j] vs load[N]() ===\n")
    
    alias rows = 4
    alias cols = 16
    alias size = rows * cols
    alias layout = Layout.row_major(rows, cols)
    
    var storage = InlineArray[Float32, size](uninitialized=True)
    var tensor = LayoutTensor[DType.float32, layout](storage)
    
    # Initialize with sequential values
    for i in range(rows):
        for j in range(cols):
            tensor[i, j] = Float32(i * cols + j)
    
    print("Method 1: Using tensor[i, j] (width 1)")
    var elem = tensor[0, 0]
    print("  tensor[0, 0] =", elem[0])
    print()
    
    print("Method 2: Using load[4]() (width 4)")
    var vec4 = tensor.load[4](0, 0)
    print("  load[4](0, 0) = [", vec4[0], vec4[1], vec4[2], vec4[3], "]")
    print()
    
    print("Method 3: Using load[8]() (width 8)")
    var vec8 = tensor.load[8](0, 0)
    print("  load[8](0, 0) = [", vec8[0], vec8[1], vec8[2], vec8[3], 
          vec8[4], vec8[5], vec8[6], vec8[7], "]")
    print()


fn demonstrate_performance_difference():
    print("=== Performance: Scalar vs SIMD ===\n")
    
    alias rows = 4
    alias cols = 16
    alias size = rows * cols
    alias layout = Layout.row_major(rows, cols)
    
    var storage_a = InlineArray[Float32, size](uninitialized=True)
    var storage_b = InlineArray[Float32, size](uninitialized=True)
    var storage_result = InlineArray[Float32, size](uninitialized=True)
    
    var tensor_a = LayoutTensor[DType.float32, layout](storage_a)
    var tensor_b = LayoutTensor[DType.float32, layout](storage_b)
    var tensor_result = LayoutTensor[DType.float32, layout](storage_result)
    
    # Initialize
    for i in range(rows):
        for j in range(cols):
            tensor_a[i, j] = Float32(j)
            tensor_b[i, j] = Float32(2.0)
    
    print("Approach 1: Scalar (width 1) - Process 1 value per iteration")
    print("  Loop iterations needed:", rows * cols)
    
    # Scalar approach
    for i in range(rows):
        for j in range(cols):
            var a = tensor_a[i, j][0]
            var b = tensor_b[i, j][0]
            var result = a * b
            tensor_result[i, j] = result
    
    print("  First row result:", tensor_result[0, 0][0], tensor_result[0, 1][0], 
          tensor_result[0, 2][0], "...")
    print()
    
    print("Approach 2: SIMD (width 4) - Process 4 values per iteration")
    alias simd_width = 4
    print("  Loop iterations needed:", rows * (cols // simd_width))
    
    # SIMD approach
    for i in range(rows):
        for j in range(0, cols, simd_width):
            var a_vec = tensor_a.load[simd_width](i, j)
            var b_vec = tensor_b.load[simd_width](i, j)
            var result_vec = a_vec * b_vec
            tensor_result.store(i, j, result_vec)
    
    print("  First row result:", tensor_result[0, 0][0], tensor_result[0, 1][0], 
          tensor_result[0, 2][0], "...")
    print()
    print("SIMD does", rows * cols // (rows * (cols // simd_width)), 
          "times fewer iterations!")
    print()
