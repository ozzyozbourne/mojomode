from math import fma
from layout import LayoutTensor, Layout
from collections import InlineArray
from sys.info import simd_width_of

fn main():
    print("=== LayoutTensor with FMA Examples ===\n")
    simple_fma_example()
    simd_fma_example()
    vectorized_tensor_example()
    tiled_fma_example()


fn simple_fma_example():
    print("--- Example 1: Simple Element-wise FMA ---")
    
    alias rows = 4
    alias cols = 4
    alias size = rows * cols
    alias layout = Layout.row_major(rows, cols)
    
    # Create storage for tensors
    var storage_a = InlineArray[Float32, size](uninitialized=True)
    var storage_b = InlineArray[Float32, size](uninitialized=True)
    var storage_c = InlineArray[Float32, size](uninitialized=True)
    var storage_result = InlineArray[Float32, size](uninitialized=True)
    
    # Create tensors
    var tensor_a = LayoutTensor[DType.float32, layout](storage_a)
    var tensor_b = LayoutTensor[DType.float32, layout](storage_b)
    var tensor_c = LayoutTensor[DType.float32, layout](storage_c)
    var tensor_result = LayoutTensor[DType.float32, layout](storage_result)
    
    # Initialize tensors
    for i in range(rows):
        for j in range(cols):
            tensor_a[i, j] = Float32(i + j)
            tensor_b[i, j] = Float32(2.0)
            tensor_c[i, j] = Float32(1.0)
    
    # Perform FMA: result = (a * b) + c
    for i in range(rows):
        for j in range(cols):
            var a_val = tensor_a[i, j][0]
            var b_val = tensor_b[i, j][0]
            var c_val = tensor_c[i, j][0]
            var result = fma(a_val, b_val, c_val)
            tensor_result[i, j] = result
    
    # Print results
    print("Matrix A:")
    print_tensor(tensor_a, rows, cols)
    print("\nFMA Result (A * 2.0 + 1.0):")
    print_tensor(tensor_result, rows, cols)
    print()


fn simd_fma_example():
    print("--- Example 2: SIMD Vectorized FMA ---")
    
    alias rows = 4
    alias cols = 16
    alias size = rows * cols
    alias layout = Layout.row_major(rows, cols)
    alias simd_width = 4  # Process 4 elements at once
    
    # Create storage
    var storage_a = InlineArray[Float32, size](uninitialized=True)
    var storage_b = InlineArray[Float32, size](uninitialized=True)
    var storage_c = InlineArray[Float32, size](uninitialized=True)
    var storage_result = InlineArray[Float32, size](uninitialized=True)
    
    # Create tensors
    var tensor_a = LayoutTensor[DType.float32, layout](storage_a)
    var tensor_b = LayoutTensor[DType.float32, layout](storage_b)
    var tensor_c = LayoutTensor[DType.float32, layout](storage_c)
    var tensor_result = LayoutTensor[DType.float32, layout](storage_result)
    
    # Initialize
    for i in range(rows):
        for j in range(cols):
            tensor_a[i, j] = Float32(j)
            tensor_b[i, j] = Float32(3.0)
            tensor_c[i, j] = Float32(5.0)
    
    # Perform SIMD FMA using load/store
    for i in range(rows):
        for j in range(0, cols, simd_width):
            # Load SIMD vectors
            var a_vec = tensor_a.load[simd_width](i, j)
            var b_vec = tensor_b.load[simd_width](i, j)
            var c_vec = tensor_c.load[simd_width](i, j)
            
            # FMA on SIMD vectors: (a * b) + c
            var result_vec = fma(a_vec, b_vec, c_vec)
            
            # Store result
            tensor_result.store(i, j, result_vec)
    
    # Print sample results
    print("First row of A:", end="")
    for j in range(cols):
        print(" " + String(tensor_a[0, j][0]), end="")
    print()
    
    print("First row of Result (A * 3.0 + 5.0):", end="")
    for j in range(cols):
        print(" " + String(tensor_result[0, j][0]), end="")
    print("\n")


fn vectorized_tensor_example():
    print("--- Example 3: Vectorized Tensor FMA ---")
    
    alias rows = 8
    alias cols = 16
    alias size = rows * cols
    alias layout = Layout.row_major(rows, cols)
    alias simd_width = 4
    
    # Create storage
    var storage_a = InlineArray[Float32, size](uninitialized=True)
    var storage_b = InlineArray[Float32, size](uninitialized=True)
    var storage_c = InlineArray[Float32, size](uninitialized=True)
    var storage_result = InlineArray[Float32, size](uninitialized=True)
    
    # Create base tensors
    var tensor_a = LayoutTensor[DType.float32, layout](storage_a)
    var tensor_b = LayoutTensor[DType.float32, layout](storage_b)
    var tensor_c = LayoutTensor[DType.float32, layout](storage_c)
    var tensor_result = LayoutTensor[DType.float32, layout](storage_result)
    
    # Initialize
    for i in range(rows):
        for j in range(cols):
            tensor_a[i, j] = Float32(i * cols + j)
            tensor_b[i, j] = Float32(0.5)
            tensor_c[i, j] = Float32(10.0)
    
    # Perform FMA using load/store with explicit SIMD width
    for i in range(rows):
        for j in range(0, cols, simd_width):
            # Load values as SIMD vectors
            var a_vec = tensor_a.load[simd_width](i, j)
            var b_vec = tensor_b.load[simd_width](i, j)
            var c_vec = tensor_c.load[simd_width](i, j)
            
            # Perform FMA: (a * b) + c
            var result_vec = fma(a_vec, b_vec, c_vec)
            
            # Store result
            tensor_result.store(i, j, result_vec)
    
    # Print sample results
    print("Sample values from Result (A * 0.5 + 10.0):")
    for i in range(3):
        print("  Row", i, ":", String(tensor_result[i, 0][0]), 
              String(tensor_result[i, 1][0]), 
              String(tensor_result[i, 2][0]), "...")
    print()


fn tiled_fma_example():
    print("--- Example 4: Tiled FMA Operation ---")
    
    alias rows = 16
    alias cols = 16
    alias size = rows * cols
    alias layout = Layout.row_major(rows, cols)
    alias tile_size = 4
    
    # Create storage
    var storage_a = InlineArray[Float32, size](uninitialized=True)
    var storage_b = InlineArray[Float32, size](uninitialized=True)
    var storage_c = InlineArray[Float32, size](uninitialized=True)
    var storage_result = InlineArray[Float32, size](uninitialized=True)
    
    # Create tensors
    var tensor_a = LayoutTensor[DType.float32, layout](storage_a)
    var tensor_b = LayoutTensor[DType.float32, layout](storage_b).fill(2.0)
    var tensor_c = LayoutTensor[DType.float32, layout](storage_c).fill(3.0)
    var tensor_result = LayoutTensor[DType.float32, layout](storage_result)
    
    # Initialize tensor_a with a pattern
    for i in range(rows):
        for j in range(cols):
            tensor_a[i, j] = Float32(i + j)
    
    # Process in tiles
    alias num_row_tiles = rows // tile_size
    alias num_col_tiles = cols // tile_size
    
    for tile_row in range(num_row_tiles):
        for tile_col in range(num_col_tiles):
            # Extract tiles
            var tile_a = tensor_a.tile[tile_size, tile_size](tile_row, tile_col)
            var tile_b = tensor_b.tile[tile_size, tile_size](tile_row, tile_col)
            var tile_c = tensor_c.tile[tile_size, tile_size](tile_row, tile_col)
            var tile_result = tensor_result.tile[tile_size, tile_size](tile_row, tile_col)
            
            # Process tile with FMA
            for i in range(tile_size):
                for j in range(tile_size):
                    var a_val = tile_a[i, j][0]
                    var b_val = tile_b[i, j][0]
                    var c_val = tile_c[i, j][0]
                    tile_result[i, j] = fma(a_val, b_val, c_val)
    
    # Print corner of result
    print("Top-left corner of Result (A * 2.0 + 3.0):")
    for i in range(4):
        print("  ", end="")
        for j in range(4):
            print(String(tensor_result[i, j][0]), end=" ")
        print()
    print()


fn print_tensor(tensor: LayoutTensor, rows: Int, cols: Int):
    for i in range(rows):
        print("  ", end="")
        for j in range(cols):
            print(String(tensor[i, j][0]), end=" ")
        print()
