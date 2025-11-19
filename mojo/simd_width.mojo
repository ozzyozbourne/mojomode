from sys.info import simd_width_of

fn main():
    print("=== Checking SIMD Widths for Your Processor ===\n")
    
    # Check for different data types
    print("Float32 SIMD width:", simd_width_of[DType.float32]())
    print("Float64 SIMD width:", simd_width_of[DType.float64]())
    print("Int32 SIMD width:", simd_width_of[DType.int32]())
    print("Int64 SIMD width:", simd_width_of[DType.int64]())
    print("Int16 SIMD width:", simd_width_of[DType.int16]())
    print("Int8 SIMD width:", simd_width_of[DType.int8]())
