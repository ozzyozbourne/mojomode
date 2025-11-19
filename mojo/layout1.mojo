from layout import IntTuple, Layout, print_layout

fn row_and_column_major_1():
    print("row and colum major")
    var l2x4row_major = Layout.row_major(2, 4)
    print_layout(l2x4row_major)
    print()
    var l6x6col_major = Layout.col_major(6, 6)
    print_layout(l6x6col_major)
    print()

fn row_and_column_major_2():
    print("row and colum major")
    var l4x1row_major = Layout.row_major(4, 1)
    print_layout(l4x1row_major)
    print()
    var l5x6col_major = Layout.col_major(5, 6)
    print_layout(l5x6col_major)
    print()

fn row_and_column_major_3():
    print("row and colum major")
    var l10x4row_major = Layout.row_major(10, 1)
    print_layout(l10x4row_major)
    print()
    var l3x6col_major = Layout.col_major(3, 6)
    print_layout(l3x6col_major)
    print()

fn coords_to_index_1():
    print("coordinates to index")
    var l3x4row_major = Layout.row_major(3, 4)
    print_layout(l3x4row_major)

    var coords: IntTuple = [1, 1]
    var idx = l3x4row_major(coords)
    print("index at (1, 1): ", idx)
    print("coordinates at index 7:", l3x4row_major.idx2crd(7))
    print()

fn coords_to_index_2():
    print("coordinates to index")
    var l3x6col_major = Layout.col_major(3, 6)
    print_layout(l3x6col_major)

    var coords: IntTuple = [1, 1]
    var idx = l3x6col_major(coords)
    print("index at (1, 1): ", idx)
    print("coordinates at index 7:", l3x6col_major.idx2crd(7))
    print()

fn coords_to_index_same():
    var l3x6col_major = Layout.col_major(3, 6)
    var l3x6row_major = Layout.row_major(3, 6)
    print("coordinates to index col major")
    print_layout(l3x6col_major)
    print()
    print("coordinates to index row major")
    print_layout(l3x6row_major)
    print()

    var coords: IntTuple = [1, 1]
    var idx = l3x6col_major(coords)
    print("col major index at (1, 1): ", idx)
    print("col major coordinates at index 7:", l3x6col_major.idx2crd(7))

    print()

    print("row major index at (1, 1): ", l3x6row_major(coords))
    print("row major coordinates at index 7:", l3x6row_major.idx2crd(7))
    print()



fn main():
    # row_and_column_major_1()
    # row_and_column_major_2()
    # row_and_column_major_3()

    # coords_to_index_1()
    # coords_to_index_2()
    coords_to_index_same()

