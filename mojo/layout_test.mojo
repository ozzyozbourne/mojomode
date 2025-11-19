from layout import IntTuple, Layout, print_layout

alias l2x4row_major = Layout.row_major(2, 4)
alias l2x4col_major = Layout.col_major(2, 4)
alias coords: IntTuple = [1, 1]

fn row_and_column_major():
    print("row and colum major layouts\n")
    print_layout(l2x4row_major)
    print()
    print_layout(l2x4col_major)
    print()
    

fn coords_to_index():
    print("\ncoordinates to index\n")
    var idx = l2x4row_major(coords)
    print("row major -> idex at:", coords, " -> ", idx, "\n")

    idx = l2x4col_major(coords)
    print("col major -> idex at:", coords, " -> ", idx, "\n")

fn index_to_coords():
    print("\nindex to coordinates\n")
    var coord = l2x4row_major.idx2crd(5)
    print("row major -> coords at 5: ", coord, "\n")

    coord = l2x4col_major.idx2crd(3)
    print("col major -> coords at 3: ", coord, "\n")

fn main():
    row_and_column_major()
    coords_to_index()
    index_to_coords()
