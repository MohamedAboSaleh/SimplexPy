import sys
import tkinter as tk
from tkinter import filedialog


class OptList(list):

    def __init__(self, *args):
        list.__init__(self, *args)

    def __add__(self, other):
        # return a new instance of OptList which represents the addition of the elements of the two lists
        return OptList(list(map(sum, zip(self, other))))

    def __sub__(self, other):
        # return a new instance of OptList which represents the subtraction of the elements of the two lists
        return OptList(list(map(lambda x: x[0] - x[1], zip(self, other))))

    def __mul__(self, other):
        # return the dot product of the elements of the two lists
        return sum(OptList(list(map(lambda x: x[0] * x[1], zip(self, other)))))


N = 128
M = 64

A = OptList([OptList([0 for _ in range(N)]) for _ in range(N)]) # double A[N][N]
B = OptList([OptList([0 for _ in range(N)]) for _ in range(N)]) # double B[N][N]
C = OptList(OptList([0 for _ in range(N)])) # double C[N]
D = OptList([OptList([0 for _ in range(N)]) for _ in range(N)]) # double D[N][N]
c = OptList(OptList([0 for _ in range(N)])) # double c[N]
b = OptList(OptList([0 for _ in range(N)])) # double b[N]
b_aux = OptList(OptList([0 for _ in range(N)])) # double b_aux[N]
cb = OptList(OptList([0 for _ in range(N)])) # double cb[N]
cbBI = OptList(OptList([0 for _ in range(N)])) # double cbBI[N]
cbBID = OptList(OptList([0 for _ in range(N)])) # double cbBID[M]
cd = OptList(OptList([0 for _ in range(N)])) # double cd[N]
rd = OptList(OptList([0 for _ in range(N)])) # double rd[N]
BID = OptList([OptList([0 for _ in range(N)]) for _ in range(N)]) # double BID[N][N]
W = OptList([OptList([0 for _ in range(N)]) for _ in range(N)]) # double W[N][N]
BI = OptList([OptList([0 for _ in range(N)]) for _ in range(N)]) # double BI[N][N]
BIA_aux = OptList([OptList([0 for _ in range(N)]) for _ in range(N)]) # double BIA_aux[N][N]

A_aux = OptList([OptList([0 for _ in range(N)]) for _ in range(N)]) # double A_aux[N][N]
BIb = OptList(OptList([0 for _ in range(N)])) # double BIb[N]
epsilon = -1 # double epsilon
d = OptList(OptList([0 for _ in range(N)])) # int d
d_aux = OptList(OptList([0 for _ in range(N)])) # int d_aux[N]
basis = OptList(OptList([0 for _ in range(N)])) # int basis[N]

n = -1 # int n
m = -1 # int m
fp = None # FILE *fp

Initial_n = -1 # int Initial_n
Initial_W = OptList([OptList([0 for _ in range(N)]) for _ in range(N)]) # double Initial_W[N][N]
Initial_cb = OptList(OptList([0 for _ in range(N)])) # double Initial_cb[N]
Initial_cd = OptList(OptList([0 for _ in range(N)])) # double Initial_cd[N]
Initial_A = OptList([OptList([0 for _ in range(N)]) for _ in range(N)]) # double Initial_A[N][N]
Initial_A_aux = OptList([OptList([0 for _ in range(N)]) for _ in range(N)]) # double Initial_A_aux[N][N]

Initial_c = OptList(OptList([0 for _ in range(N)])) # double Initial_c[N]
Initial_c_aux = OptList(OptList([0 for _ in range(N)])) # double Initial_c_aux[N]
Initial_basis = OptList(OptList([0 for _ in range(N)])) # int Initial_basis[N]
Initial_D = OptList([OptList([0 for _ in range(N)]) for _ in range(N)]) # double Initial_D[N][N]
Initial_d = OptList(OptList([0 for _ in range(N)])) # int Initial_d[N]
Initial_B = OptList([OptList([0 for _ in range(N)]) for _ in range(N)]) # double Initial_B[N][N]
Initial_BIb = OptList(OptList([0 for _ in range(N)])) # double Initial_BIb[N]

Initial_C = OptList([OptList([0 for _ in range(N)]) for _ in range(N)]) # double Initial_C[N][N]
Initial_b = OptList(OptList([0 for _ in range(N)])) # double Initial_b[N]
Initial_b_aux = OptList(OptList([0 for _ in range(N)])) # double Initial_b_aux[N]
Initial_rd = OptList(OptList([0 for _ in range(N)])) # double Initial_rd[N]
Initial_BID = OptList([OptList([0 for _ in range(N)]) for _ in range(N)]) # double Initial_BID[N][N]
Initial_BI = OptList([OptList([0 for _ in range(N)]) for _ in range(N)]) # double Initial_BI[N][N]
Initial_cbBI = OptList(OptList([0 for _ in range(N)])) # double Initial_cbBI[N]
Initial_cbBID = OptList(OptList([0 for _ in range(N)])) # double Initial_cbBID[N]
Initial_BIA_aux = OptList([OptList([0 for _ in range(N)]) for _ in range(N)]) # double Initial_BIA_aux[N][N]


# bublesort.c
def bublesort(array, n):
    flag = 1
    for i in range(n):
        if flag == 0:
            break
        flag=0
        limit = n - i - 1
        for j in range(limit):
            if array[j] > array[j + 1]:
                flag = 1
                temp = array[j]
                array[j] = array[j + 1]
                array[j + 1] = temp


# bublesort_d.c
def bublesort_d(array, darr, n):
    flag = 1
    for i in range(n):
        if flag == 0:
            break
        flag=0
        limit = n - i - 1
        for j in range(limit):
            if array[j] > array[j + 1]:
                flag = 1
                dtemp = darr[j]
                darr[j] = darr[j + 1]
                darr[j + 1] = dtemp
                temp = array[j]
                array[j] = array[j + 1]
                array[j + 1] = temp


# compute_cb_cd.c
def compute_cb_cd():
    for i in range(m):
        cb[i] = c[d[i]]
    for i in range(m, n):
        cd[i - m] = c[d[i]]
    print("d:")
    for i in range(n):
        print(f' {d[i]} ', end='')
    print()

    print("cb:")
    for i in range(m):
        print(" %lf "%cb[i], end='')
    print()

    print("cd:")
    for i in range(n):
        print(" %lf "%cd[i], end='')
    print()


# compute_Initial_cb_Initial_cd.c
def compute_Initial_cb_Initial_cd():
    for i in range(m):
        Initial_cb[i] = Initial_c[Initial_d[i]]

    for i in range (m,Initial_n):
        Initial_cd[i - m] = Initial_c[Initial_d[i]]
    print("Initial_d:")
    for i in range(Initial_n):
        print(f' {Initial_d[i]} ', end='')
    print()

    print("Initial_cb:")
    for i in range(m):
        print(" %lf "%Initial_cb[i], end='')
    print()
    print("Initial_cd:")
    for i in range(Initial_n - m):
        print(" %lf "%Initial_cd[i], end='')
    print()


# copy_matrix.c
def copy_matrix(Dest, Source, n, m):
    for i in range(m):
        for j in range(n):
            Dest[i][j] = Source[i][j]


# copy_submatrix.c
def copy_submatrix(Dest, Source, istart, depth, jstart, length):
    for i in range(istart, depth):
        for j in range(jstart, jstart + length):
            Dest[i - istart][j - jstart] = Source[i][j]


# copy_to_initial_matrix.c
def copy_to_initial_matrix():
    for i in range(m):
        for j in range(n):
            Initial_A[i][j] = Initial_A_aux[i][j] = A[i][j]

    for i in range(m):
        for j in range(n, n + m):
            if i == j - n:
                Initial_A[i][j] = Initial_A_aux[i][j] = 1.0
            else:
                Initial_A[i][j] = Initial_A_aux[i][j] = 0.0


# copy_vector.c
def copy_vector(Dest, Source, n):
    for i in range(n):
        Dest[i] = Source[i]


# erase_epsilons_matrix.c
def erase_epsilons_matrix(dmat, m, n):
    for i in range(m):
        for j in range(n):
            if abs(dmat[i][j]) < epsilon:
                dmat[i][j] = 0.0


# erase_epsilons_vector.c
def erase_epsilons_vector(darray, n):
    for i in range(n):
        if abs(darray[i]) < epsilon:
            darray[i] = 0.0


# find_all_negative_rds.c
def find_all_negative_rds(neg_ids, no_of_rds):
    """
    Assumptions: d[i] is original index, rd orderred by i=0,.., n-1
                                                        d[0] ... d[n-1]
    """
    index = 0
    for i in range(n - m):
        if rd[i] < 0:
            neg_ids[index] = d[m + i]
            index += 1
            print("\nXX:i = %d, rd[%d] = %lf, d[%d] = %d, index = %d"%(i, i, rd[i], i, d[i], index))
    no_of_rds[0] = index


# find_exiting_id.c
def find_exiting_id(y, x, enter_id, n, m, max_new_value):
    """
    Assuming y = B-1D, x = B-1b
    """
    temp_min_index, q, temp_min = 0, 0, 0.0

    for i in range(n):
        if d[i] == enter_id:
            q = i
    init_flag = 0
    # CHANGE
    unbounded_flag = 1
    # END OF CHANGE

    for i in range(m):
        print("y[%d][%d] = %lf, x[%d] = %lf %lf"%(i,q, y[i][q],i, x[i],0))
        print(f"init_flag = {init_flag}")

        if y[i][q] > 0.0:
            # CHANGE
            unbounded_flag = 0
            # END OF CHANGE
            temp = x[i] / y[i][q]
            print("i = %d, temp = %lf"%(i, temp))
            if init_flag == 0:
                temp_min = temp
                temp_min_index = i
                init_flag = 1
            elif temp < temp_min:
                temp_min = temp
                temp_min_index = i
        print("temp_min_index  = %d, temp_min  = %lf"%(temp_min_index,temp_min))
    print(f"unbounded flag = {unbounded_flag}")
    if unbounded_flag == 1:
        print("Unbounded linear program!",file=sys.stderr)
        sys.exit(0)

    max_new_value[0] = temp_min
    return temp_min_index


# find_Initial_exiting_id.c
def find_Initial_exiting_id(y, x, enter_id, n, m):
    temp_min_index, q, temp_min = 0, 0, 0.0
    for i in range(Initial_n):
        if Initial_d[i] == enter_id:
            q = i
    init_flag = 0
    for i in range(m):
        print("y[%d][%d] = %lf, x[%d] = %lf %lf"%(i,q, y[i][q],i, x[i],0))
        print(f"init_flag = {init_flag}")
        if y[i][q] > 0.0:
            temp = x[i] / y[i][q]
            print("i = %d, temp = %lf"%( i, temp))
            if init_flag == 0:
                temp_min = temp
                temp_min_index = i
                init_flag = 1
            elif temp < temp_min:
                 temp_min = temp
                 temp_min_index = i
        print("temp_min_index  = %d, temp_min  = %lf"%(temp_min_index,temp_min))
    return temp_min_index


# find_Initial_most_negative.c
def find_Initial_most_negative():
    most_index = Initial_d[m]
    temp_value = Initial_rd[0]
    for i in range(Initial_n - m):
        if Initial_rd[i] < temp_value:
            most_index = Initial_d[m + i]
            temp_value = Initial_rd[i]
    return most_index


# find_min_value.c
def find_min_value(rd, n):
    most_index = d[0]
    temp_value = rd[0]
    for i in range(n):
        if rd[i] < temp_value:
            most_index = d[i]
            temp_value = rd[i]
    return temp_value


# find_most_negative.c
def find_most_negative():
    most_index = d[m]
    temp_value = rd[0]

    for i in range(n - m):
        if rd[i] < temp_value:
            most_index = d[m + i]
            temp_value = rd[i]
    return most_index


# Initial_set_d.c
def Initial_set_d():
    for i in range(m):
        Initial_d[i] = Initial_basis[i]
    pos = m
    for i in range(Initial_n):
        flag = 1
        for j in range(m):
            if flag == 0:
                break
            if i == Initial_basis[j]:
                flag = 0
        if flag == 1:
            Initial_d[pos] = i
            pos += 1


# Initial_simplex_algorithm.c
def Initial_simplex_algorithm():
    count = 0
    optimal_flag = 0

    print(f" m = {m}, Initial_n = {Initial_n}")
    print("\nInitial_basis:")
    for i in range(m):
        print(f" {Initial_basis[i]} ", end='')
    print()

    print("Initial_A:")
    for i in range(m):
        for j in range(Initial_n):
            print(" %6.2lf "%Initial_A[i][j], end='')
        print()

    while optimal_flag == 0:
        bublesort(Initial_basis, m)
        print("\nInitial_basis:")
        for i in range(m):
            print(f" {Initial_basis[i]} ", end='')
        print()
        Initial_set_d()

        print("\nInitial_d:")
        for i in range(Initial_n):
            print(f" {Initial_d[i]} ", end='')
        print()

        set_Initial_A_aux()
        print("\nInitial_A_aux (B, D):")
        for i in range(m):
            for j in range(Initial_n):
                print(" %6.2lf "%Initial_A_aux[i][j], end='')
            print()
        copy_submatrix(Initial_B, Initial_A_aux, 0, m, 0, m)  # Set B
        print("\nInitial_B:")
        for i in range(m):
            for j in range(m):
                print(" %6.2lf "%Initial_B[i][j], end='')
            print()

        inv_gaussian(Initial_BI, Initial_B, m)  # BI = B - 1
        erase_epsilons_matrix(Initial_BI, m, m)

        print("\nInitial_BI:")
        for i in range(m):
            for j in range(m):
                print(" %6.2lf "%Initial_BI[i][j], end='')
            print()

        matrix_mult(Initial_BIA_aux, Initial_BI, Initial_A_aux, m, m, Initial_n)
        erase_epsilons_matrix(Initial_BIA_aux, m, Initial_n)

        print("\nInitial_BIA_aux (I, B-1*D):")
        for i in range(m):
            for j in range(Initial_n):
                print(" %6.2lf "%Initial_BIA_aux[i][j], end='')
            print()

        print("\nInitial_A_aux (B,D):")
        for i in range(m):
            for j in range(Initial_n):
                print(" %6.2lf "%Initial_A_aux[i][j], end='')
            print()

        print("Initial_b:")
        for i in range(m):
            print(" %6.2lf "%Initial_b[i],end='')

        matrix_vector_mult(Initial_BIb, Initial_BI, Initial_b, m, m)
        erase_epsilons_vector(Initial_BIb, m)

        print("\nInitial_BIb:")
        for i in range(m):
            print(" %6.2lf "%Initial_BIb[i], end='')
        print()

        copy_submatrix(Initial_D, Initial_A_aux, 0, m, m, Initial_n - m) # Set Initial_D
        print("Initial_D:")
        for i in range(m):
            for j in range(Initial_n-m):
                print(" %6.2lf "%Initial_D[i][j], end='')
            print()

        # END OF FOR DEBUG ONLY

        compute_Initial_cb_Initial_cd()

        print("\nInitial_cb:")
        for i in range(m):
            print(" %6.2lf "%Initial_cb[i], end='')
        print()

        print("\nInitial_cd:")
        for i in range(Initial_n-m):
            print(" %6.2lf "%Initial_cd[i], end='')
        print()

        # cbBI = cb * B-1
        vector_matrix_mult(Initial_cbBI, Initial_cb, Initial_BI, m, m)
        erase_epsilons_vector(Initial_cbBI, m)

        print("\nInitial_cbBI:")
        for i in range(m):
            print(" %6.2lf "%Initial_cbBI[i], end='')
        print()

        vector_matrix_mult(Initial_cbBID, Initial_cbBI, Initial_D, m, Initial_n - m)
        erase_epsilons_vector(Initial_cbBID, Initial_n - m)

        print("\nInitial_cbBID:")
        for i in range(Initial_n - m):
            print(" %6.2lf " %Initial_cbBID[i], end='')
        print()

        vector_subtract(Initial_rd, Initial_cd, Initial_cbBID, Initial_n - m)
        erase_epsilons_vector(Initial_rd, Initial_n - m)

        print("\nInitial_rd( cd - cbBID ):")
        for i in range(Initial_n-m):
            print(" %6.2lf "%Initial_rd[i], end='')
        print("\n")

        min_value = find_min_value(Initial_rd, n)
        if min_value >= 0.0:
            optimal_flag = 1
        else:
            enter_id = find_Initial_most_negative()
            exiting_id = find_Initial_exiting_id(Initial_BIA_aux, Initial_BIb, enter_id, Initial_n, m)
            print(f"\nenter_id  = {enter_id},  exiting_id = {exiting_id}, Initial_d[exiting_id] = {Initial_d[exiting_id]}")
            Initial_basis[exiting_id] = enter_id

            print("Initial_basis:\n")
            for i in range(m):
                print(f" {Initial_basis[i]} ", end='')
            print()


# Initial_swap_colums.c
def Initial_swap_colums(i, j):
    for k in range(m):
        Initial_A_aux[k][i] = Initial_A[k][j]
        Initial_A_aux[k][j] = Initial_A[k][i]


# invert_matrix.c
def inv_gaussian(B, A, n):
    for i in range(n):
        for j in range(n):
            W[i][j] = A[i][j]
    for i in range(n):
        for j in range(n, 2 * n):
            W[i][j] = 0.0

    for i in range(n):
        W[i][n + i] = 1.0

    print("\nBefore loop W: ")
    for i in range(n):
        for j in range(2 * n):
            print(" %8.2lf " %W[i][j], end='')
        print()

    for k in range(n):
        print(f"k = {k}")
        p = k
        MaxValue = abs(W[k][k])
        for i in range(k + 1, n):
            if abs(W[i][k]) >  MaxValue:
                p = i
                MaxValue = abs(W[i][k])
        print(f"p = {p}, k = {k}")
        if p != k:
            swap_rows(W, n, k, p)
        RelativeValue = W[k][k]
        print("RelativeValue = %8.2lf"%RelativeValue)
        W[k][k] = 1.0

        for j in range(k + 1, 2 * n):
            temp = W[k][j] / RelativeValue
            if abs(temp) < epsilon:
                W[k][j] = 0.0
            else:
                W[k][j] = temp

        for i in range(n):
            if i != k:
                RelativeValue = W[i][k]
                W[i][k] = 0.0
                for j in range(k + 1, (2 * n)+1):
                    temp = W[i][j] - RelativeValue * W[k][j]
                    if abs(temp) < epsilon:
                        W[i][j] = 0.0
                    else:
                        W[i][j] = temp

        print(" W: ")
        for i in range(n):
            for j in range(2 * n):
                print(" %8.2lf " %W[i][j], end='')
            print()
    for i in range(n):
        for j in range(n):
            B[j][i] = W[j][i + n]

    print("\nBI:")
    for i in range(n):
        for j in range(n):
            print(" %lf "%B[i][j], end='')
        print()

    print("\nW:")
    for i in range(n):
        for j in range(2 * n):
            print(" %8.2lf " %W[i][j], end='')
        print()


#matrix_mult.c
def matrix_mult(C,A,B,n,m,p):
    for i in range(n):
        for j in range(p):
            col = OptList(x[j] for x in B)
            C[i][j]=A[i]*col
            '''
            s=0
            for k in range (m):
                s+=A[i][k]*B[k][j]
            C[i][j]=s
            '''


#matrix_vector_mult.c

def matrix_vector_mult(c, A, b, n, m):
    for i in range(n):
        row = OptList(A[i])
        c[i] = row * b
    '''
    for i in range(n):
        s=0
        for k in range(m):
            s+=A[i][k]*b[k]
        c[i]=s
    '''


# print_initial_solution.c
def print_initial_solution():
    print("\nInitial basis:")
    for i in range(m):
        print(f" {Initial_basis[i]} ", end='')
    print()
    print("\nBasic Solution:")
    for i in range(m):
        print(" X%d = %lf "% (Initial_basis[i], Initial_BIb[i]), end='')
    print()

# print_no_solution.c
def print_no_solution():
    print("System A has NO solution")

# print_original_system.c
def print_original_system():
    print("Original System:")
    for i in range(m):
        for j in range(n):
            print("%10.3lf"%A[i][j], end='')
        print()

# print_result.c
def print_result():
    print("Optimal Basis:")
    for i in range(m):
        print(f" {basis[i]} ", end='')
    print()

    print("Optimal Solution:")
    for i in range(m):
        print(" X%d = %lf " %(basis[i], BIb[i]), end='')
    print()

# print_simplex_params.c
def print_simplex_params(A, A_aux, c, b, n, m, B, BID, D, basis, d, cb, cd):
    count = 0
    print(f" m = {m}, n = {n}")

    print("A:")
    for i in range(m):
        for j in range(n):
            print(" %lf "%A[i][j], end='')
        print()

    print("c:")
    for i in range(n):
        print(" %lf "%c[i], end='')
    print()

    print("b:")
    for i in range(m):
        print(" %lf "%b[i], end='')
    print()

    print("A_aux:")
    for i in range(m):
        for j in range(n):
            print(" %lf "%A_aux[i][j], end='')
        print()

    print("B:")
    for i in range(m):
        for j in range(m):
            print(" %lf "%B[i][j], end='')
        print()

    print("basis:")
    for i in range(m):
        print(f" {basis[i]} ", end='')
    print()

    count += 1

    if count >= 8:
        sys.exit(0)


# print_solution.c
def print_solution():
    print("\nbasis:")
    for i in range(m):
        print(f" {basis[i]} ", end='')
    print()

    print("\nBasic Solution:")
    for i in range(m):
        print(" X%d = %lf "%(basis[i]+1, BIb[i]), end='')
    print()
    print("\nSolution value:")

    temp = c[basis[0]] * BIb[0]
    sum = temp
    print(" %lf * %lf "%(c[basis[0]], BIb[0]),end="")
    for i in range(1, m):
        temp = c[basis[i]] * BIb[i];
        sum = sum + temp;
        print(" +  %lf * %lf "%(c[basis[i]], BIb[i]), end='')
    print(" = %lf"%sum)


# read_file.c
def read_file():
    temp = fp.readline()
    print(f"str = {temp}")
    temp = fp.readline()
    lst = temp.strip("\n").split()
    for i in range(n):
        c[i]=float(lst[i])
    for i in range(n):
        print(f"c[{i}] = %lf" %c[i])
    print(" str = \n")
    temp = fp.readline()
    print(f"A: str = {temp}")
    for i in range(m):
        temp = fp.readline()
        lst = temp.strip("\n").split()
        for j in range(n):
            A[i][j]=float(lst[j])

    for i in range(m):
        for j in range(n):
            print(" %lf " % A[i][j], end="")
        print()
    print(" str = \n")
    temp = fp.readline()
    print(f" b: str = {temp}")

    temp = fp.readline()
    lst = temp.strip("\n").split()
    for i in range(m):
        b[i]=float(lst[i])
    temp = fp.readline()  # epsilon
    print(f" str = {temp}")
    temp = fp.readline()
    lst = temp.strip("\n").split()
    global epsilon
    epsilon = float(lst[0])
    print("b: ")
    for i in range(m):
        print(" %lf "%b[i], end="")
    print()
    copy_matrix(A_aux,A,n,m)
# set_A_aux.c
def set_A_aux():
    for i in range(n):
        k = d[i]
        for j in range(m):
            A_aux[j][i] = A[j][k]

# set_d
def set_d():
    print(d)
    for i in range(m):
        d[i] = basis[i]
    pos = m
    for i in range(n):
        flag = 1
        for j in range(m):
            if flag == 0:
                break
            if i == basis[j]:
                flag = 0
        if flag == 1:
            d[pos] = i
            pos += 1
    print(d)

# set_Initial_A_aux.c
def set_Initial_A_aux():
    for i in range(Initial_n):
        k = Initial_d[i]
        for j in range(m):
            Initial_A_aux[j][i] = Initial_A[j][k]


# simplex_algorithm.c
def simplex_algorithm():
    optimal_flag = 0
    print(f" m = {m}, n = {n}")
    print("\nbasis:")
    for i in range(m):
        print(f" {basis[i]} ",end="")
    print()

    print("A:")
    for i in range (m):
        for j in range (n):
            print(" %6.2lf "%A[i][j],end="")
        print()

    while optimal_flag==0:
        bublesort(basis,m)
        print("\nbasis:")
        for i in range(m):
            print(f" {basis[i]} ",end="")
        print()

        set_d()

        print("\nd:")
        for i in range(n):
            print(f" {d[i]} ",end="")
        print()

        set_A_aux()

        print("\nA_aux (B, D):")
        for i in range(m):
            for j in range(n):
                print(" %6.2lf "%A_aux[i][j],end="")
            print()

        copy_submatrix(B,A_aux,0,m,0,m)

        print("\nB:")
        for i in range(m):
            for j in range(m):
                print(" %6.2lf " %B[i][j],end="")
            print()

        inv_gaussian(BI, B, m)
        erase_epsilons_matrix(BI, m, m)

        print("\nBI:")
        for i in range(m):
            for j in range(m):
                print(" %6.2lf "%BI[i][j],end="")
            print()

        matrix_mult(BIA_aux, BI, A_aux,m, m, n)
        erase_epsilons_matrix(BIA_aux, m, n)

        print("\nBIA_aux (I, B-1*D):")
        for i in range(m):
            for j in range(n):
                print(" %6.2lf "%BIA_aux[i][j],end="")
            print()

        print("\nA_aux (B,D):")
        for i in range(m):
            for j in range(n):
                print(" %6.2lf "%A_aux[i][j],end="")
            print()

        print("b:")
        for i in range(m):
            print(" %6.2lf "%b[i],end=" ")

        matrix_vector_mult(BIb, BI, b, m, m)
        erase_epsilons_vector(BIb, m)
        print("\nBIb:")
        for i in range(m):
            print(" %6.2lf "%BIb[i],end=" ")
        print()

        copy_submatrix(D, A_aux,0, m, m, n - m)
        print("D:")
        for i in range(m):
            for j in range(n-m):
                print(" %6.2lf "%D[i][j],end="")
            print()

        compute_cb_cd()
        print("\ncb:")
        for i in range(m):
            print(" %6.2lf "%cb[i],end="")
        print()

        print("\ncd:")
        for i in range(n-m):
            print(" %6.2lf "%cd[i],end="")
        print()

        vector_matrix_mult(cbBI, cb, BI, m, m)
        erase_epsilons_vector(cbBI, m)

        print("\ncbBI:")
        for i in range(m):
            print(" %6.2lf "%cbBI[i],end="")
        print()

        vector_matrix_mult(cbBID, cbBI, D,m, n - m)
        erase_epsilons_vector(cbBID, n - m)

        print("\ncbBID:")
        for i in range(n-m):
            print(" %6.2lf " %cbBID[i],end="")
        print()

        vector_subtract(rd, cd, cbBID,n - m)
        erase_epsilons_vector(rd, n - m)

        print("\nrd( cd - cbBID ):")
        print("\nXXd:",end="")
        for i in range(n-m):
            print(" %6d "%d[i + m],end="")
        print("\nXXrd:",end="")
        for i in range(n-m):
            print(" %6.2lf "%rd[i],end="")
        print("\n")

        min_value = find_min_value(rd, n)
        if min_value>=0.0:
            optimal_flag=1
        else:
            negative_rds=OptList(OptList([0 for _ in range(N)]))
            n_n_rds=None
            temp_improvement=None
            temp_improvement1=[0]
            n_n_rds_temp = [0]
            find_all_negative_rds(negative_rds, n_n_rds_temp)
            n_n_rds=n_n_rds_temp[0]
            best_improvement = 0.0
            best_improvement_id = -1
            for index in range (n_n_rds):
                enter_id = negative_rds[index]
                for k in range(n-m):
                    if d[m+k]==enter_id:
                        q=k
                print("\nXXenter_id = %d, q = %d, rd[q] = %lf"%(enter_id,q,rd[q]))
                exiting_id =find_exiting_id(BIA_aux, BIb,enter_id, n, m, temp_improvement1)
                temp_improvement=temp_improvement1[0]
                print("\n1:temp_improvement = %lf,  best_improvement = %lf"%(temp_improvement, best_improvement))
                temp_improvement = -rd[q] * temp_improvement
                print("\n2:temp_improvement = %lf,  best_improvement = %lf"%(temp_improvement, best_improvement))
                if temp_improvement >= best_improvement:
                    best_improvement = temp_improvement
                    best_improvement_id = enter_id
                    best_exiting_id = exiting_id

            enter_id = best_improvement_id
            exiting_id = best_exiting_id
            print("pivot: enter_id = %d, exiting_id = %d" %(enter_id,d[exiting_id]))
            basis[exiting_id] = enter_id
            print("\nbasis:")
            for i in range(m):
                print(" %d "%basis[i])
            print()


# stage2_swap_colums.c
def stage2_swap_colums(i, j):
    for k in range(m):
        A_aux[k][i] = A[k][j]
        A_aux[k][j] = A[k][i]


# swap_colums.c
def swap_colums(A, i, j, m, n):
    for k in range(m):
        temp = A[k][i]
        A[k][i] = A[k][j]
        A[k][j] = temp


# swap_rows.c
def swap_rows(W, n, m1, m2):
    for i in range(2*n + 1):
        temp = W[m1][i]
        W[m1][i] = W[m2][i]
        W[m2][i] = temp


#vector_matrix_mult.c
def vector_matrix_mult(c,b,A,n,m):
    for i in range(m):
        col=OptList([row[i] for row in A])
        c[i] = col*b
    '''
    for i in range(m):
        s=0
        for k in range(n):
            s+=A[k][i]*b[k]
        c[i]=s
    '''

# vector_subtract.c
def vector_subtract(result_v,v1,v2,n):
    temp=v1-v2
    for i in range(n):
        result_v[i]=temp[i]
    '''
    for i in range(n):
        result_v[i]=v1[i]-v2[i]
    '''


#simplex_main.c
if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    path=filedialog.askopenfilename(initialdir='C:/',title="Select a file to run simplex",filetypes=[("text files","*.txt")])
    #path="C:/Users/mblac/Desktop/simplex1/simplex/abs1.txt"
    fp=open(path,"r")
    temp=fp.readline()
    print(f"str = {temp}")
    temp=fp.readline()
    lst=temp.strip("\n").split()
    m=int(lst[0])
    n=int(lst[1])
    print(f"n = {n}, m = {m}")
    Initial_n = n + m
    print(f"str = \n")
    n_p_m = n+m

    read_file()

    print("\nepsilon = %6.2lf" %epsilon)
    print(" A: ")
    print_original_system()

    copy_to_initial_matrix()

    print("Initial_A:")
    for i in range(m):
        for j in range(n_p_m):
            print(" %6.2lf "%Initial_A[i][j],end="")
        print()

    for i in range(m):
        Initial_basis[i] = i + n

    for i in range(n):
        Initial_c[i] = 0.0

    for i in range(n,Initial_n):
        Initial_c[i] = 1.0

    for i in range(m):
        Initial_b[i] = b[i]
    for i in range(m):
        Initial_b_aux[i] = b[i]

    print("\nInitial_basis:")
    for i in range(m):
        print(" %d " %Initial_basis[i],end="")
    print()

    print("\nInitial_c:")
    for i in range (Initial_n):
        print(" %6.2lf "%Initial_c[i],end="")
    print()

    print("\nInitial_b:")
    for i in range(m):
        print(" %6.2lf "%Initial_b[i],end="")
    print()

    Initial_simplex_algorithm()

    for i in range(m):
        itemp = Initial_basis[i]
        basis[i] = itemp
        if itemp >= n:
            print_no_solution()
            sys.exit(0)

    print_initial_solution()

    simplex_algorithm()

    bublesort_d(basis, BIb, m)

    print_solution()
    fp.close()