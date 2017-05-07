import numpy as np

A = np.matrix([[6, 5, 0],
               [5, 1, 4],
               [0, 4, 3]], dtype=float)
A2 = np.matrix([[6, 5, 0],
               [5, 1, 4],
               [0, 4, 3]], dtype=float)

pos = [0, 1]
print(A[pos[0], pos[1]])

def replace_elems_close_to_zero(A):
    EPSILON = 0.000000001
    A[np.abs(A) < EPSILON] = 0

def get_rotation_matrix(A, pos_to_zero):
    """
    Creates a matrix that will zero out element of matrix A at position pos_to_zero.
    :param A: Input np.matrix
    :param pos_to_zero: Position of element to zero out. array
    :return: New np.matrix G
    """
    i = pos_to_zero[0]
    j = pos_to_zero[1]

    G = np.identity(A.shape[0], dtype=float)
    # [a, b] = vector, b will be zeroed out by matrix:
    #
    # |c -s| |a| = |r|
    # |s  c| |b| = |0|
    a = A[i - 1, j]
    b = A[i, j]

    r = np.sqrt(a * a + b * b)
    c = a / r
    s = -b / r

    G[i - 1, j] = c
    G[i - 1, j + 1] = -s
    G[i, j] = s
    G[i, j + 1] = c

    return np.matrix(G)


G1 = get_rotation_matrix(A, [1,0])
print('G1 for [1,0]:\n',G1)
A= G1*A
replace_elems_close_to_zero(A)
print('G1*A for [1,0]:\n',A)

G2 = get_rotation_matrix(A, [2,1])
print('G2 for [2,1]:\n',G2)
A= G2*A
replace_elems_close_to_zero(A)
print('G2*A for [2,1]:\n',A)

R=A
QT = G1.T*G2.T
print('Q.T:\n', QT)

print('Q * A = R\nA = Q.T * R\n', QT*R)

def QR_decomposition(A):
    """
    QR decomposition of matrix A using Givens rotations.
    https://en.wikipedia.org/wiki/Givens_rotation
    :param A: Matrix to decompose.
    :return: Q,R
    """
    print('QR decomposition for matrix A: \n', A)
    QT = np.matrix(np.identity(A.shape[0]))
    R = np.matrix(A)
    restart = True
    while restart:
        restart = False
        # If A is not triangular - restart search for another non-zero element
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if j >= i:
                    #iterate only under diagonal
                    break

                if R[i,j] != 0:
                    print('Zeroing elem at ', [i,j])
                    G = get_rotation_matrix(R, [i,j])
                    #print('G to zero elem at {} = \n{}'.format([i,j], G))
                    R = G*R
                    replace_elems_close_to_zero(R)
                    #print('G*R = \n', R)
                    QT = QT*G.T
                    #print('QT*G.T = \n', QT)
                    restart = True
                    break

            if restart: break

    replace_elems_close_to_zero(QT)
    Q = QT.T
    print('Q: \n{}\nR: \n{} \nQ * A = R\nA = Q.T * R = \n{}'.format(Q, R, QT*R))
    return Q, R

print('*****************')
QR_decomposition(A2)