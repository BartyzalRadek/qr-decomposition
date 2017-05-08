import numpy as np

A = np.matrix([[6, 5, 0],
               [5, 1, 4],
               [0, 4, 3]], dtype=float)
A2 = np.matrix([[6, 5, 0],
                [5, 1, 4],
                [0, 4, 3]], dtype=float)
A3 = np.matrix([[6, 5, 0],
                [5, 1, 4],
                [3, 2, 1],
                [0, 4, 3]], dtype=float)
A4 = np.matrix([[6, 5, 0],
                [0, 4, 3]], dtype=float)

pos = [0, 1]
print('A=\n', A)


def replace_elems_close_to_zero(A):
    EPSILON = 0.000000001
    A[np.abs(A) < EPSILON] = 0
    return A


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
    # [a, b] = vector, b will be zeroed out by matrix, a = A[j,j]
    #
    # |alpha -beta| |a| = |r|
    # |beta  alpha| |b| = |0|
    a = A[j, j]
    b = A[i, j]
    alpha = 0
    beta = 0
    if (np.abs(b) >= np.abs(a)):
        t = -a / b
        beta = 1 / np.sqrt(1 + t * t)
        alpha = beta * t
    else:
        t = -b / a
        alpha = 1 / np.sqrt(1 + t * t)
        beta = alpha * t

    G[i, i] = alpha
    G[j, j] = alpha
    G[i, j] = beta
    G[j, i] = -beta

    return np.matrix(G)


G1 = get_rotation_matrix(A, [1, 0])
print('G1 for [1,0]:\n', G1)
A = G1 * A
replace_elems_close_to_zero(A)
print('G1*A for [1,0]:\n', A)

G2 = get_rotation_matrix(A, [2, 1])
print('G2 for [2,1]:\n', G2)
A = G2 * A
replace_elems_close_to_zero(A)
print('G2*A for [2,1]:\n', A)

R = A
QT = G1.T * G2.T
print('Q.T:\n', QT)

print('Q * A = R\nA = Q.T * R\n', QT * R)


def QR_decomposition(A):
    """
    QR decomposition of matrix A using Givens rotations.
    :param A: Matrix to decompose.
    :return: Q,R
    """
    print('QR decomposition for matrix A: \n', A)
    QT = np.matrix(np.identity(A.shape[0]))
    R = np.matrix(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if j >= i:
                # iterate only under diagonal
                break

            if R[i, j] != 0:
                print('Zeroing elem at ', [i, j])
                G = get_rotation_matrix(R, [i, j])
                # print('G to zero elem at {} = \n{}'.format([i,j], G))
                R = G * R
                replace_elems_close_to_zero(R)
                # print('G*R = \n', R)
                QT = QT * G.T
                # print('QT*G.T = \n', QT)

    replace_elems_close_to_zero(QT)
    Q = QT.T
    print('Q: \n{}\nR: \n{} \nQ * A = R\nA = Q.T * R = \n{}'.format(Q, R, QT * R))
    return Q, R


def OLS_from_QR(R, b):
    w = np.linalg.solve(a=R, b=b)
    return w


def QR_decomposition2(old_Q, old_R, A2):
    """
    QR decomposition of matrix A using Givens rotations.
    :param old_Q: Q calculated for matrix A1
    :param old_R: R calculated for matrix A1
    :param A2: New additional rows to original matrix A1
    :return: QR decompositioin of matrix A = concatenated A1 with A2 using precalculated Q, R from A1
    """

    if old_Q==None or old_R==None:
        print('QR decomposition for matrix A from scratch: \n', A2)
        QT = np.matrix(np.identity(A2.shape[0]))
        R = np.matrix(A2)
    else:
        print('QR decomposition of matrix with added rows: \n')
        A1 = replace_elems_close_to_zero(old_Q.T * old_R)  # original matrix A1

        if A1.shape[1] != A2.shape[1]:
            print('Provided matrix has different number of columns ({}) than the original one ({}).\n'
                  'Returning...'.format(A2.shape[1], A1.shape[1]))
            return None

        A = np.concatenate((A1, A2), axis=0)  # Add all rows from A2 to A1 = create whole matrix A
        QT = np.matrix(np.identity(A.shape[0]))

        for i in range(old_Q.shape[0]):
            for j in range(old_Q.shape[1]):
                QT[i, j] = old_Q.T[i, j]


        # R = replace_elems_close_to_zero(QT.T * A) # equal to: np.concatenate((old_R, A2), axis=0)
        R = np.concatenate((old_R, A2), axis=0)
        print('Full matrix with added rows:\n', A)
        print('precalculated Q:\n', old_Q)
        print('precalculated R:\n', old_R)
        print('prepared Q from precalculated old_Q:\n', QT.T)
        print('prepared R = Q*A from prepared Q and full A:\n', R)

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if j >= i:
                # iterate only under diagonal
                break

            if R[i, j] != 0:
                print('Zeroing elem at ', [i, j])
                G = get_rotation_matrix(R, [i, j])
                # print('G to zero elem at {} = \n{}'.format([i,j], G))
                R = G * R
                replace_elems_close_to_zero(R)
                # print('G*R = \n', R)
                QT = QT * G.T
                # print('QT*G.T = \n', QT)

    replace_elems_close_to_zero(QT)
    Q = QT.T
    print('Q: \n{}\nR: \n{} \nQ * A = R\nA = Q.T * R = \n{}'.format(Q, R, replace_elems_close_to_zero(QT * R)))
    return Q, R



print('*****************')
Q, R = QR_decomposition(A3)
print('*****************')
QR_decomposition2(old_Q=Q, old_R=R, A2=A4)
