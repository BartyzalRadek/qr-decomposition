import numpy as np


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


def run_decomposition(R, QT, A):
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


def OLS_from_QR(R, b):
    p = R.shape[1]  # p = number of parameters including intercept
    # Solve: R * w = b
    w = np.linalg.solve(a=R[0:p + 1], b=b[0:p + 1])
    return w


def QR_decomposition(old_Q, old_R, A2):
    """
    QR decomposition of matrix A using Givens rotations.
    :param old_Q: Q calculated for matrix A1
    :param old_R: R calculated for matrix A1
    :param A2: New additional rows to original matrix A1
    :return: QR decomposition of matrix A = concatenated A1 with A2 using precalculated Q, R from A1
    """

    if old_Q is None or old_R is None:
        A = A2
        print('QR decomposition for matrix A from scratch: \n', A)
        QT = np.matrix(np.identity(A.shape[0]))
        R = np.matrix(A)
    else:
        print('QR decomposition of matrix with added rows: \n')
        A1 = replace_elems_close_to_zero(old_Q.T * old_R)  # original matrix A1

        if A1.shape[1] != A2.shape[1]:
            print('Provided matrix has different number of columns ({}) than the original one ({}).\n'
                  'Returning...'.format(A2.shape[1], A1.shape[1]))
            return None

        A = np.concatenate((A1, A2), axis=0)  # Add all rows from A2 to A1 = create whole matrix A
        QT = np.matrix(np.identity(A.shape[0]))

        # Insert precalculated old_Q into a new larger Q
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

    return run_decomposition(R, QT, A)
