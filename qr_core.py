import numpy as np

np.set_printoptions(precision=5)

DECIMALS = 5
def replace_elems_close_to_zero(A, decimals):
    EPSILON = 1.0/(10**decimals)
    # A = A.round(decimals)
    A[np.abs(A.round(decimals)) < EPSILON] = 0
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
    bottom = np.sqrt(a*a + b*b)
    alpha = a / bottom
    beta = -b / bottom

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
                replace_elems_close_to_zero(R, DECIMALS+3)
                # print('G*R = \n', R)
                QT = QT * G.T
                # print('QT*G.T = \n', QT)

    replace_elems_close_to_zero(QT, DECIMALS+3)
    replace_elems_close_to_zero(R, DECIMALS)
    Q = QT.T
    print('Q: \n{}\nR: \n{} \nQ * A = R\nA = Q.T * R = \n{}'.format(Q, R, replace_elems_close_to_zero(QT * R, 3)))

    return Q, replace_elems_close_to_zero(R, DECIMALS)


def OLS_from_QR(R, b):
    # print('OLS input: R:\n', R, '\nb=\n', b)
    if R.shape[0] > R.shape[1]:
        p = R.shape[1]  # p = number of parameters including intercept
        w = np.linalg.solve(a=R[0:p], b=b[0:p])
    else:
        # Solve: R * w = b
        w = np.linalg.solve(a=R, b=b)
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
        A1 = replace_elems_close_to_zero(old_Q.T * old_R, DECIMALS)  # original matrix A1

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
