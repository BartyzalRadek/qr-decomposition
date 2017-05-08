import numpy as np
from qr import get_rotation_matrix
from qr import replace_elems_close_to_zero
from qr import QR_decomposition

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


# G1 = get_rotation_matrix(A, [1, 0])
# print('G1 for [1,0]:\n', G1)
# A = G1 * A
# replace_elems_close_to_zero(A)
# print('G1*A for [1,0]:\n', A)
#
# G2 = get_rotation_matrix(A, [2, 1])
# print('G2 for [2,1]:\n', G2)
# A = G2 * A
# replace_elems_close_to_zero(A)
# print('G2*A for [2,1]:\n', A)
#
# R = A
# QT = G1.T * G2.T
# print('Q.T:\n', QT)
#
# print('Q * A = R\nA = Q.T * R\n', QT * R)

print('*****************')
Q, R = QR_decomposition(old_Q=None, old_R=None, A2=A)
print('*****************')
QR_decomposition(old_Q=Q, old_R=R, A2=A4)