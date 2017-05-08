import numpy as np
import pandas as pd
from qr_core import get_rotation_matrix
from qr_core import replace_elems_close_to_zero
from qr_core import QR_decomposition

from numpy import genfromtxt

A = np.matrix([[6, 5, 0, 1],
               [5, 1, 4, 2],
               [0, 4, 3, 3]], dtype=float)
A2 = np.matrix([[6, 5, 0],
                [5, 1, 4],
                [0, 4, 3]], dtype=float)
A3 = np.matrix([[6, 5, 0],
                [5, 1, 4],
                [3, 2, 1],
                [0, 4, 3]], dtype=float)
A4 = np.matrix([[6, 5, 0],
                [0, 4, 3]], dtype=float)
np.savetxt("A.csv", A, delimiter=",", fmt='%.5f')

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
Q2, R2 = QR_decomposition(old_Q=Q, old_R=R, A2=A4)
print('*****************')
np.set_printoptions(precision=5)
np.savetxt("A.csv", A, delimiter=",", fmt='%.5f')

my_data = np.matrix(genfromtxt('A.csv', delimiter=','))
print('Data\n', my_data)

