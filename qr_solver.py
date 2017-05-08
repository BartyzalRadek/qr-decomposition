import numpy as np
import argparse
from numpy import genfromtxt

from util import remove_target_column
from qr_core import QR_decomposition
from qr_core import OLS_from_QR

parser = argparse.ArgumentParser(description='QR decomposition solver by Radek Bartyzal')
parser.add_argument('-data', help='Path to CSV file containing an input matrix.', required=True)
parser.add_argument('-sep', help='Separator of input data.', required=True)
parser.add_argument('-ycol', help='Number of target column for OLS. Indexed from 0.', required=True)
parser.add_argument('-Q', help='Path to CSV file with precalculated Q matrix.', required=False)
parser.add_argument('-R', help='Path to CSV file with precalculated R matrix.', required=False)
# args = vars(parser.parse_args())
args = vars(parser.parse_args("-data=A.csv -sep=, -ycol=3".split(' ')))
# args = vars(parser.parse_args("-data=A.csv -sep=, -ycol=3 -Q=Q.csv -R=R.csv".split()))

PATH = args['data']
SEPARATOR = args['sep']
ycol = int(args['ycol'])
print(args)
input_matrix = np.matrix(genfromtxt(PATH, delimiter=SEPARATOR))

print('Read input matrix:\n', input_matrix)

b = input_matrix[:,ycol]
A = remove_target_column(input_matrix, ycol)

print('A:\n', A)
print('b:\n', b)

if args['Q'] is None or args['R'] is None:
    print('Running QR decomposition from scratch, Q/R were not specified...')
    Q, R = QR_decomposition(old_Q=None, old_R=None, A2=A)
else:
    print('Using precalculated Q and R')
    old_Q = np.matrix(genfromtxt(args['Q'], delimiter=SEPARATOR))
    old_R = np.matrix(genfromtxt(args['R'], delimiter=SEPARATOR))
    Q, R = QR_decomposition(old_Q=old_Q, old_R=old_R, A2=A)


np.savetxt("Q.csv", Q, delimiter=",", fmt='%.5f')
np.savetxt("R.csv", R, delimiter=",", fmt='%.5f')
print('Saved Q to Q.csv')
print('Saved R to R.csv')
OLS_solution = OLS_from_QR(R=R, b=b)
print('OLS solution = \n', OLS_solution)

