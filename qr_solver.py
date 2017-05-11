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
parser.add_argument('-prev_data', help='Path to CSV file containing the previous input matrix.', required=False)
parser.add_argument('-Q', help='Path to CSV file with precalculated Q matrix.', required=False)
parser.add_argument('-R', help='Path to CSV file with precalculated R matrix.', required=False)
args = vars(parser.parse_args())

# Calculate Q and R for matrix A.csv
# args = vars(parser.parse_args("-data=A.csv -sep=, -ycol=3".split(' ')))

# Append matrix A4.csv to previous matrix A and recalculate Q and R while using
# the previously obtained Q and R
# args = vars(parser.parse_args("-data=A4.csv -sep=, -ycol=3 -prev_data=A.csv -Q=Q.csv -R=R.csv".split(' ')))

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

new_Q_path = 'Q.csv'
new_R_path = 'R.csv'

if args['Q'] is None or args['R'] is None or args['prev_data'] is None:
    print('Running QR decomposition from scratch, Q/R were not specified...')
    Q, R = QR_decomposition(old_Q=None, old_R=None, A2=A)
else:
    print('Using precalculated Q and R')
    prev_input_matrix = np.matrix(genfromtxt(args['prev_data'], delimiter=SEPARATOR))
    prev_b = prev_input_matrix[:,ycol]
    b = np.concatenate((b, prev_b), axis=0)

    old_Q = np.matrix(genfromtxt(args['Q'], delimiter=SEPARATOR))
    old_R = np.matrix(genfromtxt(args['R'], delimiter=SEPARATOR))
    Q, R = QR_decomposition(old_Q=old_Q, old_R=old_R, A2=A)

    new_Q_path = 'Q2.csv'
    new_R_path = 'R2.csv'


np.savetxt(new_Q_path, Q, delimiter=",", fmt='%.5f')
np.savetxt(new_R_path, R, delimiter=",", fmt='%.5f')
print('Saved Q to ', new_Q_path)
print('Saved R to ', new_R_path)
OLS_solution = OLS_from_QR(R=R, b=b)
print('OLS solution = \n', OLS_solution)

