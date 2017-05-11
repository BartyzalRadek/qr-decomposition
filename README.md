# QR Decomposition

## Examples:

### Calculate Q and R for input matrix stored in `A.csv`:

```
python qr_solver.py -data=A.csv -sep=, -ycol=3
```

### Recalculate QR for an updated matrix

Append matrix stored in `A4.csv` to previous matrix stored in `A.csv` and recalculate new Q/R while using the previously obtained Q/R 
saved in `Q.csv` and `R.csv` :                               

```
python qr_solver.py -data=A4.csv -sep=, -ycol=3 -prev_data=A.csv -Q=Q.csv -R=R.csv
```