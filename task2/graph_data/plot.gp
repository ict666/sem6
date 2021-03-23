set title "K = 11, N = 28"
set xlabel "PROCS"
set ylabel "ACCELERATION"
set output 'k1.png'
plot "k11.txt" using 1:2 with lp
plot "k1.txt" using 1:2 with lp title "K = 1", "k11.txt" using 1:2 with lp title "K = 11", "kn.txt" using 1:2 with lp title "K = N"  

