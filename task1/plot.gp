set logscale x 2
set title "K = 1, N = 20"
set xlabel "PROCS"
set ylabel "ACCELERATION"
set output 'k1.png'
plot "k1.txt" using 1:2 with lp

