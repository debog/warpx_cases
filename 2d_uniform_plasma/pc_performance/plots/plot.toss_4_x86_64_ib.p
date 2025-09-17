set terminal postscript enhanced eps color "Times" 18

set style line 11 dt 2 lw 2 lc rgbcolor "black"        pt  5 ps 2
set style line 12 dt 4 lw 1 lc rgbcolor "blue"         pt  5 ps 2

set style line 21 dt 4 lw 1 lc rgbcolor "dark-green"   pt  6 ps 1
set style line 22 dt 4 lw 1 lc rgbcolor "dark-green"   pt  8 ps 1
set style line 23 dt 4 lw 1 lc rgbcolor "dark-green"   pt 10 ps 1
set style line 24 dt 4 lw 1 lc rgbcolor "dark-green"   pt 12 ps 1
set style line 25 dt 4 lw 1 lc rgbcolor "dark-green"   pt 14 ps 1

set style line 31 dt 4 lw 1 lc rgbcolor "red"          pt  6 ps 1
set style line 32 dt 4 lw 1 lc rgbcolor "red"          pt  8 ps 1
set style line 33 dt 4 lw 1 lc rgbcolor "red"          pt 10 ps 1
set style line 34 dt 4 lw 1 lc rgbcolor "red"          pt 12 ps 1

set format x "%1.1f"
set format y "%g"

set key width 0

set grid  xtics lw 1 dt 4 lc rgbcolor "gray"
set grid  ytics lw 1 dt 4 lc rgbcolor "gray"
set grid mxtics lw 1 dt 4 lc rgbcolor "gray"
set grid mytics lw 1 dt 4 lc rgbcolor "gray"

set output "pc_performance.Dane.eps"
set title "2D uniform plasma, 512^2 grid, 32^2 ppc, 10 time steps"
set xlabel "Wall time (s)"
set ylabel "Average number of GMRES iterations per Newton iteration"
set xrange [0:500]
set yrange [0:25]
set key outside right
plot \
'../wtimes.petsc_ksp.pc_none.toss_4_x86_64_ib.nx00512.np032.dat'                            u 4:($11/$13) w p ls 11 t "No PC",\
'../wtimes.petsc_ksp.pc_petsc_lu.toss_4_x86_64_ib.nx00512.np032.dat'                        u 4:($11/$13) w p ls 12 t "PCLU",\
'../wtimes.petsc_ksp.pc_petsc_asm_lu_asmovlp000.toss_4_x86_64_ib.nx00512.np032.dat'         u 4:($11/$13) w p ls 21 t "PCASM ( 0)(LU)",\
'../wtimes.petsc_ksp.pc_petsc_asm_lu_asmovlp008.toss_4_x86_64_ib.nx00512.np032.dat'         u 4:($11/$13) w p ls 22 t "PCASM ( 8)(LU)",\
'../wtimes.petsc_ksp.pc_petsc_asm_lu_asmovlp016.toss_4_x86_64_ib.nx00512.np032.dat'         u 4:($11/$13) w p ls 23 t "PCASM (16)(LU)",\
'../wtimes.petsc_ksp.pc_petsc_asm_lu_asmovlp024.toss_4_x86_64_ib.nx00512.np032.dat'         u 4:($11/$13) w p ls 24 t "PCASM (24)(LU)",\
'../wtimes.petsc_ksp.pc_petsc_asm_lu_asmovlp032.toss_4_x86_64_ib.nx00512.np032.dat'         u 4:($11/$13) w p ls 25 t "PCASM (32)(LU)",\
'../wtimes.petsc_ksp.pc_ccmlmg_nvcyc001.toss_4_x86_64_ib.nx00512.np032.dat'                 u 4:($11/$13) w p ls 31 t "PCCCMLMG (1)",\
'../wtimes.petsc_ksp.pc_ccmlmg_nvcyc002.toss_4_x86_64_ib.nx00512.np032.dat'                 u 4:($11/$13) w p ls 32 t "PCCCMLMG (2)",\
'../wtimes.petsc_ksp.pc_ccmlmg_nvcyc004.toss_4_x86_64_ib.nx00512.np032.dat'                 u 4:($11/$13) w p ls 33 t "PCCCMLMG (4)",\
'../wtimes.petsc_ksp.pc_ccmlmg_nvcyc008.toss_4_x86_64_ib.nx00512.np032.dat'                 u 4:($11/$13) w p ls 34 t "PCCCMLMG (8)",\

