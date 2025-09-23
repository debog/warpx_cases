set terminal postscript enhanced eps color "Times" 8

set style line 11 dt 2 lw 2 lc rgbcolor "black"              pt  4 ps 1
set style line 12 dt 4 lw 1 lc rgbcolor "blue"               pt  4 ps 1

set style line 21 dt 4 lw 1 lc rgbcolor "dark-green"         pt  6 ps 1
set style line 22 dt 4 lw 1 lc rgbcolor "dark-green"         pt  8 ps 1
set style line 23 dt 4 lw 1 lc rgbcolor "dark-green"         pt 10 ps 1
set style line 24 dt 4 lw 1 lc rgbcolor "dark-green"         pt 12 ps 1
set style line 25 dt 4 lw 1 lc rgbcolor "dark-green"         pt 14 ps 1

set style line 31 dt 4 lw 1 lc rgbcolor "red"                pt  6 ps 1
set style line 32 dt 4 lw 1 lc rgbcolor "red"                pt  8 ps 1
set style line 33 dt 4 lw 1 lc rgbcolor "red"                pt 10 ps 1
set style line 34 dt 4 lw 1 lc rgbcolor "red"                pt 12 ps 1

set style line 41 dt 4 lw 1 lc rgbcolor "light-magenta"      pt  6 ps 1
set style line 42 dt 4 lw 1 lc rgbcolor "light-magenta"      pt  8 ps 1
set style line 43 dt 4 lw 1 lc rgbcolor "light-magenta"      pt 10 ps 1
set style line 44 dt 4 lw 1 lc rgbcolor "light-magenta"      pt 12 ps 1
set style line 45 dt 4 lw 1 lc rgbcolor "light-magenta"      pt 14 ps 1

set style line 51 dt 4 lw 1 lc rgbcolor "goldenrod"          pt  6 ps 1
set style line 52 dt 4 lw 1 lc rgbcolor "goldenrod"          pt  8 ps 1
set style line 53 dt 4 lw 1 lc rgbcolor "goldenrod"          pt 10 ps 1
set style line 54 dt 4 lw 1 lc rgbcolor "goldenrod"          pt 12 ps 1
set style line 55 dt 4 lw 1 lc rgbcolor "goldenrod"          pt 14 ps 1

set style line 61 dt 4 lw 1 lc rgbcolor "sienna4"            pt  6 ps 1
set style line 62 dt 4 lw 1 lc rgbcolor "sienna4"            pt  8 ps 1
set style line 63 dt 4 lw 1 lc rgbcolor "sienna4"            pt 10 ps 1
set style line 64 dt 4 lw 1 lc rgbcolor "sienna4"            pt 12 ps 1
set style line 65 dt 4 lw 1 lc rgbcolor "sienna4"            pt 14 ps 1

set style line 71 dt 4 lw 1 lc rgbcolor "dark-violet"        pt  6 ps 1
set style line 72 dt 4 lw 1 lc rgbcolor "dark-violet"        pt  8 ps 1
set style line 73 dt 4 lw 1 lc rgbcolor "dark-violet"        pt 10 ps 1
set style line 74 dt 4 lw 1 lc rgbcolor "dark-violet"        pt 12 ps 1
set style line 75 dt 4 lw 1 lc rgbcolor "dark-violet"        pt 14 ps 1

set style line 81 dt 4 lw 1 lc rgbcolor "green"              pt  6 ps 1
set style line 82 dt 4 lw 1 lc rgbcolor "green"              pt  8 ps 1
set style line 83 dt 4 lw 1 lc rgbcolor "green"              pt 10 ps 1
set style line 84 dt 4 lw 1 lc rgbcolor "green"              pt 12 ps 1
set style line 85 dt 4 lw 1 lc rgbcolor "green"              pt 14 ps 1

set format x "%g"
set format y "%1.1e"

set key width 0

set grid  xtics lw 1 dt 4 lc rgbcolor "gray"
set grid  ytics lw 1 dt 4 lc rgbcolor "gray"
set grid mxtics lw 1 dt 4 lc rgbcolor "gray"
set grid mytics lw 1 dt 4 lc rgbcolor "gray"

set logscale x
set logscale y

set output "pc_cost.Dane.eps"
set title "2D uniform plasma, 512^2 grid, 32^2 ppc, 200 time steps" font "Times,14"
set xlabel "Number of MPI ranks" font "Times,14"
set ylabel "Wall time (seconds) per time step" font "Times,14"
set key outside right
plot \
'../wtimes.petsc_ksp.pc_none.toss_4_x86_64_ib.nx00512.np032.dat'                            u 2:($4/$3) w lp ls 11 t "No PC",\
'../wtimes.petsc_ksp.pc_ccmlmg_nvcyc002.toss_4_x86_64_ib.nx00512.np032.dat'                 u 2:($4/$3) w lp ls 32 t "PCCCMLMG (2)",\
'../wtimes.petsc_ksp.pc_ccmlmg_nvcyc004.toss_4_x86_64_ib.nx00512.np032.dat'                 u 2:($4/$3) w lp ls 33 t "PCCCMLMG (4)",\
'../wtimes.petsc_ksp.pc_petsc_lu.toss_4_x86_64_ib.nx00512.np032.dat'                        u 2:($4/$3) w lp ls 12 t "PCLU",\
'../wtimes.petsc_ksp.pc_petsc_asm_lu_asmovlp008.toss_4_x86_64_ib.nx00512.np032.dat'         u 2:($4/$3) w lp ls 22 t "PCASM ( 8)(LU)",\
'../wtimes.petsc_ksp.pc_petsc_asm_lu_asmovlp016.toss_4_x86_64_ib.nx00512.np032.dat'         u 2:($4/$3) w lp ls 23 t "PCASM (16)(LU)",\
'../wtimes.petsc_ksp.pc_petsc_asm_lu_asmovlp032.toss_4_x86_64_ib.nx00512.np032.dat'         u 2:($4/$3) w lp ls 25 t "PCASM (32)(LU)",\
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp008_ilu002.toss_4_x86_64_ib.nx00512.np032.dat' u 2:($4/$3) w lp ls 52 t "PCASM ( 8)(ILU 2)", \
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp008_ilu004.toss_4_x86_64_ib.nx00512.np032.dat' u 2:($4/$3) w lp ls 53 t "PCASM ( 8)(ILU 4)", \
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp008_ilu008.toss_4_x86_64_ib.nx00512.np032.dat' u 2:($4/$3) w lp ls 54 t "PCASM ( 8)(ILU 8)", \
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp016_ilu002.toss_4_x86_64_ib.nx00512.np032.dat' u 2:($4/$3) w lp ls 62 t "PCASM (16)(ILU 2)", \
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp016_ilu004.toss_4_x86_64_ib.nx00512.np032.dat' u 2:($4/$3) w lp ls 63 t "PCASM (16)(ILU 4)", \
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp016_ilu008.toss_4_x86_64_ib.nx00512.np032.dat' u 2:($4/$3) w lp ls 64 t "PCASM (16)(ILU 8)", \
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp032_ilu002.toss_4_x86_64_ib.nx00512.np032.dat' u 2:($4/$3) w lp ls 82 t "PCASM (32)(ILU 2)", \
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp032_ilu004.toss_4_x86_64_ib.nx00512.np032.dat' u 2:($4/$3) w lp ls 83 t "PCASM (32)(ILU 4)", \
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp032_ilu008.toss_4_x86_64_ib.nx00512.np032.dat' u 2:($4/$3) w lp ls 84 t "PCASM (32)(ILU 8)", \

