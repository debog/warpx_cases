set terminal postscript enhanced eps color "Times" 8

set style line 11 dt 2 lw 2 lc rgbcolor "black"              pt  5 ps 2
set style line 12 dt 4 lw 1 lc rgbcolor "blue"               pt  5 ps 2

set style line 21 dt 4 lw 1 lc rgbcolor "dark-green"         pt  6 ps 1.5
set style line 22 dt 4 lw 1 lc rgbcolor "dark-green"         pt  8 ps 1.5
set style line 23 dt 4 lw 2 lc rgbcolor "dark-green"         pt 10 ps 1.5
set style line 24 dt 4 lw 2 lc rgbcolor "dark-green"         pt 12 ps 1.5
set style line 25 dt 4 lw 2 lc rgbcolor "dark-green"         pt 14 ps 1.5

set style line 31 dt 4 lw 1 lc rgbcolor "red"                pt  6 ps 1.5
set style line 32 dt 4 lw 2 lc rgbcolor "red"                pt  8 ps 1.5
set style line 33 dt 4 lw 2 lc rgbcolor "red"                pt 10 ps 1.5
set style line 34 dt 4 lw 2 lc rgbcolor "red"                pt 12 ps 1.5

set style line 41 dt 4 lw 1 lc rgbcolor "light-magenta"      pt  6 ps 1.5
set style line 42 dt 4 lw 1 lc rgbcolor "light-magenta"      pt  8 ps 1.5
set style line 43 dt 4 lw 2 lc rgbcolor "light-magenta"      pt 10 ps 1.5
set style line 44 dt 4 lw 2 lc rgbcolor "light-magenta"      pt 12 ps 1.5
set style line 45 dt 4 lw 2 lc rgbcolor "light-magenta"      pt 14 ps 1.5

set style line 51 dt 4 lw 1 lc rgbcolor "goldenrod"          pt  6 ps 1.5
set style line 52 dt 4 lw 1 lc rgbcolor "goldenrod"          pt  8 ps 1.5
set style line 53 dt 4 lw 2 lc rgbcolor "goldenrod"          pt 10 ps 1.5
set style line 54 dt 4 lw 2 lc rgbcolor "goldenrod"          pt 12 ps 1.5
set style line 55 dt 4 lw 2 lc rgbcolor "goldenrod"          pt 14 ps 1.5

set style line 61 dt 4 lw 1 lc rgbcolor "sienna4"            pt  6 ps 1.5
set style line 62 dt 4 lw 1 lc rgbcolor "sienna4"            pt  8 ps 1.5
set style line 63 dt 4 lw 2 lc rgbcolor "sienna4"            pt 10 ps 1.5
set style line 64 dt 4 lw 2 lc rgbcolor "sienna4"            pt 12 ps 1.5
set style line 65 dt 4 lw 2 lc rgbcolor "sienna4"            pt 14 ps 1.5

set style line 71 dt 4 lw 1 lc rgbcolor "dark-violet"        pt  6 ps 1.5
set style line 72 dt 4 lw 1 lc rgbcolor "dark-violet"        pt  8 ps 1.5
set style line 73 dt 4 lw 2 lc rgbcolor "dark-violet"        pt 10 ps 1.5
set style line 74 dt 4 lw 2 lc rgbcolor "dark-violet"        pt 12 ps 1.5
set style line 75 dt 4 lw 2 lc rgbcolor "dark-violet"        pt 14 ps 1.5

set style line 81 dt 4 lw 1 lc rgbcolor "green"              pt  6 ps 1.5
set style line 82 dt 4 lw 1 lc rgbcolor "green"              pt  8 ps 1.5
set style line 83 dt 4 lw 2 lc rgbcolor "green"              pt 10 ps 1.5
set style line 84 dt 4 lw 2 lc rgbcolor "green"              pt 12 ps 1.5
set style line 85 dt 4 lw 2 lc rgbcolor "green"              pt 14 ps 1.5

set format x "%1.1f"
set format y "%g"

set key width 0

set grid  xtics lw 1 dt 4 lc rgbcolor "gray"
set grid  ytics lw 1 dt 4 lc rgbcolor "gray"
set grid mxtics lw 1 dt 4 lc rgbcolor "gray"
set grid mytics lw 1 dt 4 lc rgbcolor "gray"

set size square

set output "pc_performance.nx00512.np032.ngpu004.Perlmutter.eps"
set title "2D uniform plasma, 512^2 grid, 32^2 ppc, 100 time steps, 4 GPUs" font "Times,14"
set xlabel "Wall time (s)" font "Times,14"
set ylabel "Average number of GMRES iterations per Newton iteration" font "Times,14"
set key outside right
plot \
'../wtimes.petsc_ksp.pc_none.perlmutter.nx00512.np032.dat'                            u 4:($11/($13)) w p ls 11 t "No PC",\
'../wtimes.petsc_ksp.pc_ccmlmg_nvcyc001.perlmutter.nx00512.np032.dat'                 u 4:($11/($13)) w p ls 31 t "PCCCMLMG (1)",\
'../wtimes.petsc_ksp.pc_ccmlmg_nvcyc002.perlmutter.nx00512.np032.dat'                 u 4:($11/($13)) w p ls 32 t "PCCCMLMG (2)",\
'../wtimes.petsc_ksp.pc_ccmlmg_nvcyc004.perlmutter.nx00512.np032.dat'                 u 4:($11/($13)) w p ls 33 t "PCCCMLMG (4)",\
'../wtimes.petsc_ksp.pc_ccmlmg_nvcyc008.perlmutter.nx00512.np032.dat'                 u 4:($11/($13)) w p ls 34 t "PCCCMLMG (8)",\
'../wtimes.petsc_ksp.pc_petsc_asm_lu_asmovlp000.perlmutter.nx00512.np032.dat'         u 4:($11/($13)) w p ls 21 t "PCASM ( 0)(LU)",\
'../wtimes.petsc_ksp.pc_petsc_asm_lu_asmovlp008.perlmutter.nx00512.np032.dat'         u 4:($11/($13)) w p ls 22 t "PCASM ( 8)(LU)",\
'../wtimes.petsc_ksp.pc_petsc_asm_lu_asmovlp016.perlmutter.nx00512.np032.dat'         u 4:($11/($13)) w p ls 23 t "PCASM (16)(LU)",\
'../wtimes.petsc_ksp.pc_petsc_asm_lu_asmovlp024.perlmutter.nx00512.np032.dat'         u 4:($11/($13)) w p ls 24 t "PCASM (24)(LU)",\
'../wtimes.petsc_ksp.pc_petsc_asm_lu_asmovlp032.perlmutter.nx00512.np032.dat'         u 4:($11/($13)) w p ls 25 t "PCASM (32)(LU)",\
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp000_ilu001.perlmutter.nx00512.np032.dat' u 4:($11/($13)) w p ls 41 t "PCASM ( 0)(ILU 1)", \
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp000_ilu002.perlmutter.nx00512.np032.dat' u 4:($11/($13)) w p ls 42 t "PCASM ( 0)(ILU 2)", \
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp000_ilu004.perlmutter.nx00512.np032.dat' u 4:($11/($13)) w p ls 43 t "PCASM ( 0)(ILU 4)", \
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp000_ilu008.perlmutter.nx00512.np032.dat' u 4:($11/($13)) w p ls 44 t "PCASM ( 0)(ILU 8)", \
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp000_ilu016.perlmutter.nx00512.np032.dat' u 4:($11/($13)) w p ls 45 t "PCASM ( 0)(ILU16)", \
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp008_ilu001.perlmutter.nx00512.np032.dat' u 4:($11/($13)) w p ls 51 t "PCASM ( 8)(ILU 1)", \
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp008_ilu002.perlmutter.nx00512.np032.dat' u 4:($11/($13)) w p ls 52 t "PCASM ( 8)(ILU 2)", \
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp008_ilu004.perlmutter.nx00512.np032.dat' u 4:($11/($13)) w p ls 53 t "PCASM ( 8)(ILU 4)", \
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp008_ilu008.perlmutter.nx00512.np032.dat' u 4:($11/($13)) w p ls 54 t "PCASM ( 8)(ILU 8)", \
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp008_ilu016.perlmutter.nx00512.np032.dat' u 4:($11/($13)) w p ls 55 t "PCASM ( 8)(ILU16)", \
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp016_ilu001.perlmutter.nx00512.np032.dat' u 4:($11/($13)) w p ls 61 t "PCASM (16)(ILU 1)", \
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp016_ilu002.perlmutter.nx00512.np032.dat' u 4:($11/($13)) w p ls 62 t "PCASM (16)(ILU 2)", \
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp016_ilu004.perlmutter.nx00512.np032.dat' u 4:($11/($13)) w p ls 63 t "PCASM (16)(ILU 4)", \
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp016_ilu008.perlmutter.nx00512.np032.dat' u 4:($11/($13)) w p ls 64 t "PCASM (16)(ILU 8)", \
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp016_ilu016.perlmutter.nx00512.np032.dat' u 4:($11/($13)) w p ls 65 t "PCASM (16)(ILU16)", \
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp024_ilu001.perlmutter.nx00512.np032.dat' u 4:($11/($13)) w p ls 71 t "PCASM (24)(ILU 1)", \
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp024_ilu002.perlmutter.nx00512.np032.dat' u 4:($11/($13)) w p ls 72 t "PCASM (24)(ILU 2)", \
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp024_ilu004.perlmutter.nx00512.np032.dat' u 4:($11/($13)) w p ls 73 t "PCASM (24)(ILU 4)", \
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp024_ilu008.perlmutter.nx00512.np032.dat' u 4:($11/($13)) w p ls 74 t "PCASM (24)(ILU 8)", \
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp024_ilu016.perlmutter.nx00512.np032.dat' u 4:($11/($13)) w p ls 75 t "PCASM (24)(ILU16)", \
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp032_ilu001.perlmutter.nx00512.np032.dat' u 4:($11/($13)) w p ls 81 t "PCASM (32)(ILU 1)", \
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp032_ilu002.perlmutter.nx00512.np032.dat' u 4:($11/($13)) w p ls 82 t "PCASM (32)(ILU 2)", \
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp032_ilu004.perlmutter.nx00512.np032.dat' u 4:($11/($13)) w p ls 83 t "PCASM (32)(ILU 4)", \
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp032_ilu008.perlmutter.nx00512.np032.dat' u 4:($11/($13)) w p ls 84 t "PCASM (32)(ILU 8)", \
'../wtimes.petsc_ksp.pc_petsc_asm_ilu_asmovlp032_ilu016.perlmutter.nx00512.np032.dat' u 4:($11/($13)) w p ls 85 t "PCASM (32)(ILU16)", \

