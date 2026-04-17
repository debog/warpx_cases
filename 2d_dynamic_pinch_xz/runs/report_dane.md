# WarpX Simulation Analysis Report - dane

Analysis generated on: Thu Mar 26 17:02:33 PDT 2026

## Semi-Implicit (4 MPI ranks on Dane)
---------------

| PC     | Solver          | MMW | Linear/Step | Newton/Step | Linear/Newton | Walltime (s) |
|--------|-----------------|-----|-------------|-------------|---------------|--------------|
| None   | native_jfnk     | --  | 20          | 2           | 10            | 16.9         |
| None   | petsc_ksp       | --  | 20          | 2           | 10            | 16.6         |
| None   | weighted_jacobi | --  | 110         | 6           | 19            | 27.4         |
| None   | chebyshev       | --  | 105         | 6           | 18            | 27.4         |
| Jacobi | native_jfnk     | 0   | 20          | 2           | 10            | 17.4         |
| Jacobi | petsc_ksp       | 0   | 20          | 2           | 10            | 18.7         |
| Jacobi | native_jfnk     | 1   | 8           | 2           | 4             | 16.4         |
| Jacobi | petsc_ksp       | 1   | 8           | 2           | 4             | 16.0         |
| Jacobi | native_jfnk     | 2   | 4           | 2           | 2             | 17.4         |
| Jacobi | petsc_ksp       | 2   | 4           | 2           | 2             | 15.6         |
| Cheby  | native_jfnk     | 0   | 20          | 2           | 10            | 15.8         |
| Cheby  | petsc_ksp       | 0   | 20          | 2           | 10            | 15.6         |
| Cheby  | native_jfnk     | 1   | 8           | 2           | 4             | 16.8         |
| Cheby  | petsc_ksp       | 1   | 8           | 2           | 4             | 16.5         |
| Cheby  | native_jfnk     | 2   | 4           | 2           | 2             | 15.8         |
| Cheby  | petsc_ksp       | 2   | 4           | 2           | 2             | 18.4         |
| ASM-LU | petsc_ksp       | 0   | 20          | 2           | 10            | 18.8         |
| ASM-LU | petsc_ksp       | 1   | 8           | 2           | 4             | 16.6         |
| ASM-LU | petsc_ksp       | 2   | 4           | 2           | 2             | 17.3         |
| LU     | petsc_ksp       | 0   | 20          | 2           | 10            | 17.8         |
| LU     | petsc_ksp       | 1   | 8           | 2           | 4             | 15.9         |
| LU     | petsc_ksp       | 2   | 4           | 2           | 2             | 18.7         |
| SOR    | petsc_ksp       | 0   | 20          | 2           | 10            | 17.4         |
| SOR    | petsc_ksp       | 1   | 12          | 2           | 6             | 16.8         |
| SOR    | petsc_ksp       | 2   | 12          | 2           | 6             | 16.2         |

## Theta-Implicit (4 MPI ranks on Dane)
----------------

| PC     | Solver          | MMW | Linear/Step | Newton/Step | Linear/Newton | Walltime (s) |
|--------|-----------------|-----|-------------|-------------|---------------|--------------|
| None   | native_jfnk     | --  | 20          | 2           | 10            | 17.6         |
| None   | petsc_ksp       | --  | 20          | 2           | 10            | 17.6         |
| None   | weighted_jacobi | --  | 110         | 6           | 19            | 27.9         |
| None   | chebyshev       | --  | 105         | 6           | 18            | 28.5         |
| Jacobi | native_jfnk     | 0   | 20          | 2           | 10            | 18.4         |
| Jacobi | petsc_ksp       | 0   | 20          | 2           | 10            | 15.5         |
| Jacobi | native_jfnk     | 1   | 8           | 2           | 4             | 19.9         |
| Jacobi | petsc_ksp       | 1   | 8           | 2           | 4             | 16.4         |
| Jacobi | native_jfnk     | 2   | 4           | 2           | 2             | 18.1         |
| Jacobi | petsc_ksp       | 2   | 4           | 2           | 2             | 16.5         |
| Cheby  | native_jfnk     | 0   | 20          | 2           | 10            | 17.3         |
| Cheby  | petsc_ksp       | 0   | 20          | 2           | 10            | 17.7         |
| Cheby  | native_jfnk     | 1   | 8           | 2           | 4             | 18.4         |
| Cheby  | petsc_ksp       | 1   | 8           | 2           | 4             | 16.2         |
| Cheby  | native_jfnk     | 2   | 4           | 2           | 2             | 17.3         |
| Cheby  | petsc_ksp       | 2   | 4           | 2           | 2             | 15.9         |
| ASM-LU | petsc_ksp       | 0   | 20          | 2           | 10            | 20.7         |
| ASM-LU | petsc_ksp       | 1   | 8           | 2           | 4             | 18.5         |
| ASM-LU | petsc_ksp       | 2   | 4           | 2           | 2             | 16.2         |
| LU     | petsc_ksp       | 0   | 20          | 2           | 10            | 16.7         |
| LU     | petsc_ksp       | 1   | 8           | 2           | 4             | 16.2         |
| LU     | petsc_ksp       | 2   | 4           | 2           | 2             | 21.7         |

## Notes

- **PC**: Preconditioner (None, Jacobi, Cheby=Chebyshev, ASM-LU, LU, SOR)
- **Solver**: Nonlinear/linear solver (native_jfnk, petsc_ksp, weighted_jacobi, chebyshev)
- **MMW**: Mass matrix width (-- = not applicable)
- **Linear/Step**: Average linear iterations per timestep
- **Newton/Step**: Average Newton iterations per timestep
- **Linear/Newton**: Average linear iterations per Newton iteration
- **Walltime (s)**: Total execution time in seconds

