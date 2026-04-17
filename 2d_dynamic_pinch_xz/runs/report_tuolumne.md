# WarpX Simulation Analysis Report - tuolumne

Analysis generated on: Thu Mar 26 17:22:19 PDT 2026

## Semi-Implicit (4 MPI ranks on Dane)
---------------

| PC     | Solver          | MMW | Linear/Step | Newton/Step | Linear/Newton | Walltime (s) |
|--------|-----------------|-----|-------------|-------------|---------------|--------------|
| None   | native_jfnk     | --  | 20          | 2           | 10            | 2.5          |
| None   | petsc_ksp       | --  | 21          | 2           | 10            | 2.7          |
| None   | weighted_jacobi | --  | 110         | 6           | 19            | 3.2          |
| None   | chebyshev       | --  | 108         | 6           | 18            | 3.6          |
| Jacobi | native_jfnk     | 0   | 22          | 2           | 10            | 2.4          |
| Jacobi | petsc_ksp       | 0   | 22          | 2           | 10            | 2.3          |
| Jacobi | native_jfnk     | 1   | 8           | 2           | 4             | 3.1          |
| Jacobi | petsc_ksp       | 1   | 9           | 2           | 4             | 3.7          |
| Jacobi | native_jfnk     | 2   | 4           | 2           | 2             | 2.8          |
| Jacobi | petsc_ksp       | 2   | 4           | 2           | 2             | 2.7          |
| Cheby  | native_jfnk     | 0   | 22          | 2           | 10            | 2.4          |
| Cheby  | petsc_ksp       | 0   | 22          | 2           | 10            | 2.3          |
| Cheby  | native_jfnk     | 1   | 8           | 2           | 4             | 3.8          |
| Cheby  | petsc_ksp       | 1   | 9           | 2           | 4             | 4.0          |
| Cheby  | native_jfnk     | 2   | 4           | 2           | 2             | 2.8          |
| Cheby  | petsc_ksp       | 2   | 4           | 2           | 2             | 2.9          |
| ASM-LU | petsc_ksp       | 0   | 22          | 2           | 10            | 2.2          |
| ASM-LU | petsc_ksp       | 1   | 8           | 2           | 4             | 2.2          |
| ASM-LU | petsc_ksp       | 2   | 4           | 2           | 2             | 2.6          |
| SOR    | petsc_ksp       | 0   | 22          | 2           | 10            | 2.2          |
| SOR    | petsc_ksp       | 1   | 13          | 2           | 6             | 1.9          |
| SOR    | petsc_ksp       | 2   | 12          | 2           | 6             | 2.1          |

## Theta-Implicit (4 MPI ranks on Dane)
----------------

| PC     | Solver          | MMW | Linear/Step | Newton/Step | Linear/Newton | Walltime (s) |
|--------|-----------------|-----|-------------|-------------|---------------|--------------|
| None   | native_jfnk     | --  | 21          | 2           | 10            | 2.5          |
| None   | petsc_ksp       | --  | 20          | 2           | 10            | 2.2          |
| None   | weighted_jacobi | --  | 110         | 6           | 19            | 3.6          |
| None   | chebyshev       | --  | 107         | 6           | 18            | 3.5          |
| Jacobi | native_jfnk     | 0   | 21          | 2           | 10            | 2.3          |
| Jacobi | petsc_ksp       | 0   | 21          | 2           | 10            | 2.3          |
| Jacobi | native_jfnk     | 1   | 8           | 2           | 4             | 3.5          |
| Jacobi | petsc_ksp       | 1   | 8           | 2           | 4             | 3.4          |
| Jacobi | native_jfnk     | 2   | 4           | 2           | 2             | 2.7          |
| Jacobi | petsc_ksp       | 2   | 4           | 2           | 2             | 2.6          |
| Cheby  | native_jfnk     | 0   | 22          | 2           | 10            | 2.4          |
| Cheby  | petsc_ksp       | 0   | 21          | 2           | 10            | 2.0          |
| Cheby  | native_jfnk     | 1   | 9           | 2           | 4             | 3.9          |
| Cheby  | petsc_ksp       | 1   | 10          | 2           | 4             | 4.0          |
| Cheby  | native_jfnk     | 2   | 4           | 2           | 2             | 2.6          |
| Cheby  | petsc_ksp       | 2   | 4           | 2           | 2             | 2.6          |
| ASM-LU | petsc_ksp       | 0   | 22          | 2           | 10            | 2.3          |
| ASM-LU | petsc_ksp       | 1   | 8           | 2           | 4             | 2.2          |
| ASM-LU | petsc_ksp       | 2   | 4           | 2           | 2             | 3.0          |

## Notes

- **PC**: Preconditioner (None, Jacobi, Cheby=Chebyshev, ASM-LU, LU, SOR)
- **Solver**: Nonlinear/linear solver (native_jfnk, petsc_ksp, weighted_jacobi, chebyshev)
- **MMW**: Mass matrix width (-- = not applicable)
- **Linear/Step**: Average linear iterations per timestep
- **Newton/Step**: Average Newton iterations per timestep
- **Linear/Newton**: Average linear iterations per Newton iteration
- **Walltime (s)**: Total execution time in seconds

