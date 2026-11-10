# 3d_collisional_boxes

Collisional relaxation boxes for LEANPIC (Learned Emulators for Accelerating
Non-equilibrium PIC), step 0 of `~/Codes/LEANPIC/docs/feasibility.md` (§8.1).

Every cell is a closed box: deuterium particles are not pushed, there is no field solve,
and only deuterium-deuterium Coulomb collisions act. Cells differ in their initial
temperature anisotropy (x and y cell indices); the cells along z are replicas.
`particle_fields` writes the raw velocity moments up to order 4 of two pseudo-random
halves of each cell's particles.

- `scripts/writeBoxInputs.py`: writes the input decks (design in its docstring).
- `inputs/boxes_test_3d.in`: test deck, 8 x 8 x 8 cells, 10^4 particles per cell.
