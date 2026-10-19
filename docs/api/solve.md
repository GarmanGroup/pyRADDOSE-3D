# Running

::: pyraddose_3d.solve.solve

::: pyraddose_3d.solve.simulate

::: pyraddose_3d.solve.Simulation

::: pyraddose_3d.solve.TUNABLE

::: pyraddose_3d.solve.FIXED

## MicroED

A crystal with `subprogram: microed` and an electron beam runs RADDOSE-3D's
MicroED engine instead, through its own entry point.

::: pyraddose_3d.solve.microed

::: pyraddose_3d.backend.microed.MicroEDResult

## Monte Carlo, GOS and XFEL

A crystal with `subprogram: montecarlo`, `gos` or `xfel` runs RADDOSE-3D's
Monte Carlo, photon by photon, through its own entry point: on NumPy, or with
PyTorch on a device such as a GPU.

::: pyraddose_3d.solve.monte_carlo

::: pyraddose_3d.backend.montecarlo.simple.SimpleRun
