"""Program for choosing the beam energy that maximises diffraction efficiency."""

import time

from pyraddose_3d.backend.detector import sensor_absorption, sensor_edges
from pyraddose_3d.models.experiment import Experiment
from pyraddose_3d.optimise import diffraction_efficiency, maximise_energy
from pyraddose_3d.solve import apply_overrides, solve

#######################################
### Editable parameters for program ###
#######################################

# The experiment: a YAML configuration like optimise_energy_example_config.yaml,
# or a RADDOSE-3D input file. Everything but the beam energy is kept as given.
CONFIG_PATH = "/path/to/optimise_energy_example_config.yaml"

# The energies the beamline can reach, in keV.
ENERGY_RANGE = (6.0, 25.0)

# The detector's sensor, as (material, thickness in micrometres), where the
# material is "Si", "CdTe" or "GaAs". Diffraction efficiency then counts the
# photons the sensor absorbs rather than all those the crystal scatters --
# which changes the answer, since a thin silicon sensor lets most hard photons
# through. None optimises what the crystal scatters.
SENSOR = ("Si", 450.0)

# How closely to locate the best energy, in keV.
TOLERANCE = 0.01

# How far to keep from each absorption edge, in keV. Efficiency often peaks
# right below an edge, where the model's absorption is a sharp step; a real
# edge has fine structure for tens of eV around it, and the beam has an
# energy spread. 50 eV is conservative.
EDGE_CLEARANCE = 0.05

# Where to write the experiment at the best energy. run_dose.py reads it.
OPTIMISED_CONFIG_PATH = "optimised_config.yaml"

#####################################################
### Main function called to run the optimisation ###
#####################################################


def main() -> None:
    """Main function for running the energy optimisation."""
    experiment = Experiment.from_file(CONFIG_PATH)
    low, high = ENERGY_RANGE

    def detected(summaries, energy):
        efficiency = diffraction_efficiency(summaries)
        if SENSOR is None:
            return efficiency
        return efficiency * sensor_absorption(energy, *SENSOR)

    edges = () if SENSOR is None else sensor_edges(SENSOR[0])
    start_time = time.time()
    searches = maximise_energy(
        experiment,
        detected,
        low,
        high,
        edges=edges,
        tolerance=TOLERANCE,
        clearance=EDGE_CLEARANCE,
    )
    elapsed = time.time() - start_time

    print(
        f"Searched {low}-{high} keV, one piece between each pair of edges, "
        f"keeping {EDGE_CLEARANCE * 1000:.0f} eV clear of each:"
    )
    for search in searches:
        energy, value = search.best
        print(
            f"  {search.low:7.3f}-{search.high:7.3f} keV: best {energy:7.3f} keV, "
            f"{value:.4g} photons/MGy ({search.evaluations} runs)"
        )
    best = max(searches, key=lambda search: search.best[1])
    energy = best.best[0]
    runs = sum(search.evaluations for search in searches)
    print(f"{runs} runs with gradients, {elapsed:.0f} s")

    # Re-run at the best energy without gradients: NumPy is the backend the
    # agreement with RADDOSE-3D is established for.
    optimised = apply_overrides(experiment, {"beam.energy": energy})
    efficiency = float(diffraction_efficiency(solve(optimised)))
    print(f"\nBest energy: {energy:.3f} keV")
    print(f"  diffraction efficiency      {efficiency:.4g} photons/MGy")
    if SENSOR is not None:
        absorbed = float(sensor_absorption(energy, *SENSOR))
        print(f"  absorbed by the sensor      {absorbed:.1%}")
        print(f"  detected                    {efficiency * absorbed:.4g} photons/MGy")

    optimised.to_yaml(OPTIMISED_CONFIG_PATH)
    print(f"\nWrote {OPTIMISED_CONFIG_PATH}")


if __name__ == "__main__":
    main()
