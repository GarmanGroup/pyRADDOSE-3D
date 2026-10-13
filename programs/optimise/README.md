# Energy optimisation program example script and config

This directory contains an example script for choosing the beam energy that maximises diffraction efficiency -- elastically scattered photons per MGy of diffraction-weighted dose -- and an example configuration.

## Files

- `run_optimise_energy.py` - The Python script. Edit the parameters at the top: your experiment, the energies your beamline can reach, and your detector's sensor.
- `optimise_energy_example_config.yaml` - An example experiment: an insulin crystal, whose zinc puts an absorption edge at 9.659 keV.

Gradients need PyTorch: `pip install pyRADDOSE-3D[torch]`.

## Running

```bash
python run_optimise_energy.py
```

It prints the best energy in each stretch between absorption edges, the best overall, and the diffraction efficiency there; and it writes the experiment at the best energy to `optimised_config.yaml`, which `../dose/run_dose.py` can run to get RADDOSE-3D's output files.

## How it searches

Absorption jumps at each element's absorption edges, and diffraction efficiency jumps with it -- for a selenomethionine crystal, by up to a third at the Se K edge. A gradient cannot see a jump, so the range is split at every edge of every element in the crystal, its container and the detector's sensor, and each piece is searched on its own. Within a piece the sign of the gradient says which side the best energy lies, so the search halves the piece with each run: about a dozen runs to locate it to 0.01 keV, a few seconds for the example.

Efficiency often peaks right below an edge. The model's edges are sharp steps; real ones have fine structure for tens of eV around them, and a beam has an energy spread, so the search keeps `EDGE_CLEARANCE` (50 eV) away from each.

## What it leaves out

- **Anomalous signal.** Diffraction efficiency counts elastically scattered photons, not the anomalous differences a SAD or MAD experiment needs. For those the energy is set by the edge, whatever the efficiency.
- **Photoelectron escape.** From a crystal of a few micrometres, a photoelectron can carry its energy out before depositing it, more so at higher energies; leaving that out understates what high energies offer small crystals. Escape is modelled, as RADDOSE-3D models it, but by Monte Carlo, which has no gradient -- a configuration with escape on is refused here. Run the chosen energy through `programs/dose` with `escape: {photoelectron: true}` to see how much it changes.
- **The detector's real efficiency.** The sensor model is the fraction of photons absorbed in it, an upper bound that ignores charge sharing, fluorescence escape, counting thresholds and dead area.
- **The beamline's flux.** It does not matter: with no decay model, flux and exposure time scale the elastic yield and the dose together, so the efficiency is the same whatever the flux at each energy.
