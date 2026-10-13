# pyraddose-3d-elsepa-high

ELSEPA's differential cross-sections for the elastic scattering of electrons from 21 to 300 keV, for hydrogen to tellurium (Z = 1 to 52): the tables from which RADDOSE-3D's Monte Carlo and XFEL subprograms draw the deflection of an electron above 20 keV. RADDOSE-3D has none for heavier elements, and refuses a run that would need one. pyRADDOSE-3D installs this package with its `montecarlo` extra.

The tables are RADDOSE-3D's `constants/above_20000/{Z}.csv` (https://github.com/GarmanGroup/RADDOSE-3D), each compressed with xz and otherwise unchanged. They are computed with ELSEPA (F. Salvat, A. Jablonski and C. J. Powell, *Comput. Phys. Commun.* 165, 157–190 (2005)).

At 23 MB compressed the tables are not committed. Build them from a RADDOSE-3D checkout:

    python tests/backend/data/build_elsepa_packages.py ~/git/RADDOSE-3D high

`tables/MANIFEST.tsv` is committed, and gives the RADDOSE-3D commit the tables were taken from and each file's SHA-256 before compression. pyRADDOSE-3D's tests check a build against it.

Like RADDOSE-3D, and pyRADDOSE-3D, this package is licensed under the GNU Affero General Public License version 3.
