# pyraddose-3d-elsepa-low

ELSEPA's differential cross-sections for the elastic scattering of electrons up to 20 keV, for every element from hydrogen to plutonium: the tables from which RADDOSE-3D's Monte Carlo and XFEL subprograms draw each electron's deflection. pyRADDOSE-3D installs this package with its `montecarlo` extra, which also installs `pyraddose-3d-elsepa-high` for electrons above 20 keV.

The tables are RADDOSE-3D's `constants/below_20000/{Z}.csv` (https://github.com/GarmanGroup/RADDOSE-3D), each compressed with xz and otherwise unchanged. `tables/MANIFEST.tsv` gives the RADDOSE-3D commit they were taken from and each file's SHA-256 before compression. They are computed with ELSEPA (F. Salvat, A. Jablonski and C. J. Powell, *Comput. Phys. Commun.* 165, 157–190 (2005)).

Like RADDOSE-3D, and pyRADDOSE-3D, this package is licensed under the GNU Affero General Public License version 3.
