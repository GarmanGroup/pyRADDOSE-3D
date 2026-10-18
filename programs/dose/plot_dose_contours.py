"""Program for drawing the final dose as contours, as RADDOSE-3D's DoseState.R does."""

import sys

from pyraddose_3d.plotting import (
    CONTOUR_COLOURS,
    CONTOUR_LEVELS,
    CONTOUR_OPACITIES,
    VIEWS,
    DoseField,
    plot_dose_contours,
    save_dose_contours,
)

#######################################
### Editable parameters for program ###
#######################################

# The DoseState.csv run_dose.py writes: the dose in every voxel at the end of
# the run. Drawing it again with other levels needs no new run. A file named
# on the command line,
#     python plot_dose_contours.py output-DoseState.csv
# is drawn instead of this one.
DOSE_STATE_PATH = "output-DoseState.csv"

# A surface is drawn where the dose crosses each level, in MGy, in its colour
# and opacity (0 transparent, 1 opaque). RADDOSE-3D's DoseState.R draws 0.1,
# 20 and 30 MGy in light blue, dark blue and red; give a crystal that gets
# less, or more, levels of its own, such as (1.0, 5.0, 10.0).
LEVELS = CONTOUR_LEVELS
COLOURS = CONTOUR_COLOURS
OPACITIES = CONTOUR_OPACITIES

# A picture is saved from each view, as DoseState.R saves them: (theta, phi)
# in degrees, theta round the rotation axis from looking along the beam, and
# phi up from the horizontal. Each is written to
# <OUTPUT_PREFIX>DoseContours_<theta>_<phi>.png.
VIEWS_TO_SAVE = VIEWS
OUTPUT_PREFIX = "output-"

# Whether to open a window as well, in which the contours can be turned with
# the mouse.
SHOW_WINDOW = True

#############################################################
### Main function called to run the dose contour program ###
#############################################################


def main() -> None:
    """Main function for drawing the dose contours."""
    field = DoseField.read_dose_state(DOSE_STATE_PATH)
    print(f"Loaded {DOSE_STATE_PATH}: highest dose {field.dose.max():.2f} MGy")

    options = {"levels": LEVELS, "colours": COLOURS, "opacities": OPACITIES}
    for path in save_dose_contours(
        field, prefix=OUTPUT_PREFIX, views=VIEWS_TO_SAVE, **options
    ):
        print(f"Wrote {path}")

    if SHOW_WINDOW:
        import matplotlib.pyplot as plt

        plot_dose_contours(field, **options)
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise SystemExit(f"usage: python {sys.argv[0]} [DoseState.csv]")
    if len(sys.argv) == 2:
        DOSE_STATE_PATH = sys.argv[1]
    main()
