"""Shared matplotlib styling so every figure in the assignment looks consistent."""
import matplotlib as mpl
import matplotlib.pyplot as plt

# Colours (categorical order is fixed: blue, orange, aqua)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#9a9892"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

# Colour roles, identical in Part 1 and Part 2. Every figure uses these names,
# so a colour always means the same thing.
DATA = INK                # given points (Part 1) and data points (Part 2)
CURVE = INK2              # a known curve y = f(x)
NEWTON = BLUE             # Newton-Raphson (Part 1) and coordinate Newton (Part 2)
NEWTON_LIGHT = "#9cc3ee"  # a secondary Newton quantity, e.g. the step size
OTHER = AQUA              # the second method: golden section (Part 1), full multivariate Newton (Part 2)
OTHER_LIGHT = "#bfe8d6"   # the interval kept by golden section
SOLUTION = ORANGE         # the answer: x*, the minimum, the best fit
GUIDE = MUTED             # tolerance lines, residuals, contours, leader lines
ITERATION_RAMP = ["#c9ddf5", "#9cc3ee", "#6ea6e6", "#3f89dc", "#1d5aa6", "#113a70"]  # early -> late iterates

# Marker roles
DATA_MARKER = "o"
NEWTON_MARKER = "s"
OTHER_MARKER = "^"
SOLUTION_MARKER = "*"

# Shared sizes
SINGLE_FIGSIZE = (9, 6)   # every single-panel figure
POINT_SIZE = 50           # data points and method markers
STAR_SIZE = 140           # the solution star


def apply_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.labelcolor": INK2,
        "axes.edgecolor": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 2,
        "mathtext.fontset": "dejavusans",
    })


def plot_points(ax, x, y, role, label=None, zorder=5, **kwargs):
    """Scatter points in one of the roles above: 'data', 'newton', 'other' or 'solution'."""
    colour, marker, size = {
        "data": (DATA, DATA_MARKER, POINT_SIZE),
        "newton": (NEWTON, NEWTON_MARKER, POINT_SIZE),
        "other": (OTHER, OTHER_MARKER, POINT_SIZE),
        "solution": (SOLUTION, SOLUTION_MARKER, STAR_SIZE),
    }[role]
    ax.scatter(x, y, color=colour, marker=marker, s=size, edgecolor="white", linewidth=1,
               zorder=zorder, label=label, **kwargs)


def note_box(ax, text, x, y, ha="left", va="top"):
    """Boxed note (equations, update rules) placed in axes coordinates."""
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": MUTED})


def figure_title(fig, text):
    """
    Overall title for a multi-panel figure, left-aligned like the panel titles.
    Also lays out the panels, leaving the same 0.2 inch gap below the title at any figure height.
    """
    fig.suptitle(text, x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.2 / fig.get_figheight()))


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
