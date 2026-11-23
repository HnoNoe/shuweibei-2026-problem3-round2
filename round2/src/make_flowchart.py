import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

BLUE = "#386795"
GREEN = "#4E9A6F"
AMBER = "#C9962E"
GREY = "#8C8C8C"
INK = "0.13"


def node(ax, cx, cy, w, h, text, color, fs=10.5, bold_first=False):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=0.9",
                                fc="white", ec=color, lw=1.3, zorder=2))
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), 1.6, h,
                                boxstyle="round,pad=0,rounding_size=0.5",
                                fc=color, ec=color, lw=0, zorder=3))
    ax.text(cx + 0.6, cy, text, ha="center", va="center", fontsize=fs,
            color=INK, zorder=4, linespacing=1.45)


def arrow(ax, x0, y0, x1, y1, color="0.45", elbow=None):
    style = "-|>"
    kw = dict(arrowstyle=style, color=color, lw=1.2, shrinkA=0, shrinkB=0,
              mutation_scale=11)
    if elbow:
        kw["connectionstyle"] = elbow
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=kw, zorder=1)


def stage_label(ax, x, y, text, color):
    ax.text(x, y, text, fontsize=8, color=color, fontweight="bold",
            ha="left", va="center")
    ax.plot([x, x + 5.5], [y - 1.8, y - 1.8], color=color, lw=0.8)


def fig_workflow():
    W, H = 100, 102
    fig, ax = plt.subplots(figsize=(6.86, 7.01))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    cx = 52

    stage_label(ax, 2, 96, "DATA", BLUE)
    node(ax, cx, 90.5, 62, 8, "Prepare the official data split", BLUE)

    stage_label(ax, 2, 79, "MODEL", GREEN)
    node(ax, cx, 72.5, 78, 11,
         "Estimate family duration distributions\n"
         "Compare ridge prefix models against the family baseline",
         GREEN, fs=10)

    stage_label(ax, 2, 57.5, "DECISION", AMBER)
    node(ax, cx, 51.5, 80, 11,
         "Compare dispatch with free prefixes and paid reviews\n"
         "using matched validation replays", AMBER, fs=10)

    node(ax, 27, 33.5, 42, 9, "Free-prefix comparator", GREEN, fs=10)
    node(ax, 77, 33.5, 42, 9, "Paid-review comparator", GREEN, fs=10)

    node(ax, cx, 16.5, 82, 11,
         "Retain travel-aware ATC with zero paid reviews\n"
         "for the primary policy under the tested conditions",
         BLUE, fs=10)

    stage_label(ax, 2, 4.5, "VALIDATION", GREY)
    node(ax, cx, 4.0, 70, 7, "Evaluate dispatch and reservation sensitivity",
         GREY, fs=10)

    arrow(ax, cx, 86.4, cx, 78.2)
    arrow(ax, cx, 67.9, cx, 57.2)
    arrow(ax, cx - 10, 45.9, 27, 38.2, elbow="angle,angleA=-90,angleB=0")
    arrow(ax, cx + 10, 45.9, 77, 38.2, elbow="angle,angleA=-90,angleB=180")
    arrow(ax, 27, 28.9, cx - 12, 22.2, elbow="angle,angleA=-90,angleB=180")
    arrow(ax, 77, 28.9, cx + 12, 22.2, elbow="angle,angleA=-90,angleB=0")
    arrow(ax, cx, 10.9, cx, 7.6)

    fig.savefig("round2/figures3/fig1_workflow.png", dpi=200)
    plt.close(fig)


def fig_route():
    W, H = 100, 125.2
    fig, ax = plt.subplots(figsize=(6.38, 8.0))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    cx = 50

    node(ax, cx, 119.5, 22, 7, "Start", "0.25", fs=10.5)
    node(ax, cx, 104.5, 52, 10,
         "1. Data Preprocessing\n(Train/Val/Test split)", BLUE, fs=10)
    node(ax, cx, 87.5, 56, 10,
         "2. Duration Analysis\n(Ridge/RF prefix prediction)", GREEN, fs=10)

    dy = 71
    d = 9.5
    ax.add_patch(Polygon([(cx - 26, dy), (cx, dy + d), (cx + 26, dy),
                          (cx, dy - d)], closed=True, fc="#FDF6E7",
                         ec=AMBER, lw=1.3, zorder=2))
    ax.text(cx, dy, "Information value ≈ 0?", ha="center", va="center",
            fontsize=10, color=INK, zorder=3)

    node(ax, 20, 47, 32, 9.5, "No (ignore)", GREEN, fs=9.5)
    node(ax, 80, 47, 32, 9.5, "Yes (buy reviews)", GREY, fs=9.5)

    node(ax, cx, 26.5, 58, 10,
         "3. Dispatch Rule\n(Travel-aware ATC index)", BLUE, fs=10)
    node(ax, cx, 8.5, 56, 10,
         "4. Simulation & Validation", GREY, fs=10)

    arrow(ax, cx, 116, cx, 109.7)
    arrow(ax, cx, 99.4, cx, 92.7)
    arrow(ax, cx, 82.4, cx, 80.8)
    arrow(ax, cx - 19, 63, 22, 52, elbow="angle,angleA=-90,angleB=0")
    arrow(ax, cx + 19, 63, 78, 52, elbow="angle,angleA=-90,angleB=180")
    arrow(ax, 22, 42.2, cx - 13, 31.7, elbow="angle,angleA=-90,angleB=180")
    arrow(ax, 78, 42.2, cx + 13, 31.7, elbow="angle,angleA=-90,angleB=0")
    arrow(ax, cx, 21.4, cx, 13.7)

    fig.savefig("round2/figures3/fig0_route.png", dpi=200)
    plt.close(fig)


fig_workflow()
fig_route()
print("done")
