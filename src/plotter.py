import matplotlib.pyplot as plt
import numpy as np

from src.data_utils import DataUtils

plt.rcParams.update(
    {
        "axes.linewidth": 1.5,
        "axes.labelsize": "xx-large",
        "axes.titlesize": "xx-large",
        "xtick.labelsize": "xx-large",
        "ytick.labelsize": "xx-large",
    }
)


class Plotter(DataUtils):
    def __init__(self, data_utils):
        if not isinstance(data_utils, DataUtils):
            raise TypeError("data_utils must be an instance of DataUtils")
        self.__dict__.update(data_utils.__dict__)

    @staticmethod
    def _smart_fmt(x, pos):
        if abs(x) < 1:
            s = f"{x:.2f}"
        else:
            s = f"{x:.1f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s

    def get_mode_frequencies(self, modes, evals):
        freqs = np.abs(np.angle(evals)) / (2 * np.pi * self.dt)
        mags = np.array([np.linalg.norm(modes[:, i]) * np.abs(evals[i]) for i in range(len(freqs))])
        mags /= np.max(mags)
        return freqs, mags

    def get_top_r_modes(self, modes, evals):
        b = np.linalg.pinv(modes) @ self.X[:, 0]  # x1 = modes @ b
        idx = np.argsort(np.abs(b))[::-1][: self.cfg.streaming_rank]
        return modes[:, idx], evals[idx]

    def plot_mode_frequencies(self):
        # select top r dmd modes
        dmd_modes_r, dmd_evals_r = self.get_top_r_modes(self.dmd_modes, self.dmd_evals)

        # get DMD, SDMD, and ESDMD frequencies and magnitudes
        freqs_mags = dict(
            dmd=self.get_mode_frequencies(dmd_modes_r, dmd_evals_r),
            sdmd=self.get_mode_frequencies(self.sdmd_modes[-1], self.sdmd_evals[:, -1]),
            esdmd=self.get_mode_frequencies(self.esdmd_modes[-1], self.esdmd_evals[:, -1]),
        )

        # styles
        stemlw = dict(dmd=1.5, sdmd=2, esdmd=2)
        msizes = dict(dmd=20, sdmd=20, esdmd=20)
        edges = dict(dmd=1, sdmd=1, esdmd=1.5)
        mstyles = {
            alg: dict(
                marker=self.cfg.markerstyle[alg],
                markeredgecolor=self.cfg.markercolor[alg],
                markersize=msizes[alg],
                markerfacecolor=self.cfg.markerface[alg],
                markeredgewidth=edges[alg],
                alpha=self.cfg.markeralpha[alg],
            )
            for alg in freqs_mags.keys()
        }
        sstyles = {
            alg: dict(
                linewidth=stemlw[alg],
                color=self.cfg.markercolor[alg],
                alpha=self.cfg.markeralpha[alg],
            )
            for alg in freqs_mags.keys()
        }

        # margins and ticks
        freqs, mags = freqs_mags["dmd"]
        fmin, fmax = freqs.min(), freqs.max()
        fmargin = (fmax - fmin) * 0.1
        xticks = np.linspace(fmin, fmax, 4)
        yticks = np.array([1e-2, 0.5, 1.0])

        # set up plot
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.set_yscale("log")

        for algs in (("dmd", "sdmd"), ("dmd", "esdmd")):
            for alg in algs:
                freqs, mags = freqs_mags[alg]
                m, s, _ = ax.stem(freqs, mags, basefmt=" ")
                m.set(**mstyles[alg])
                s.set(**sstyles[alg])

        # formatting
        ax.set_xlim(fmin - fmargin, fmax + fmargin)
        ax.set_ylim(1e-3, 1.5)
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(self._smart_fmt))
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(self._smart_fmt))
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Scaled magnitude")
        ax.grid(axis="y", which="both", linestyle=":", lw=0.5, alpha=0.5)
        ax.tick_params(axis="y", which="both", length=0)
        ax.tick_params(axis="x", which="both", length=0)

        return fig

    @staticmethod
    def streaming_near_region(evals, region_coords, radius):
        distances = np.abs(evals - complex(*region_coords))
        return np.any(distances <= radius)

    def plot_eigenvalues(self):
        evals = dict(dmd=self.dmd_evals, sdmd=self.sdmd_evals[:, -1], esdmd=self.esdmd_evals[:, -1])

        # set styles
        edgesize = dict(dmd=1, sdmd=1, esdmd=2)
        msizes = dict(dmd=100, sdmd=120, esdmd=150)
        msizes_zoom = dict(dmd=200, sdmd=250, esdmd=300)

        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1], wspace=0.3, hspace=0.3)
        ax_main = fig.add_subplot(gs[:, 0])

        radius = 0.3  # radius for zoom regions

        sdmd_near_0 = self.streaming_near_region(evals["sdmd"], (0, 0), radius)
        esdmd_near_0 = self.streaming_near_region(evals["esdmd"], (0, 0), radius)
        sdmd_near_1 = self.streaming_near_region(evals["sdmd"], (1, 0), radius)
        esdmd_near_1 = self.streaming_near_region(evals["esdmd"], (1, 0), radius)
        axr = {}

        if (sdmd_near_0 or esdmd_near_0) and (sdmd_near_1 or esdmd_near_1):
            axr[0] = fig.add_subplot(gs[0, 1])
            axr[1] = fig.add_subplot(gs[1, 1])
        else:
            region_idx = 0 if (sdmd_near_0 or esdmd_near_0) else 1
            cell = gs[region_idx, 1]
            pos_cell = cell.get_position(fig)
            inset_w = pos_cell.width
            inset_h = pos_cell.height
            top = gs[0, 1].get_position(fig).y1
            bottom = gs[1, 1].get_position(fig).y0
            center = bottom + 0.5 * (top - bottom)
            inset_x = pos_cell.x0
            inset_y = center - 0.5 * inset_h
            axr[region_idx] = fig.add_axes([inset_x, inset_y, inset_w, inset_h])

        # main plot
        ax_main.spines["left"].set_position("zero")
        ax_main.spines["bottom"].set_position("zero")
        ax_main.spines["top"].set_visible(False)
        ax_main.spines["right"].set_visible(False)

        # unit circle
        unit_circle = plt.Circle((0, 0), 1, color="black", alpha=0.5, fill=False, linestyle=":", linewidth=0.6)
        ax_main.add_patch(unit_circle)

        # plot all evals
        for alg in ("dmd", "sdmd", "esdmd"):
            ax_main.scatter(
                np.real(evals[alg]),
                np.imag(evals[alg]),
                marker=self.cfg.markerstyle[alg],
                color=self.cfg.markercolor[alg],
                facecolor=self.cfg.markerface[alg],
                s=msizes[alg],
                linewidth=edgesize[alg],
                alpha=self.cfg.markeralpha[alg],
            )

        # axis limits and arrows
        lims = 1.1
        max_eval = np.max([np.max(np.abs(evals["dmd"])), np.max(np.abs(evals["sdmd"])), np.max(np.abs(evals["esdmd"]))])
        ax_main.set_xlim(-lims, max_eval + 0.05)
        ax_main.set_ylim(-lims, lims)

        asize = 15
        ax_main.plot(1, 0, ">k", markersize=asize, transform=ax_main.get_yaxis_transform(), clip_on=False)
        ax_main.plot(0, 1, "^k", markersize=asize, transform=ax_main.get_xaxis_transform(), clip_on=False)
        ax_main.plot(-lims, 0, marker="<", color="k", markersize=asize, transform=ax_main.transData, clip_on=False)
        ax_main.plot(0, -lims, marker="v", color="k", markersize=asize, transform=ax_main.transData, clip_on=False)
        ax_main.plot(
            [-lims * 1.02, 0],
            [0, -lims * 1.02],
            linestyle="",
            marker="o",
            markersize=1,
            alpha=0,
            transform=ax_main.transData,
            clip_on=False,
        )
        ax_main.set_aspect("equal")

        ax_main.text(lims, 0.1, "Re", ha="left", va="center", fontsize="x-large", fontstyle="italic", clip_on=False)
        ax_main.text(0.1, lims, "Im", ha="center", va="bottom", fontsize="x-large", fontstyle="italic", clip_on=False)

        ax_main.minorticks_off()
        ax_main.set_xticks([-1, 1])
        ax_main.set_yticks([-1, 1])
        ax_main.tick_params(bottom=False, left=False, right=False, top=False)

        # regions
        region_text = {0: "Region around $0+0j$", 1: "Region around $1+0j$"}
        lims = {
            0: {"x": (-radius, radius), "y": (-radius, radius)},
            1: {"x": (1 - radius, 1 + radius), "y": (-radius, radius)},
        }
        for i in axr.keys():
            ax = axr[i]
            for alg in ("dmd", "sdmd", "esdmd"):
                ax.scatter(
                    np.real(evals[alg]),
                    np.imag(evals[alg]),
                    marker=self.cfg.markerstyle[alg],
                    edgecolor=self.cfg.markercolor[alg],
                    facecolor=self.cfg.markerface[alg],
                    s=msizes_zoom[alg],
                    linewidth=edgesize[alg],
                    alpha=self.cfg.markeralpha[alg],
                )
            ax.set_xlim(*lims[i]["x"])
            ax.set_ylim(*lims[i]["y"])
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(region_text[i])

        plt.tight_layout()

        return fig

    def plot_exectimes(self):
        # load data
        esdmd = self.esdmd_diff_ms
        sdmd = self.sdmd_diff_ms

        # compute summary stats
        mean_es, std_es = np.mean(esdmd), np.std(esdmd)
        mean_sd, std_sd = np.mean(sdmd), np.std(sdmd)

        algs = ("esDMD", "sDMD")
        means = (mean_es, mean_sd)
        errors = (std_es, std_sd)
        colors = (self.cfg.markercolor["esdmd"], self.cfg.markercolor["sdmd"])

        # plot
        fig, ax = plt.subplots(figsize=(8, 6))
        y_pos = np.arange(len(algs))
        xticks = np.linspace(0, max(means) + 1.25 * max(errors), 4)

        bars = ax.barh(
            y_pos,
            means,
            xerr=errors,
            color=colors,
            alpha=0.8,
            edgecolor="none",
            capsize=10,
            error_kw={"elinewidth": 3, "capthick": 3},
        )

        ax.set_yticks([])
        ax.tick_params(axis="y", length=0)
        for bar, label in zip(bars, algs):
            w = bar.get_width()
            h = bar.get_height()
            y = bar.get_y() + h / 2
            ax.text(
                w / 2,
                y,
                label,
                va="center",
                ha="center",
                color="white",
                fontsize="xx-large",
                fontweight="bold",
                rotation=90
            )
        ax.invert_yaxis()

        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(self._smart_fmt))
        ax.minorticks_off()
        ax.tick_params(axis="x", which="both", length=0)

        ax.set_xlabel("Execution Time (ms)")
        ax.grid(axis="x", linestyle=":", alpha=0.5)
        plt.tight_layout()
        return fig

    def plot(self):
        plots = {
            "eigenvalues": self.plot_eigenvalues,
            "mode-frequency": self.plot_mode_frequencies,
            "exectime": self.plot_exectimes,
        }

        if "all" in self.cfg.show_plots:
            to_plot = [fn for _, fn in plots.items()]
        else:
            to_plot = [plots[p] for p in self.cfg.show_plots if p in plots]

        for fn in to_plot:
            fn()

        plt.show()
