import matplotlib.pyplot as plt
import numpy as np

from src.data_utils import DataUtils

plt.rcParams.update(
    {
        "axes.linewidth": 1.5,
        "axes.labelsize": "x-large",
        "xtick.labelsize": "x-large",
        "ytick.labelsize": "x-large",
    }
)


class Plotter(DataUtils):
    def __init__(self, data_utils):
        if not isinstance(data_utils, DataUtils):
            raise TypeError("data_utils must be an instance of DataUtils")
        self.__dict__.update(data_utils.__dict__)

    def get_mode_frequencies(self, modes, evals):
        freqs = np.abs(np.angle(evals)) / (2 * np.pi * self.dt)
        mags = np.array([np.linalg.norm(modes[:, i]) * np.abs(evals[i]) for i in range(len(freqs))])
        mags /= np.max(mags)
        return freqs, mags

    def plot_mode_frequencies(self):
        # get DMD, SDMD, and ESDMD frequencies and magnitudes
        freqs_mags = dict(
            dmd=self.get_mode_frequencies(self.dmd_modes, self.dmd_evals),
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
        ax.set_ylim(1e-3, 1.25)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x:.0f}" for x in xticks])
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{y:.1f}" for y in yticks])
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Scaled Magnitude")
        ax.grid(axis="y", which="both", linestyle=":", lw=0.5, alpha=0.5)
        ax.tick_params(axis="y", which="both", length=0.5, width=0.5)
        ax.tick_params(axis="x", which="both", length=0)

        return fig

    def plot_eigenvalues(self):
        evals = dict(dmd=self.dmd_evals, sdmd=self.sdmd_evals[:, -1], esdmd=self.esdmd_evals[:, -1])

        # set styles
        msizes = dict(dmd=100, sdmd=120, esdmd=150)
        edgesize = dict(dmd=1, sdmd=1, esdmd=1.5)

        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1], wspace=0.3, hspace=0.3)
        ax_main = fig.add_subplot(gs[:, 0])
        ax_z0 = fig.add_subplot(gs[0, 1])
        ax_z1 = fig.add_subplot(gs[1, 1])

        # main plot
        ax = ax_main
        ax.spines["left"].set_position("zero")
        ax.spines["bottom"].set_position("zero")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # unit circle
        unit_circle = plt.Circle((0, 0), 1, color="black", alpha=0.5, fill=False, linestyle=":", linewidth=0.6)
        ax.add_patch(unit_circle)

        # plot all eigenvalues
        for alg in ("dmd", "sdmd", "esdmd"):
            vals = evals[alg]
            ax.scatter(
                np.real(vals),
                np.imag(vals),
                marker=self.cfg.markerstyle[alg],
                edgecolor=self.cfg.markercolor[alg],
                facecolor=self.cfg.markerface[alg],
                s=msizes[alg],
                linewidth=edgesize[alg],
                alpha=self.cfg.markeralpha[alg],
            )

        # axis limits and arrows
        lims = 1.08
        max_eval = np.max(
            np.array([np.max(np.abs(evals["dmd"])), np.max(np.abs(evals["sdmd"])), np.max(np.abs(evals["esdmd"]))])
        )
        ax.set_xlim(-lims, max_eval + 0.05)
        ax.set_ylim(-lims, lims)

        asize = 15
        ax.plot(1, 0, ">k", markersize=asize, transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(0, 1, "^k", markersize=asize, transform=ax.get_xaxis_transform(), clip_on=False)
        ax.plot(-lims, 0, marker="<", color="k", markersize=asize, transform=ax.transData, clip_on=False)
        ax.plot(0, -lims, marker="v", color="k", markersize=asize, transform=ax.transData, clip_on=False)
        ax.set_aspect("equal")

        ax.text(lims, 0.1, "Re", ha="left", va="center", fontsize="x-large", fontstyle="italic", clip_on=False)
        ax.text(0.1, lims, "Im", ha="center", va="bottom", fontsize="x-large", fontstyle="italic", clip_on=False)

        ax.minorticks_off()
        ax.set_xticks([-1, 1])
        ax.set_yticks([-1, 1])
        ax.tick_params(bottom=False, left=False, right=False, top=False)

        # define zoom regions
        zoom_radius = 0.3

        # zoom about (0, 0)
        ax = ax_z0
        for alg in ("dmd", "sdmd", "esdmd"):
            vals = evals[alg]
            ax.scatter(
                np.real(vals),
                np.imag(vals),
                marker=self.cfg.markerstyle[alg],
                facecolor=self.cfg.markerface[alg],
                edgecolor=self.cfg.markercolor[alg],
                s=msizes[alg],
                linewidth=edgesize[alg],
                alpha=self.cfg.markeralpha[alg],
            )
        ax.set_xlim(-zoom_radius, zoom_radius)
        ax.set_ylim(-zoom_radius, zoom_radius)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Region around $0+0j$", fontsize="x-large")

        # zoom around (1, 0)
        ax = ax_z1
        for alg in ("dmd", "sdmd", "esdmd"):
            vals = evals[alg]
            ax.scatter(
                np.real(vals),
                np.imag(vals),
                marker=self.cfg.markerstyle[alg],
                facecolor=self.cfg.markerface[alg],
                edgecolor=self.cfg.markercolor[alg],
                s=msizes[alg],
                linewidth=edgesize[alg],
                alpha=self.cfg.markeralpha[alg],
            )
        ax.set_xlim(1 - zoom_radius, 1 + zoom_radius)
        ax.set_ylim(-zoom_radius, zoom_radius)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Region around $1+0j$", fontsize="x-large")

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

        bars = ax.barh(y_pos, means, xerr=errors, color=colors, alpha=0.8, edgecolor="none", capsize=3)

        ax.set_yticks([])
        ax.tick_params(axis="y", length=0)
        for bar, label in zip(bars, algs):
            w = bar.get_width()
            h = bar.get_height()
            y = bar.get_y() + h / 2
            ax.text(w / 2, y, label, va="center", ha="center", color="white", fontsize="large", fontweight="bold")
        ax.invert_yaxis()

        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x:.2f}" for x in xticks])
        ax.minorticks_off()
        ax.tick_params(axis="x", which="both", length=0.5, width=0.5)

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
