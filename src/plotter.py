import matplotlib.pyplot as plt
import numpy as np

from src.data_utils import DataUtils


class Plotter(DataUtils):
    def __init__(self, data_utils: DataUtils):
        if not isinstance(data_utils, DataUtils):
            raise TypeError("data_utils must be an instance of DataUtils")
        self.__dict__.update(data_utils.__dict__)

        self.common_stem_markerstyles = dict(markersize=12, markerfacecolor="none", markeredgewidth=0.8)
        self.tickparams = dict(length=0.8, width=0.5, color="grey")

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

        # linewidths
        stemlw = dict(dmd=0.5, sdmd=1, esdmd=1)
        mstyles = {
            alg: dict(
                marker=self.cfg.markerstyle[alg],
                markeredgecolor=self.cfg.markercolor[alg],
                **self.common_stem_markerstyles,
            )
            for alg in freqs_mags.keys()
        }
        sstyles = {alg: dict(linewidth=stemlw[alg], color=self.cfg.markercolor[alg]) for alg in freqs_mags.keys()}

        # margins and ticks
        freqs, mags = freqs_mags["dmd"]
        fmin, fmax = freqs.min(), freqs.max()
        fmargin = (fmax - fmin) * 0.1
        xticks = np.linspace(fmin, fmax, 4)
        yticks = np.array([1e-2, 0.5, 1.0, 1.25])

        # set up plot
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 8))

        for ax, algs in zip((ax0, ax1), (("dmd", "sdmd"), ("dmd", "esdmd"))):
            ax.set_yscale("log")
            for alg in algs:
                freqs, mags = freqs_mags[alg]
                m, s, _ = ax.stem(freqs, mags, basefmt=" ")
                m.set(**mstyles[alg])
                s.set(**sstyles[alg])

            # formatting
            ax.set_xlim(fmin - fmargin, fmax + fmargin)
            ax.set_ylim(1e-3, 1 + 0.25)
            ax.set_xticks(xticks)
            ax.set_xticklabels([f"{x:.0f}" for x in xticks])
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{y:.1f}" for y in yticks])
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Scaled Magnitude")
            ax.grid(True, which="both", linestyle=":", lw=0.5, alpha=0.7)
            ax.tick_params(axis="y", which="both", **self.tickparams)
            ax.tick_params(axis="x", which="both", length=0)

    def plot_eigenvalues(self):
        evals = dict(dmd=self.dmd_evals, sdmd=self.sdmd_evals[:, -1], esdmd=self.esdmd_evals[:, -1])

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 8))

        # draw unit circle

        # set styles
        markerstyles = dict(dmd=dict(s=55, alpha=0.5), sdmd=dict(s=60, alpha=0.8), esdmd=dict(s=60, alpha=0.8))

        for ax, algs in zip((ax0, ax1), (("dmd", "sdmd"), ("dmd", "esdmd"))):
            ax.spines["left"].set_position("zero")
            ax.spines["bottom"].set_position("zero")
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

            unit_circle = plt.Circle((0, 0), 1, color="black", fill=False, linestyle=":", linewidth=0.6)
            ax.add_artist(unit_circle)

            for alg in algs:
                ax.scatter(
                    np.real(evals[alg]),
                    np.imag(evals[alg]),
                    marker=self.cfg.markerstyle[alg],
                    color=self.cfg.markercolor[alg],
                    **markerstyles[alg],
                )

            # formatting
            lims = 1.1
            max_eval = np.maximum(np.max(np.abs(evals["dmd"])), np.max(np.abs(evals[algs[1]])))
            ax.set_xlim(-lims, max_eval + 0.05)
            ax.set_ylim(-lims, lims)
            asize = 8
            ax.plot(1, 0, ">k", markersize=asize, transform=ax.get_yaxis_transform(), clip_on=False)
            ax.plot(0, 1, "^k", markersize=asize, transform=ax.get_xaxis_transform(), clip_on=False)
            ax.plot(
                -lims, 0, marker="<", color="k", markersize=asize, linestyle="", transform=ax.transData, clip_on=False
            )
            ax.plot(
                0, -lims, marker="v", color="k", markersize=asize, linestyle="", transform=ax.transData, clip_on=False
            )
            ax.set_aspect("equal")

            ax.text(lims, 0.1, "Re", ha="left", va="center", fontsize="medium", fontstyle="italic", clip_on=False)
            ax.text(0.1, lims, "Im", ha="left", va="top", fontsize="medium", fontstyle="italic", clip_on=False)

            ax.minorticks_off()
            ax.set_xticks([-1, 1])
            ax.set_yticks([-1, 1])
            ax.tick_params(bottom=False, left=False, right=False, top=False)

    def plot_exectimes(self):
        fig, ax = plt.subplots(figsize=(12, 4))

        ts = np.arange(1, self.m + 1)

        ax.plot(ts, self.esdmd_diff_ms, label="esDMD", color=self.cfg.markercolor["esdmd"], lw=1.2)
        ax.plot(ts, self.sdmd_diff_ms, label="sDMD", color=self.cfg.markercolor["sdmd"], lw=1.2, linestyle=":")

        # truncate view
        all_times = np.concatenate([self.esdmd_diff_ms, self.sdmd_diff_ms])
        cutoff = np.percentile(all_times, 99.8)
        ax.set_ylim(0, cutoff)

        # arrowheads to show lines going out of view
        y_offset = cutoff * 0.01
        arrow_y = cutoff - y_offset

        for vals, color in [
            (self.esdmd_diff_ms, self.cfg.markercolor["esdmd"]),
            (self.sdmd_diff_ms, self.cfg.markercolor["sdmd"]),
        ]:
            mask = vals > cutoff
            if mask.any():
                xs = ts[mask]
                ys = np.full(mask.sum(), arrow_y)
                ax.scatter(xs, ys, marker="^", color=color, s=50, clip_on=True)

        # styling
        ax.legend(loc="best", fontsize="large", frameon=False)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Execution Time (ms)")

        # ticks
        xt = np.linspace(1, self.m + 1, 4, dtype=int)
        ax.set_xticks(xt)
        ax.set_xticklabels([f"{x}" for x in xt])
        yt = np.linspace(0, cutoff, 3)
        ax.set_yticks(yt)
        ax.set_yticklabels([f"{y:.1f}" for y in yt])

        ax.minorticks_off()
        ax.grid(True, which="major", linestyle=":", alpha=0.5)
        ax.margins(x=0, y=0)
        ax.tick_params(**self.tickparams)

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
