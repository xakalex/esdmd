import argparse

from config import Config
from src.data_utils import DataUtils
from src.plotter import Plotter


def main():
    # parse cli flag
    parser = argparse.ArgumentParser(description="Run DMD, sDMD and esDMD and produce plots.")
    parser.add_argument(
        "-f",
        "--figure",
        choices=["all", "eigenvalues", "mode-frequency", "exectime"],
        nargs="+",
        default=["all"],
        help="Specify which figures to plot. Default is all.",
    )
    args = parser.parse_args()

    # override config
    cfg = Config()
    if "all" in args.figure:
        cfg.show_plots = {"all"}
    else:
        cfg.show_plots = set(args.figure)

    # load utils
    du = DataUtils(cfg)

    # run streaming algorithms
    for k in range(du.m):
        du.run_streaming_algorithms(k)

    # plots
    p = Plotter(du)
    p.plot()


if __name__ == "__main__":
    main()
