from config import Config
from src.data_utils import DataUtils
from src.plotter import Plotter


def main():
    # override config if required
    cfg = Config()

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
