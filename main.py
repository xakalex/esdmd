from config import Config
from src.data_utils import DataUtils
from src.plotter import Plotter


def main():
    du = DataUtils(Config())

    for k in range(du.m):
        du.run_streaming_algorithms(k)

    p = Plotter(du)
    p.plot()


if __name__ == "__main__":
    main()
