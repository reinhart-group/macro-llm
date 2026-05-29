import numpy as np


def main():
    # testing this outside the multiprocessing pool:
    ridx = 0
    rng = np.random.RandomState(ridx)
    len_possible_sequences = 63090
    init_idx = rng.choice(np.arange(len_possible_sequences), 5, replace=False)
    print(ridx, init_idx)


if __name__ == "__main__":
    main()
