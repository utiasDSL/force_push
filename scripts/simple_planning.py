#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import IPython

from mm2d.model import ThreeInputModel


# model parameters
# link lengths
L1 = 1
L2 = 1

# input bounds
LB = -1
UB = 1


def main():
    model = ThreeInputModel(L1, L2, LB, UB, output_idx=[0, 1])


    IPython.embed()


if __name__ == '__main__':
    main()
