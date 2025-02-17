from scipy.integrate import quad
import numpy as np
import pandas as pd


def integrand(x):
    return x**2 #write  your function here

print(quad(integrand, 0, 1))