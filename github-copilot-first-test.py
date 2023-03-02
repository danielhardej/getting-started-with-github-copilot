# create a function to convert a csv to a python dict

import csv
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import scipy
from sklearn import linear_model, preprocessing, metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


def csv_to_dict(filename):
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


# wow, this is amazing... I can't believe how well this works

# Path: github-copilot-first-test.py

# create main function to run the program

def __main__():
    # get the data from the csv
    data = csv_to_dict('data.csv')
    # print the data
    print(data)

main()
