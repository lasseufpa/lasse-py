"""
Generate several different random processes to be later analyzed.

@TODO use https://github.com/crflynn/stochastic - stochastic (random) processes

@author: Aldebaro Klautau
"""

from random import expovariate, normalvariate, random, randrange, uniform
from statistics import mean

import numpy as np


def calculate_mean_of_given_random_variable(all_realizations, time_instant):
    """
    Calculate the mean of a random variable extracted from a random process
    :param all_realizations: random process matrix
    :param time_instant: time corresponding to the desired random variable
    :return: mean of specified random variable
    """
    return mean(all_realizations[:, time_instant])


def get_realization_process_number_1(num_samples=100):
    """
    Generate one realization of your (customized) random process
    :param num_samples: number of samples in this realization
    :return: the waveform (vector) corresponding to the realization
    """
    x_shape = (num_samples,)  # define a shape
    x = np.zeros(x_shape)  # initialize
    previous_sample = -1
    for i in range(num_samples):  # loop to generate all samples
        this_sample = (
            previous_sample
            + randrange(10)
            + 5 * random()
            - uniform(2.5, 10.0)
            + expovariate(1 / 4)
        )
        x[i] = this_sample
        previous_sample = this_sample
    return x


def get_realization_process_number_2(num_samples=100):
    """
    Generate one realization of your (customized) random process
    :param num_samples: number of samples in this realization
    :return: the waveform (vector) corresponding to the realization
    """
    x_shape = (num_samples,)  # define a shape
    x = np.zeros(x_shape)  # initialize
    chosen_variance = 12  # variance for both distributions
    uniform_support = np.sqrt(12 * chosen_variance)  # variance = support^2 / 12
    for i in range(num_samples):  # loop to generate all samples
        coin = randrange(2)
        if coin == 0:
            this_sample = normalvariate(mu=0, sigma=np.sqrt(chosen_variance))
        elif coin == 1:
            this_sample = uniform(-uniform_support / 2.0, uniform_support / 2.0)
        else:
            raise Exception("Logic error!", coin)
        x[i] = this_sample
    return x


def generate_process_realizations(
    method_to_generate_realization,
    num_realizations=100,
    num_samples_per_realization=100,
):
    """
    Generates realizations of a given process.
    :param method_to_generate_realization: method that will be called to get realization
    :param num_realizations: number of realizations of the stochastic process
    :param num_samples_per_realization: number of samples in each realization
    :param output_file_name: name of file that will be written
    :return all realizations of the random process as a numpy array of dimension
            num_realizations x  num_samples_per_realization
    """
    # initialize with zeros
    all_realizations = np.zeros((num_realizations, num_samples_per_realization))
    for m in range(num_realizations):  # generate all realizations
        all_realizations[m] = method_to_generate_realization(
            num_samples=num_samples_per_realization
        )
    return all_realizations


def plot_histogram_of_given_random_variable(all_realizations, time_instant):
    pass  # YOU NEED TO IMPLEMENT THIS METHOD


def estimate_auto_correlation(all_realizations, time_instant1, time_instant2):
    pass  # YOU NEED TO IMPLEMENT THIS METHOD
