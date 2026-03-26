import os

from numpy import linspace, zeros, float64
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# imports for type hints
from numpy import ndarray
from typing import List, Optional

def ensemble(time_list: List[ndarray], array_list: List[ndarray], interp_points: int,
             output_dir: Optional[str]) -> None:
    """
    :param time_list: list containing the numpy time arrays
    :param array_list: list containing the numpy arrays of different sizes
    :param interp_points: number of interpolation points
    :return: None
    """
    N: int = len(time_list)
    T: int = array_list[0].shape[-1]
    final_array: ndarray = zeros((N, T), dtype=float64)

    for i, (time_arr, data_arr) in enumerate(zip(time_list, array_list)):
        # compute cubic spline coefficients
        cs: CubicSpline = CubicSpline(time_arr, data_arr)
        time_norm: ndarray = linspace(time_arr[0], time_arr[-1], interp_points)

        final_array[i] = cs(time_norm)

    new_time: ndarray = linspace(0, 1, interp_points)

    mean: ndarray = final_array.mean(axis=0)
    std: ndarray = final_array.std(axis=0)

    lower_bound: ndarray = mean - std
    upper_bound: ndarray = mean + std

    plt.figure()
    plt.plot(new_time, final_array, color='darkred', label='mean')
    plt.fill_between(new_time, lower_bound, upper_bound, color='darkorange', alpha=0.5)
    plt.show()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'ensemble.png'))
    plt.close()

