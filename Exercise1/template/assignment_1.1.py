import numpy as np
import numpy.typing as npt


def load_grades(filename: str) -> npt.NDArray:
    # TODO: read grades from the file.
    # ====================================================================
    with open(filename, 'r') as file:
        grades = [float(line) for line in file]
    # ====================================================================
    return np.array(grades)


def python_compute(array: npt.NDArray) -> tuple[float, float]:
    # TODO: compute the mean and the variance using standard Python.
    # ====================================================================
    mean = 0
    for grade in array:
        mean += grade
    mean /= len(array)

    var = 0
    for grade in array:
        var += (grade - mean)**2
    var /= (len(array) - 1)
    # ====================================================================
    return mean, var


def numpy_compute(array: npt.NDArray, ddof: int = 0) -> tuple[float, float]:
    # TODO: compute the mean and the variance using numpy.
    # ====================================================================
    mean = np.mean(array)
    var = np.var(array, ddof=1)
    # ====================================================================
    return mean, var


if __name__ == "__main__":
    # TODO: load the grades from the file, compute the mean and the
    # variance using both implementations and report the results.
    # ====================================================================
    arr = load_grades(filename="./data/G.txt")
    mean1, var1 = python_compute(arr)
    mean2, var2 = numpy_compute(arr)
    print(mean1, var1)
    print(mean2, var2)
    # ====================================================================
