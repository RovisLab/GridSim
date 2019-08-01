import os
from math import isclose


MAX_RAY_LEN = 199.0


def get_elements_from_gridsim_record_file_simplified(tmp_fp):
    with open(tmp_fp, "r") as tmp_f:
        lines = tmp_f.readlines()
    elements = list()
    for line in lines:
        actual_delta, perceived_delta, in_fov, action = line.split(",")
        elements.append((float(actual_delta), float(perceived_delta), float(in_fov), float(action)))
    return elements


def get_elements_from_gridsim_record_file_sensor_array(tmp_fp):
    with open(tmp_fp, "r") as f:
        elements = list()
        lines = f.readlines()
        for line in lines:
            s_elems = line.split(",")
            f_elems = list()
            try:
                for s in s_elems:
                    f_elems.append(float(s))
            except ValueError:
                pass
            elements.append(f_elems)
    return elements


def in_fov_sensor_array(rays):
    for r in rays:
        if not isclose(r, MAX_RAY_LEN):
            return True
    return False


def get_frequency_in_fov_data_simplified(training_fp):
    """
    get data count in_fov vs out_of_fov
    e.g. [free, free, free, obstacle, obstacle, free, free] will return [3, 2, 2]
    :param training_fp:
    :return: list of counts
    """
    accumulator_in_fov = list()
    accumulator_not_in_fov = list()
    counter_in_fov = 0
    counter_not_in_fov = 0
    raw_elements = get_elements_from_gridsim_record_file_simplified(training_fp)
    for idx in range(len(raw_elements) - 1):
        _, _, in_fov, _ = raw_elements[idx]
        _, _, next_in_fov, _ = raw_elements[idx+1]
        if in_fov == 1:
            counter_in_fov += 1
        else:
            counter_not_in_fov += 1
        if in_fov != next_in_fov:
            accumulator_in_fov.append(counter_in_fov) if in_fov == 1 else accumulator_not_in_fov.append(counter_not_in_fov)
            counter_in_fov = 0
            counter_not_in_fov = 0
    return accumulator_in_fov, accumulator_not_in_fov


def get_frequency_in_fov_data_sensor_array(training_fp):
    """
    get data count in_fov vs out_of_fov
    e.g. [free, free, free, obstacle, obstacle, free, free] will return [3, 2, 2]
    :param training_fp:
    :return: list of counts
    """
    accumulator_in_fov = list()
    accumulator_not_in_fov = list()
    counter_in_fov = 0
    counter_not_in_fov = 0
    raw_elements = get_elements_from_gridsim_record_file_sensor_array(training_fp)
    for idx in range(len(raw_elements) - 1):
        rays = raw_elements[idx]
        next_rays = raw_elements[idx+1]
        in_fov_crt = in_fov_sensor_array(rays)
        in_fov_next = in_fov_sensor_array(next_rays)
        if in_fov_crt:
            counter_in_fov += 1
        else:
            counter_not_in_fov += 1
        if in_fov_crt != in_fov_next:
            accumulator_in_fov.append(counter_in_fov) if in_fov_crt is True \
                else accumulator_not_in_fov.append(counter_not_in_fov)
            counter_in_fov = 0
            counter_not_in_fov = 0
    return accumulator_in_fov, accumulator_not_in_fov


def get_frequency_in_fov_sensor_array_folder(base_path):
    fl = os.listdir(base_path)
    fl = [os.path.join(base_path, f) for f in fl if "front_sensor" in f]
    for f in fl:
        print("Frequency for file {0}".format(f))
        i_f, n_i_f = get_frequency_in_fov_data_sensor_array(f)
        print("In FOV: {0}".format(i_f))
        print("Not in FOV: {0}".format(n_i_f))


def get_frequency_in_fov_simplified_folder(base_path):
    fl = os.listdir(base_path)
    fl = [os.path.join(base_path, f) for f in fl if "tmp" in f and ".npy" in f]
    for f in fl:
        print("Frequency for file {0}".format(f))
        i_f, n_i_f = get_frequency_in_fov_data_simplified(f)
        print("In FOV: {0}".format(i_f))
        print("Not in FOV: {0}".format(n_i_f))


if __name__ == "__main__":
    get_frequency_in_fov_sensor_array_folder("d:\\dev\\gridsim_state_estimation_data\\sensor_array\\training_data")
