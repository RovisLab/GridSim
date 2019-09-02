import random
import numpy as np
import scipy
import os

GRIDSIM_MIN = 0.0
GRIDSIM_MAX = 200.0
MODEL_CAR_MIN = 0.0
MODEL_CAR_MAX = 2.55

GRIDSIM_VEL_MIN = 0.0
GRIDSIM_VEL_MAX = 25.0
MODEL_CAR_VEL_MIN = 0.0
MODEL_CAR_VEL_MAX = 1.0


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


def linear_interpolation(value, in_min=0, in_max=200, out_min=0, out_max=2.55):
    out = out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)
    return out


def array_rescaling(gridsim_array, gs_min, gs_max, mc_min, mc_max):
    return [linear_interpolation(gridsim_elem, gs_min, gs_max, mc_min, mc_max) for gridsim_elem in gridsim_array]


def randomize(gridsim_array, rnd_min, rnd_max, how_many):
    indices = random.choices(range(len(gridsim_array) - 1), how_many)
    random_values = np.random.uniform(rnd_min, rnd_max, how_many)
    for idx in indices:
        gridsim_array[idx] = random_values[idx]
    return gridsim_array


def variate_samples(gridsim_array):
    return [scipy.random.normal(elem, 0.07) for elem in gridsim_array]


def convert_gridsim_to_model_car(gridsim_fp, mc_fp, randomize_seq=False, variate=False):
    gridsim_elements = get_elements_from_gridsim_record_file_sensor_array(gridsim_fp)
    with open(mc_fp, "w") as mc_f:
        for sequence in gridsim_elements:
            scaled_sequence = array_rescaling(sequence, GRIDSIM_MIN, GRIDSIM_MAX, MODEL_CAR_MIN, MODEL_CAR_MAX)
            if randomize_seq:
                scaled_sequence = randomize(scaled_sequence, 0.3, 0.8, len(scaled_sequence) / 20)
            if variate:
                scaled_sequence = variate_samples(scaled_sequence)
            for elem in scaled_sequence:
                mc_f.write("{0},".format(elem))
            mc_f.write("\n")


def convert_gridsim_action_to_model_car(gridsim_fp, mc_fp, randomize_seq=False, variate=False):
    gridsim_elements = get_elements_from_gridsim_record_file_sensor_array(gridsim_fp)
    with open(mc_fp, 'w') as mc_f:
        for sequence in gridsim_elements:
            scaled_sequence = array_rescaling(sequence, GRIDSIM_VEL_MIN, GRIDSIM_VEL_MAX, MODEL_CAR_VEL_MIN, MODEL_CAR_VEL_MAX)
            if randomize_seq:
                scaled_sequence = randomize(scaled_sequence, 0.03, 0.08, len(scaled_sequence) / 20)
            if variate:
                scaled_sequence = variate_samples(scaled_sequence)
            for elem in scaled_sequence:
                mc_f.write("{0}".format(elem))
            mc_f.write("\n")


def convert_all_data(base_path):
    front_sensor_files_path = [os.path.join(base_path, f) for f in os.listdir(base_path) if "distances" in f]
    velocity_files_path = [os.path.join(base_path, f) for f in os.listdir(base_path) if "velocity" in f]

    temp_front_sensor_files_path = [os.path.join(base_path, 'temp_' + f) for f in os.listdir(base_path) if "distances"]
    temp_velocity_files_paths = [os.path.join(base_path, 'temp_' + f) for f in os.listdir(base_path) if "velocity"]

    for file, temp_file in zip(front_sensor_files_path, temp_front_sensor_files_path):
        convert_gridsim_to_model_car(file, temp_file)
        os.remove(file)
        os.rename(temp_file, file)

    for file, temp_file in zip(velocity_files_path, temp_velocity_files_paths):
        convert_gridsim_action_to_model_car(file, temp_file)
        os.remove(file)
        os.rename(temp_file, file)

if __name__ == "__main__":
    bp = "D:\Python\gridsim_data"
    convert_all_data(bp)
