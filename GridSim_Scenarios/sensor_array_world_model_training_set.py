import os
import numpy as np
from math import isclose
MAX_SENSOR_DISTANCE = 199.0


def object_in_sensor_fov(sensor_array):
    for s in sensor_array:
        if not isclose(s, MAX_SENSOR_DISTANCE):
            return True
    return False


def get_elements_from_gridsim_record_file(tmp_fp):
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


class FrontSensorArrayTrainingSet(object):
    def __init__(self, base_path, strict=False, h_size=10, pred_size=10):
        self.base_path = base_path
        self.strict = strict
        self.h_size = h_size
        self.pred_size = pred_size

    def get_immediate_history(self, data, h_size, index):
        if index - h_size <= 0:
            h_size = index
        if index >= len(data):
            index = len(data) - 1
        return data[index - h_size:index]

    def get_future(self, data, pred_size, index):
        return data[index:index+pred_size]

    def find_in_fov_data_end_indices(self, data):
        end_of_visibility_indices = list()
        for idx in range(len(data)-1):
            if object_in_sensor_fov(data[idx]) and not object_in_sensor_fov(data[idx+1]):
                end_of_visibility_indices.append(idx + 1)
        return end_of_visibility_indices

    def find_in_fov_data_begin_indices(self, data):
        begin_of_visibility_indices = list()
        for idx in range(1, len(data)):
            if object_in_sensor_fov(data[idx]) and not object_in_sensor_fov(data[idx-1]):
                begin_of_visibility_indices.append(idx)
        return begin_of_visibility_indices

    def write_output_data(self, history_data, prediction_data, actions_data, prev_actions_data, val):
        obs_file = os.path.join(self.base_path, "observations.npy") if val is False \
            else os.path.join(self.base_path, "observations_val.npy")

        pred_file = os.path.join(self.base_path, "predictions.npy") if val is False \
            else os.path.join(self.base_path, "predictions_val.npy")

        action_file = os.path.join(self.base_path, "actions.npy") if val is False \
            else os.path.join(self.base_path, "actions_val.npy")

        prev_action_file = os.path.join(self.base_path, "prev_actions.npy") if val is False \
            else os.path.join(self.base_path, "prev_actions_val.npy")
        
        with open(obs_file, "a") as f:
            for h in history_data:
                f.write("{0},{1},".format(len(h), len(h[0])))
                for h_elem in h:
                    for ray in h_elem:
                        f.write("{0},".format(ray))
                f.write("\n")

        with open(pred_file, "a") as f:
            for p in prediction_data:
                for p_elem in p:
                    for ray in p_elem:
                        f.write("{0},".format(ray))
                f.write("\n")

        with open(action_file, "a") as f:
            for a in actions_data:
                for a_elem in a:
                    f.write("{0},".format(a_elem))
                f.write("\n")

        with open(prev_action_file, "a") as f:
            for a in prev_actions_data:
                for a_elem in a:
                    f.write("{0},".format(a_elem))
                f.write("\n")

    def process_training_file(self, training_fp, action_fp, h_size, pred_size, val=False):
        raw_elements = get_elements_from_gridsim_record_file(training_fp)
        action_elements = get_elements_from_gridsim_record_file(action_fp)
        action_elements = np.array(action_elements).reshape((len(action_elements))).tolist()
        d_b = self.find_in_fov_data_begin_indices(raw_elements)
        d_e = self.find_in_fov_data_end_indices(raw_elements)
        history_data, prediction_data = list(), list()
        action_data, prev_action_data = list(), list()
        for idx in d_b:
            h_slice = self.get_immediate_history(raw_elements, h_size, idx)
            p_a_slice = h_size * [self.get_immediate_history(action_elements, h_size, idx)[-1]]
            if self.strict is True and len(h_slice) != h_size:
                continue
            p_slice = self.get_future(raw_elements, pred_size, idx)
            a_slice = self.get_future(action_elements, pred_size, idx)
            if len(p_slice) == pred_size:
                history_data.append(h_slice)
                prediction_data.append(p_slice)
                action_data.append(a_slice)
                prev_action_data.append(p_a_slice)
        for idx in d_e:
            h_slice = self.get_immediate_history(raw_elements, h_size, idx)
            if self.strict is True and len(h_slice) != h_size:
                continue
            p_slice = self.get_future(raw_elements, pred_size, idx)
            if len(p_slice) == pred_size:
                history_data.append(h_slice)
                prediction_data.append(p_slice)
        self.write_output_data(history_data, prediction_data, action_data, prev_action_data, val)

    def process_all_data(self):
        files = os.listdir(self.base_path)
        training_files = [os.path.join(self.base_path, f) for f in files if "front_sensor" in f and "npy" in f]
        vel_files = [os.path.join(self.base_path, f) for f in files if "velocity" in f and "npy" in f]
        t_files = list()
        t_a_files = list()
        val_files = list()
        val_a_files = list()
        for f, a in zip(training_files, vel_files):
            if "_val" in f:
                val_files.append(f)
            else:
                t_files.append(f)
            if "_val" in a:
                val_a_files.append(a)
            else:
                t_a_files.append(a)

        for f, v in zip(t_files, t_a_files):
            self.process_training_file(training_fp=f, action_fp=v, h_size=self.h_size, pred_size=self.pred_size, val=False)

        for f, v in zip(val_files, val_a_files):
            self.process_training_file(training_fp=f, action_fp=v, h_size=self.h_size, pred_size=self.pred_size, val=True)

    def get_all_output_files(self):
        fl = ["actions.npy", "observations.npy", "predictions.npy", "prev_actions.npy",
              "actions_val.npy", "observations_val.npy", "predictions_val.npy", "prev_actions_val.npy"]

        return [os.path.join(self.base_path, f) for f in fl if os.path.exists(os.path.join(self.base_path, f))]


if __name__ == "__main__":
    training_set = FrontSensorArrayTrainingSet(base_path=os.path.join(os.path.dirname(__file__),
                                                                      "resources",
                                                                      "traffic_cars_data",
                                                                      "state_estimation_data"),
                                               strict=True,
                                               h_size=50,
                                               pred_size=10)

    training_set.process_all_data()
