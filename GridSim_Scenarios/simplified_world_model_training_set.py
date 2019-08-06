import os


def get_elements_from_gridsim_record_file(tmp_fp):
    with open(tmp_fp, "r") as tmp_f:
        lines = tmp_f.readlines()
    elements = list()
    for line in lines:
        actual_delta, perceived_delta, in_fov, action = line.split(",")
        elements.append((float(actual_delta), float(perceived_delta), float(in_fov), float(action)))
    return elements


class SimplifiedWorldModelTrainingSet(object):
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

    def find_in_fov_data_indices(self, data):
        end_of_visibility_indices = list()
        found = False
        for idx in range(len(data)):
            if found is True:
                if data[idx][2] == 0:
                    end_of_visibility_indices.append(idx)
                    found = False
            if data[idx][2] == 1.0:
                found = True
        return end_of_visibility_indices

    def write_output_data(self, history, predictions, val=False):
        obs_file = os.path.join(self.base_path, "observations.npy") if val is False \
            else os.path.join(self.base_path, "observations_val.npy")

        pred_file = os.path.join(self.base_path, "predictions.npy") if val is False \
            else os.path.join(self.base_path, "predictions_val.npy")

        action_file = os.path.join(self.base_path, "actions.npy") if val is False \
            else os.path.join(self.base_path, "actions_val.npy")

        prev_action_file = os.path.join(self.base_path, "prev_actions.npy") if val is False \
            else os.path.join(self.base_path, "prev_actions_val.npy")

        with open(obs_file, "a") as obs:
            for h in history:
                for _, perceived_delta, in_fov, _ in h:
                    obs.write("{0},{1},".format(perceived_delta, in_fov))
                obs.write("\n")

        with open(pred_file, "a") as pred:
            for p in predictions:
                for actual_delta, _, in_fov, _ in p:
                    pred.write("{0},".format(actual_delta))
                pred.write("\n")

        with open(action_file, "a") as action:
            for p in predictions:
                for _, _, _, a in p:
                    action.write("{0},".format(a))
                action.write("\n")

        with open(prev_action_file, "a") as prev:
            for h in history:
                prev_action = h[-1][3]
                prev_action_vec = len(h) * [prev_action]
                for pa in prev_action_vec:
                    prev.write("{0},".format(pa))
                prev.write("\n")

    def process_training_file(self, training_fp, h_size, pred_size, val=False):
        raw_elements = get_elements_from_gridsim_record_file(training_fp)
        end_of_fov_indices = self.find_in_fov_data_indices(raw_elements)
        history_data, prediction_data = list(), list()
        for idx in end_of_fov_indices:
            h_slice = self.get_immediate_history(raw_elements, h_size, idx)
            if self.strict is True and len(h_slice) != h_size:
                continue
            p_slice = self.get_future(raw_elements, pred_size, idx)
            if len(p_slice) == pred_size:
                history_data.append(h_slice)
                prediction_data.append(p_slice)
        self.write_output_data(history_data, prediction_data, val)

    def process_training_file_all(self, training_fp, h_size, pred_size, val=False):
        raw_elements = get_elements_from_gridsim_record_file(training_fp)
        history_data, prediction_data = list(), list()
        for idx in range(len(raw_elements) - (h_size + pred_size)):
            h_slice = raw_elements[idx:idx + h_size]
            p_slice = raw_elements[idx + h_size:idx + h_size + pred_size]
            history_data.append(h_slice)
            prediction_data.append(p_slice)
        self.write_output_data(history_data, prediction_data, val)

    def process_all_data(self):
        files = os.listdir(self.base_path)
        training_files = [os.path.join(self.base_path, f) for f in files if "tmp" in f and "npy" in f]
        t_files = list()
        val_files = list()
        for f in training_files:
            if "_val" in f:
                val_files.append(f)
            else:
                t_files.append(f)

        for f in t_files:
            self.process_training_file_all(training_fp=f, h_size=self.h_size, pred_size=self.pred_size, val=False)

        for f in val_files:
            self.process_training_file_all(training_fp=f, h_size=self.h_size, pred_size=self.pred_size, val=True)

    def get_all_output_files(self):
        fl = ["actions.npy", "observations.npy", "predictions.npy", "prev_actions.npy",
              "actions_val.npy", "observations_val.npy", "predictions_val.npy", "prev_actions_val.npy"]

        return [os.path.join(self.base_path, f) for f in fl if os.path.exists(os.path.join(self.base_path, f))]


if __name__ == "__main__":
    training_set = SimplifiedWorldModelTrainingSet(base_path=os.path.join(os.path.dirname(__file__),
                                                                          "resources",
                                                                          "traffic_cars_data",
                                                                          "state_estimation_data"),
                                                   strict=True,
                                                   h_size=20,
                                                   pred_size=10)
    training_set.process_all_data()
