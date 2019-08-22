import os


class ModelCarTrainingSet(object):
    """
    rovis_data_descriptor_file:
    timestamp, delta_ts, path2image_rgb (e.g 1234, 12, ./samples/1234RGB.png)
    timestamp, delta_ts, path2image_d
    ...

    sensor_data_descriptor_file:
    timestamp, velocity, steering_angle, x, y, yaw, us_fl, us_fcl, us_fc, us_fcr, us_fr
    ...
    """
    def __init__(self, base_path, strict=True, h_size=10, pred_size=10):
        self.base_path = base_path
        self.strict = strict
        self.h_size = h_size
        self.pred_size = pred_size
        self.rovis_data_descriptor = os.path.join(self.base_path, "rovis_data_description.csv")
        self.sensor_data_descriptor = os.path.join(self.base_path, "sensor_data_description.csv")

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
            for h_idx in range(len(history)):
                for idx in range(len(history[h_idx])):
                    _, _, _, _, _, _, us_fl, us_fcl, us_fc, us_fcr, us_fr = history[h_idx][idx]
                    obs.write("{0},{1},{2},{3},{4},".format(us_fl, us_fcl, us_fc, us_fcr, us_fr))
                obs.write("\n")
        with open(pred_file, "a") as pred:
            for p_idx in range(len(predictions)):
                for idx in range(len(predictions[p_idx])):
                    _, _, _, _, _, _, us_fl, us_fcl, us_fc, us_fcr, us_fr = predictions[p_idx][idx]
                    pred.write("{0},{1},{2},{3},{4},".format(us_fl, us_fcl, us_fc, us_fcr, us_fr))
                pred.write("\n")
        with open(action_file, "a") as action:
            for a_idx in range(len(predictions)):
                for idx in range(len(predictions[a_idx])):
                    _, v, a, _, _, _, _, _, _, _, _ = predictions[a_idx][idx]
                    action.write("{0},{1},".format(v, a))
                action.write("\n")

        with open(prev_action_file, "a") as prev:
            for h in history:
                prev_action = [h[-1][1], h[-1][2]]
                prev_action_vec = len(h) * [prev_action]
                for pa in prev_action_vec:
                    prev.write("{0},{1},".format(pa[0], pa[1]))
                prev.write("\n")

    def process_training_file(self, sensor_descriptor_file, h_size, pred_size, val=False):
        elements = list()
        history, predictions = list(), list()
        with open(sensor_descriptor_file, "r") as sensor_file:
            while True:
                line = sensor_file.readline()
                if len(line) == 0:
                    break
                str_elems = line.split(",")
                f_elems = list()
                for elem in str_elems:
                    try:
                        f_elems.append(float(elem))
                    except ValueError:
                        pass
                elements.append(f_elems)
        for idx in range(0, len(elements) - (h_size + pred_size)):
            history.append(elements[idx: idx + h_size])
            predictions.append(elements[idx + h_size: idx + h_size + pred_size])
        self.write_output_data(history, predictions, val)

    def process_all_data(self):
        training_files = list()
        for root, dirs, files in os.walk(self.base_path):
            for f in files:
                if "sensor_data_description" in f:
                    training_files.append(os.path.join(root, f))
        t_files = list()
        val_files = list()
        for f in training_files:
            if "_val" in f:
                val_files.append(f)
            else:
                t_files.append(f)

        for f in t_files:
            self.process_training_file(sensor_descriptor_file=f, h_size=self.h_size,
                                       pred_size=self.pred_size, val=False)

        for f in val_files:
            self.process_training_file(sensor_descriptor_file=f, h_size=self.h_size,
                                       pred_size=self.pred_size, val=True)

    def get_all_output_files(self):
        fl = ["actions.npy", "observations.npy", "predictions.npy", "prev_actions.npy",
              "actions_val.npy", "observations_val.npy", "predictions_val.npy", "prev_actions_val.npy"]

        return [os.path.join(self.base_path, f) for f in fl if os.path.exists(os.path.join(self.base_path, f))]


if __name__ == "__main__":
    training_set = ModelCarTrainingSet(base_path="D:\\ModelCarDataset\\datastream5216945\\sets",
                                       strict=True,
                                       h_size=50,
                                       pred_size=10)
    training_set.process_all_data()
