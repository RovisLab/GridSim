import os
import shutil
from keras.models import load_model
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from data_loader import StateEstimationDataGenerator, StateEstimationSensorArrayDataGenerator
from sensor_grid_model_network import WorldModel
from simplified_world_model_network import WorldModel as SimplifiedWorldModel


def fit_distance_into_percentile(dist, percentile):
    if dist <= percentile[0]:
        return 0
    elif dist > percentile[-1]:
        return -1
    for idx in range(0, len(percentile) - 1):
        if percentile[idx] < dist <= percentile[idx + 1]:
            return idx


def calculate_statistics_sensor_array(results, ground_truth):
    percentile = [i for i in range(5, 101, 5)]
    accumulator = np.zeros(shape=(results.shape[0], len(percentile)))
    for i in range(len(results)):
        for j in range(len(results[i])):
            dist = abs(np.mean(results[i][j]) - np.mean(ground_truth[i][j])) / abs(np.mean(ground_truth[i][j])) * 100.0
            idx = fit_distance_into_percentile(dist, percentile)
            accumulator[i][idx] += 1
    return percentile, accumulator


def error_to_percentile_index(error, percentile):
    if error <= percentile[0]:
        return 0
    elif error > percentile[-1]:
        return -1
    for idx in range(0, len(percentile) - 1):
        if percentile[idx] < error <= percentile[idx + 1]:
            return idx


def calculate_statistics_simplified(results, ground_truth):
    percentile = [i for i in range(5, 101, 5)]
    accumulator = np.zeros(shape=(results.shape[0], len(percentile)))
    for i in range(len(results)):
        for j in range(len(results[i])):
            err = (abs(results[i][j] - ground_truth[i][j]) / abs(ground_truth[i][j])) * 100
            idx = error_to_percentile_index(err, percentile)
            accumulator[i][idx] += 1
    return percentile, accumulator


def draw_graphic(percentile, accumulator, base_path, graph_name):
    n_groups = len(percentile)

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.1
    opacity = 0.8
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
              'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan']
    rects = list()
    for i in range(len(percentile)):
        for j in range(len(accumulator)):
            means = accumulator[j]
            if i == 0:
                rects.append(plt.bar(index + bar_width * j / 2, means, bar_width, alpha=opacity, color=colors[j],
                                     label="Frame_t+{0}".format(j + 1)))
            else:
                rects.append(plt.bar(index + bar_width * j / 2, means, bar_width, alpha=opacity, color=colors[j]))

    plt.xlabel("Error [%]")
    plt.ylabel("Num Samples")
    plt.title("Errors Percentile")
    plt.xticks(index + bar_width, (str(i) for i in percentile))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, "perf", "{0}.png".format(graph_name)))
    plt.clf()


def calculate_array_diff(gt, pred):
    diff = abs(gt - pred)
    return np.mean(diff)


def draw_per_sample_error_sensor_array(ground_truth, predictions, base_path, graph_name):
    means_gt, means_p = list(), list()
    diff = list()
    for g_t, p in zip(ground_truth, predictions):
        m_g_t = list()
        m_p = list()
        d = list()
        for sample_num_g_t, sample_num_p in zip(g_t, p):
            m_g_t.append(np.mean(sample_num_g_t))
            m_p.append(np.mean(sample_num_p))
            d.append(calculate_array_diff(sample_num_g_t, sample_num_p))
        means_gt.append(m_g_t)
        means_p.append(m_p)
        diff.append(d)
    ox = np.arange(len(means_gt[0]))
    for idx in range(len(means_gt)):
        plt.plot(ox, means_gt[idx], label="Ground Truth")
        plt.plot(ox, means_p[idx], label="Predicted")
        plt.plot(ox, diff[idx], label="Error")
        plt.legend()
        plt.savefig(os.path.join(base_path, "perf", "{0}_frame_{1}.png".format(graph_name, idx + 1)))
        plt.clf()


def draw_per_sample_error_simplified(ground_truth, predictions, base_path, graph_name):
    ox = np.arange(len(ground_truth[0]))
    for idx in range(len(ground_truth)):
        plt.plot(ox, ground_truth[idx], label="Ground Truth")
        plt.plot(ox, predictions[idx], label="Predicted")
        plt.plot(ox, abs(predictions[idx] - ground_truth[idx]), label="Error")
        plt.legend()
        plt.savefig(os.path.join(base_path, "perf", "{0}_frame_{1}.png".format(graph_name, idx + 1)))
        plt.clf()


def create_graphs_sensor_array(weights_path, base_path, graph_name):
    model = WorldModel(prediction_horizon_size=10, validation=True, num_rays=50)
    model.load_weights(weights_path)

    test_generator = StateEstimationSensorArrayDataGenerator(input_file_path=base_path, batch_size=1,
                                                             prediction_horizon_size=10, shuffle=False, validation=True)
    results = model.predict_generator(test_generator)
    test_generator.reset_file_markers()
    ground_truth = list()
    for input_data, output_data in test_generator:
        ground_truth.append(output_data)
    ground_truth = np.swapaxes(np.array(ground_truth), 0, 1)

    # ground_truth = ground_truth.reshape((ground_truth.shape[0], -1))

    # results = results.reshape((results.shape[0], -1))

    percentile, accumulator = calculate_statistics_sensor_array(results, ground_truth)
    draw_graphic(percentile, accumulator, base_path, graph_name)
    draw_per_sample_error_sensor_array(ground_truth, results, base_path, graph_name)

    model.save_model()


def create_graphs_simplified(weights_path, base_path, graph_name):
    model = SimplifiedWorldModel(prediction_horizon_size=10, validation=True)
    model.load_weights(weights_path)

    test_generator = StateEstimationDataGenerator(input_file_path=base_path, batch_size=1,
                                                  prediction_horizon_size=10, shuffle=False, validation=True)
    results = model.predict_generator(test_generator)
    test_generator.reset_file_markers()
    ground_truth = list()

    for input_data, output_data in test_generator:
        ground_truth.append(output_data)
    ground_truth = np.swapaxes(np.array(ground_truth), 0, 1)
    ground_truth = ground_truth.reshape((ground_truth.shape[0], -1))

    results = results.reshape((results.shape[0], -1))

    percentile, accumulator = calculate_statistics_simplified(results, ground_truth)
    draw_graphic(percentile, accumulator, base_path, graph_name)
    draw_per_sample_error_simplified(ground_truth, results, base_path, graph_name)

    model.save_model()


def find_best_model_weights(model_path):
    files = os.listdir(model_path)
    weigths_files = [f for f in files if "weights" in f and "hdf5" in f]
    min_loss = 10000.0
    loss = 10000.0
    best_model = os.path.join(model_path, weigths_files[0])
    for weight in weigths_files:
        w = os.path.splitext(weight)[0]
        loss_val = float(w.split("-")[1])
        if loss_val < min_loss:
            best_model = os.path.join(model_path, weight)
            loss = loss_val
    return best_model, loss


def switch_sign(in_fp, out_fp):
    with open(in_fp, "r") as in_f:
        with open(out_fp, "w") as out_f:
            while True:
                line = in_f.readline()
                if len(line) == 0:
                    break
                actual_delta, perceived_delta, in_fov, vel = line.split(",")
                actual_delta = float(actual_delta)
                perceived_delta = float(perceived_delta)
                in_fov = float(in_fov)
                vel = float(vel)
                actual_delta = - actual_delta
                perceived_delta = - perceived_delta
                out_f.write("{0},{1},{2},{3}".format(actual_delta, perceived_delta, in_fov, vel))
                out_f.write("\n")


def convert_sign(base_path):
    old_files = os.listdir(base_path)
    old_files = [os.path.join(base_path, f) for f in old_files]
    bk_files = [f + ".bk" for f in old_files]
    for f in old_files:
        bk_name = f + ".bk"
        shutil.move(f, bk_name)
    for bk, converted in zip(bk_files, old_files):
        switch_sign(bk, converted)

    for bk in bk_files:
        os.remove(bk)


if __name__ == "__main__":
    # convert_sign("d:\\dev\\gridsim_state_estimation_data\\test\\training_set")
    bp = os.path.join(os.path.dirname(__file__), "resources", "traffic_cars_data", "state_estimation_data")
    wp = os.path.join(bp, "models", "weights.892-10.30.hdf5")
    create_graphs_simplified(weights_path=wp, base_path=bp, graph_name="statistics_892-10.30")
