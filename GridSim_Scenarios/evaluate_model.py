import os
from keras.models import load_model
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from data_loader import StateEstimationDataGenerator
from simplified_world_model_network import WorldModel


def calculate_statistics(results, ground_truth):
    percentile = [i for i in range(5, 101, 5)]
    accumulator = np.zeros(shape=(results.shape[0], len(percentile)))
    for i in range(len(results)):
        for j in range(len(results[i])):
            if ground_truth[i][j] != 0:
                percent_diff = (abs(results[i][j] - ground_truth[i][j]) / abs(ground_truth[i][j])) * 100.0
            else:
                continue
            broke = False
            if int(percent_diff) < percentile[0]:
                accumulator[i][0] += 1
                continue
            for k in range(0, len(percentile) - 1):
                if percentile[k] <= int(percent_diff) < percentile[k + 1]:
                    accumulator[i][k] += 1
                    broke = True
                    break
            if not broke:
                accumulator[i][-1] += 1
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

    plt.xlabel("Error [%]")
    plt.ylabel("Num Samples")
    plt.title("Errors Percentile")
    plt.xticks(index + bar_width, (str(i) if i != 100 else "{0}+".format(i) for i in percentile))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, "perf", graph_name))


def draw_per_sample_error(ground_truth, predictions):
    pass


def create_graphs(weights_path, base_path, graph_name):
    model = WorldModel(prediction_horizon_size=10, validation=True).create_model()
    model.load_weights(weights_path)

    test_generator = StateEstimationDataGenerator(input_file_path=base_path, batch_size=1,
                                                  prediction_horizon_size=10, shuffle=False, validation=True)
    results = np.array(model.predict_generator(test_generator, verbose=1))
    test_generator.reset_file_markers()
    ground_truth = list()
    for input_data, output_data in test_generator:
        ground_truth.append(output_data)
    ground_truth = np.swapaxes(np.array(ground_truth), 0, 1)
    ground_truth = ground_truth.reshape((ground_truth.shape[0], -1))
    results = results.reshape((results.shape[0], -1))

    percentile, accumulator = calculate_statistics(results, ground_truth)
    draw_graphic(percentile, accumulator, base_path, graph_name)

if __name__ == "__main__":
    bp = os.path.join(os.path.dirname(__file__), "resources", "traffic_cars_data", "state_estimation_data")
    wp = os.path.join(bp, "models", "weights.02-599.81.hdf5")
    create_graphs(weights_path=wp, base_path=bp, graph_name="errors_02-599.81.png")
