import os
import numpy as np
from data_visualizer import distances_visualizer


MODEL_CAR_ULTRASONIC_NUM_SENSORS = 5


def get_elements_from_prediction_file(pred_str):
    elements = pred_str.split(",")
    processed = list()
    if len(elements[-1]) == 0:
        elements.remove(elements[-1])
    for elem in elements:
        try:
            processed.append(float(elem))
        except ValueError:
            pass
    return processed


def model_car_rays_to_gridsim_rays(sequence):
    converted_sequence = list()
    for ray in sequence:
        converted_sequence.append(int(round(ray * (150.0 / 2.55))))
    return converted_sequence


def convert_gt_to_grids(gt_fp, dest_fp, num_rays):
    with open(gt_fp, "r") as ground_truth_file:
        with open(dest_fp, "w") as dest_f:
            while True:
                line = ground_truth_file.readline()
                if len(line) == 0:
                    break
                series_prediction = get_elements_from_prediction_file(line)
                for idx in range(0, len(series_prediction), num_rays):
                    sequence = series_prediction[idx:idx+num_rays]
                    if num_rays == MODEL_CAR_ULTRASONIC_NUM_SENSORS:
                        sequence = model_car_rays_to_gridsim_rays(sequence)
                    for s in sequence:
                        dest_f.write("{0},".format(s))
                    dest_f.write("\n")


def convert_predictions_to_grids(predictions, dest_fp, num_rays):
    predictions = np.swapaxes(predictions, 0, 1).tolist()
    with open(dest_fp, "w") as dest:
        for prediction_batch in predictions:
            for idx in range(0, len(prediction_batch)):
                sequence = prediction_batch[idx]
                if num_rays == MODEL_CAR_ULTRASONIC_NUM_SENSORS:
                    sequence = model_car_rays_to_gridsim_rays(sequence)
                for s in sequence:
                    dest.write("{0},".format(s))
                dest.write("\n")


def create_visual_evaluation(gt_file, predictions, dest_base_path, num_rays):
    eval_dir = os.path.join(dest_base_path, "grid_evaluation")
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)

    gt_dest_file = os.path.join(os.path.dirname(gt_file), "ground_truth.npy")
    pred_dest_file = os.path.join(os.path.dirname(gt_file), "predictions_output.npy")
    convert_gt_to_grids(gt_file, gt_dest_file, num_rays)
    convert_predictions_to_grids(predictions, pred_dest_file, num_rays)

    dist_vis = distances_visualizer.DistancesVisualizer(car_data_path=None,
                                                        front_data_path=gt_dest_file,
                                                        rear_data_path=None,
                                                        car_length=42,
                                                        debug_window=False)
    dist_vis.save_parallel_sensor_data(gt_dest_file,
                                       pred_dest_file,
                                       image_h=500,
                                       image_w=500,
                                       batch_size=10,
                                       draw_car=True,
                                       sensor_length=200,
                                       base_path=eval_dir)


if __name__ == "__main__":
    pass
