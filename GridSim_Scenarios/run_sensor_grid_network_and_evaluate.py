import os
import datetime
import shutil
from sensor_grid_model_network import WorldModel as SensorGridWorldModel
from model_car_network import WorldModel as ModelCarWorldModel
from sensor_array_world_model_training_set import FrontSensorArrayTrainingSet
from model_car_training_set import ModelCarTrainingSet
from evaluate_model import create_graphs_sensor_array, create_graphs_model_car, find_best_model_weights, create_sensor_output, create_model_car_sensor_output


def run(*args, model_car=False, preprocess_data=True, train=True, perf_graph=True, grid_output=True, cleanup=True, plot_model=True):
    h_size = args[0]
    pred_size = args[1]
    validation = args[2]
    epochs = args[3]
    batch_size = args[4]
    num_rays = args[5]
    normalize = args[6]
    dest_path = args[7]
    base_path_training_set = args[8]

    if model_car is False:
        dirname = "sensor_array_{0}epochs_{1}".format(epochs,
                                                      str(datetime.datetime.now().time()).replace(":", "_").replace(".",
                                                                                                                    "_"))
    else:
        dirname = "model_car_{0}epochs_{1}".format(epochs,
                                                   str(datetime.datetime.now().time()).replace(":", "_").replace(".",
                                                                                                                 "_"))
    crt_dirname = os.path.join(dest_path, dirname)
    os.mkdir(crt_dirname)
    model_dir = os.path.join(crt_dirname, "model")
    eval_dir = os.path.join(crt_dirname, "evaluation")
    train_dir = os.path.join(crt_dirname, "training_data")
    os.mkdir(model_dir)
    os.mkdir(eval_dir)
    os.mkdir(train_dir)

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    if not os.path.exists(base_path_training_set):
        raise IOError

    print("[#] Initializing neural network model")
    if model_car is False:
        world_model = SensorGridWorldModel(prediction_horizon_size=pred_size, num_rays=num_rays,
                                           validation=validation, normalize=normalize)
    else:
        world_model = ModelCarWorldModel(pred_horizon_size=pred_size, num_rays=num_rays, val=validation)
    print("[##] Finished")
    base_path_neural_network = world_model.state_estimation_data_path
    if preprocess_data:
        print("[#] Preparing training data")
        if model_car is False:
            training_set = FrontSensorArrayTrainingSet(base_path=base_path_training_set,
                                                       strict=True,
                                                       h_size=h_size,
                                                       pred_size=pred_size)
        else:
            training_set = ModelCarTrainingSet(base_path=base_path_training_set,
                                               strict=True,
                                               h_size=h_size,
                                               pred_size=pred_size)
        training_set.process_all_data()

        print("[##] Finished")
        print("[#] Copying training data to neural network working directory")
        output_files = training_set.get_all_output_files()

        for training_file in output_files:
            try:
                shutil.copyfile(training_file, os.path.join(base_path_neural_network, os.path.basename(training_file)))
            except shutil.SameFileError:
                pass
        print("[##] Finished")
    else:
        print("[##] Skipping training data preprocessing")

    if train:
        print("[#] Training neural network")
        world_model.train_model(epochs=epochs, batch_size=batch_size)
        print("[##] Finished")
    else:
        print("[##] Skipping training network")

    if perf_graph:
        print("[#] Creating performance graphics")

        best_model, loss_val = find_best_model_weights(os.path.join(base_path_neural_network, "models"))

        if os.path.exists(best_model):
            if model_car is False:
                create_graphs_sensor_array(weights_path=best_model,
                                           base_path=base_path_neural_network,
                                           graph_name="statistics_{0}".format(loss_val),
                                           num_rays=num_rays,
                                           pred_horizon_size=pred_size,
                                           validation=validation)
            else:
                create_graphs_model_car(weights_path=best_model,
                                        base_path=base_path_neural_network,
                                        graph_name="statistics_{0}".format(loss_val),
                                        num_rays=num_rays,
                                        pred_horizon_size=pred_size,
                                        validation=validation)
        else:
            print("[###] Weights file {0} does not exist.".format(best_model))
            print("[##] Skipping performance graphics")
        print("[##] Finished")
    else:
        print("[##] Skipping performance graphics")

    if grid_output:
        print("[#] Creating sensor grid output visualization")
        best_model, _ = find_best_model_weights(os.path.join(base_path_neural_network, "models"))
        if model_car is False:
            create_sensor_output(weights_path=best_model,
                                 base_path=base_path_neural_network,
                                 num_rays=num_rays,
                                 pred_horizon_size=pred_size,
                                 validation=validation)
        else:
            create_model_car_sensor_output(weights_path=best_model,
                                           base_path=base_path_neural_network,
                                           num_rays=num_rays,
                                           pred_horizon_size=pred_size,
                                           validation=validation)
        print("[##] Finished")
    else:
        print("[##] Skipping grid visualization")

    if plot_model:
        m_p = os.path.join(base_path_neural_network, "perf", "model.png")
        print("[#] Plotting neural network model to file {0}".format(m_p))
        world_model.plot_model(m_p)
        print("[##] Finished")

    # Copy output
    training_graphs_path = os.path.join(base_path_neural_network, "perf")
    training_graphs = os.listdir(training_graphs_path)
    training_graphs = [os.path.join(training_graphs_path, t) for t in training_graphs if ".png" in t]
    for f in training_graphs:
        shutil.copyfile(f, os.path.join(model_dir, os.path.basename(f)))
    best_model, _ = find_best_model_weights(os.path.join(base_path_neural_network, "models"))
    if len(best_model) > 0:
        shutil.copyfile(best_model, os.path.join(model_dir, os.path.basename(best_model)))
    model_json = os.path.join(base_path_neural_network, "models", "model.json")
    if os.path.exists(model_json):
        shutil.copyfile(model_json, os.path.join(model_dir, "model.json"))

    grid_eval_dir = os.path.join(base_path_neural_network, "perf", "grid_evaluation")
    if os.path.exists(grid_eval_dir):
        shutil.copytree(grid_eval_dir, os.path.join(eval_dir, os.path.basename(grid_eval_dir)))

    output_files_nn = os.listdir(base_path_neural_network)
    output_files_nn = [os.path.join(base_path_neural_network, f) for f in output_files_nn if ".npy" in f]
    for f in output_files_nn:
        shutil.copy(f, os.path.join(eval_dir, os.path.basename(f)))

    if preprocess_data:
        output_files = training_set.get_all_output_files()
        fl = os.listdir(training_set.base_path)
        fl = [os.path.join(training_set.base_path, f) for f in fl if ".npy" in f]
        output_files.extend(fl)

        for f in output_files:
            shutil.copy(f, os.path.join(train_dir, os.path.basename(f)))

    if cleanup:
        for f in training_graphs:
            os.remove(f)
        weights_files = os.listdir(os.path.join(base_path_neural_network, "models"))
        weights_files = [os.path.join(base_path_neural_network, "models", w)
                         for w in weights_files if "weights" in w and "hdf5" in w]
        for w in weights_files:
            os.remove(w)
        if os.path.exists(model_json):
            os.remove(model_json)
        if os.path.exists(grid_eval_dir):
            shutil.rmtree(grid_eval_dir)
        for f in output_files_nn:
            os.remove(f)

        if preprocess_data:
            output_files = training_set.get_all_output_files()
            for f in output_files:
                os.remove(f)

    else:
        print("[##] Skipping cleanup")


if __name__ == "__main__":
    h_size = 50
    pred_size = 10
    validation = True
    epochs = 1000
    batch_size = 32
    num_rays = 5
    model_car = True
    normalize = False

    dest_path = os.path.join(os.path.dirname(__file__),
                             "resources",
                             "traffic_cars_data",
                             "state_estimation_data",
                             "evaluated")
    base_path_training_set = "d:\\ModelCarDataset\\datastream5216945\\sets"
    evaluation_base_path = "d:\\dev\\gridsim_state_estimation_data\\model_car\\eval"

    run(h_size, pred_size, validation, epochs, batch_size, num_rays, normalize,
        dest_path, base_path_training_set, preprocess_data=True,
        train=True, perf_graph=True, grid_output=True, cleanup=True, plot_model=True, model_car=model_car)


'''
if __name__ == "__main__":
    h_size = 150
    pred_size = 10
    validation = True
    epochs = 2000
    batch_size = 64
    num_rays = 30
    normalize = False

    dest_path = os.path.join(os.path.dirname(__file__),
                             "resources",
                             "traffic_cars_data",
                             "state_estimation_data",
                             "evaluated")
    base_path_training_set = "d:\\dev\\gridsim_state_estimation_data\\sensor_array\\training_data"
    evaluation_base_path = "d:\\dev\\gridsim_state_estimation_data\\sensor_array\\eval"

    print("[#] Initializing neural network model")
    world_model = SensorGridWorldModel(prediction_horizon_size=10, num_rays=num_rays,
                                       validation=True, normalize=normalize)
    print("[##] Finished")
    base_path_neural_network = world_model.state_estimation_data_path
    print("[#] Preparing training data")

    training_set = FrontSensorArrayTrainingSet(base_path=base_path_training_set,
                                               strict=True,
                                               h_size=h_size,
                                               pred_size=pred_size)
    training_set.process_all_data()

    print("[##] Finished")
    print("[#] Copying training data to neural network working directory")
    output_files = training_set.get_all_output_files()

    for training_file in output_files:
        try:
            shutil.copyfile(training_file, os.path.join(base_path_neural_network, os.path.basename(training_file)))
        except shutil.SameFileError:
            pass
    print("[##] Finished")
    print("[#] Training neural network")
    world_model.train_model(epochs=epochs, batch_size=batch_size)
    print("[##] Finished")
    print("[#] Creating performance graphics")

    perf_nn = os.listdir(os.path.join(base_path_neural_network, "perf"))
    p_graphs = [os.path.join(base_path_neural_network, "perf", p) for p in perf_nn]

    if not os.path.exists(os.path.join(evaluation_base_path, "perf")):
        os.makedirs(os.path.join(evaluation_base_path, "perf"))

    for graph in p_graphs:
        shutil.copyfile(graph, os.path.join(evaluation_base_path, "perf", os.path.basename(graph)))

    for training_file in output_files:
        try:
            shutil.copyfile(training_file, os.path.join(evaluation_base_path, os.path.basename(training_file)))
        except shutil.SameFileError:
            pass

    best_model, loss_val = find_best_model_weights(os.path.join(base_path_neural_network, "models"))

    create_graphs_sensor_array(weights_path=best_model,
                               base_path=evaluation_base_path,
                               graph_name="statistics_{0}".format(loss_val),
                               num_rays=num_rays)
    print("[##] Finished")
    print("[#] Copying results to output directory")
    dirname = "sensor_array_{0}epochs_{1}".format(epochs,
                                                  str(datetime.datetime.now().time()).replace(":", "_").replace(".", "_"))
    crt_dirname = os.path.join(dest_path, dirname)
    os.mkdir(crt_dirname)
    model_dir = os.path.join(crt_dirname, "model")
    eval_dir = os.path.join(crt_dirname, "evaluation")
    train_dir = os.path.join(crt_dirname, "training_data")
    os.mkdir(model_dir)
    os.mkdir(eval_dir)
    os.mkdir(train_dir)

    shutil.copyfile(os.path.join(base_path_neural_network, "models", "model.json"),
                    os.path.join(model_dir, "model.json"))

    shutil.copyfile(best_model, os.path.join(model_dir, os.path.basename(best_model)))

    for training_file in output_files:
        try:
            shutil.copyfile(training_file, os.path.join(train_dir, os.path.basename(training_file)))
        except shutil.SameFileError:
            pass

    temp_files = os.listdir(base_path_training_set)
    temp_files = [os.path.join(base_path_training_set, t) for t in temp_files
                  if ("front_sensor_distances" in t and ".npy" in t) or ("velocity" in t and ".npy" in t)]

    for t in temp_files:
        try:
            shutil.copyfile(t, os.path.join(train_dir, os.path.basename(t)))
        except shutil.SameFileError:
            pass

    graphics = os.listdir(os.path.join(evaluation_base_path, "perf"))
    graphics = [os.path.join(evaluation_base_path, "perf", g) for g in graphics if "png" in g]
    stats_graphics = list()
    train_graphics = list()
    for g in graphics:
        if "statistics" in g:
            stats_graphics.append(g)
        else:
            train_graphics.append(g)

    for g in train_graphics:
        shutil.copyfile(g, os.path.join(model_dir, os.path.basename(g)))

    for g in stats_graphics:
        shutil.copyfile(g, os.path.join(eval_dir, os.path.basename(g)))
    print("[##] Finished")

    print("[#] Creating visual grid evaluation")
    gt_file = os.path.join(train_dir, "predictions.npy")

    print("[#] Removing clutter")

    for output_file in output_files:
        os.remove(output_file)

    for f in output_files:
        os.remove(os.path.join(base_path_neural_network, os.path.basename(f)))

    graphs = os.listdir(os.path.join(base_path_neural_network, "perf"))
    graphs = [os.path.join(base_path_neural_network, "perf", g) for g in graphs]

    for g in graphs:
        os.remove(g)

    for f in output_files:
        os.remove(os.path.join(evaluation_base_path, os.path.basename(f)))

    perf_files = os.listdir(os.path.join(evaluation_base_path, "perf"))
    for f in perf_files:
        os.remove(os.path.join(evaluation_base_path, "perf", f))

    model_files = os.listdir(os.path.join(base_path_neural_network, "models"))
    for m_f in model_files:
        os.remove(os.path.join(base_path_neural_network, "models", m_f))
    print("[##] Finished")
'''
