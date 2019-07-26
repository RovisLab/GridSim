import os
import datetime
import shutil
from simplified_world_model_network import WorldModel as SimplifiedWorldModel
from simplified_world_model_training_set import SimplifiedWorldModelTrainingSet
from evaluate_model import create_graphs_simplified, find_best_model_weights


if __name__ == "__main__":
    h_size = 50
    pred_size = 10
    validation = True
    epochs = 2000
    batch_size = 20

    dest_path = os.path.join(os.path.dirname(__file__), "resources", "traffic_cars_data", "state_estimation_data", "evaluated")
    base_path_training_set = "d:\\dev\\gridsim_state_estimation_data\\test\\training_set"
    evaluation_base_path = "d:\\dev\\gridsim_state_estimation_data\\test\\eval"

    print("[#] Initializing neural network model")
    world_model = SimplifiedWorldModel(prediction_horizon_size=pred_size, validation=validation)
    print("[##] Finished")
    base_path_neural_network = world_model.state_estimation_data_path
    print("[#] Preparing training data")

    training_set = SimplifiedWorldModelTrainingSet(base_path=base_path_training_set,
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
    world_model.train_network(epochs=epochs, batch_size=batch_size)
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
    create_graphs_simplified(weights_path=best_model,
                             base_path=evaluation_base_path,
                             graph_name="statistics_{0}".format(loss_val))
    print("[##] Finished")
    print("[#] Copying results to output directory")
    dirname = "simplified_{0}epochs_{1}".format(epochs,
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
    temp_files = [os.path.join(base_path_training_set, t) for t in temp_files if "tmp" in t and ".npy" in t]

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
