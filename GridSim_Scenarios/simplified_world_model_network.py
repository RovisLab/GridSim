import os
from keras.models import Model, load_model
from keras.layers import Input, Dense, GRU, Concatenate, Lambda, BatchNormalization, Reshape, Flatten
from keras.optimizers import Adam, SGD
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from data_loader import StateEstimationDataGenerator


class WorldModel(object):
    def __init__(self, prediction_horizon_size, history_size, validation=False):
        self.state_estimation_data_path = os.path.join(os.path.dirname(__file__),
                                                       "resources",
                                                       "traffic_cars_data",
                                                       "state_estimation_data")
        self.input_shape = (history_size, 1)
        self.fov_shape = (history_size, 1)
        self.mlp_layer_num_units = 10
        self.gru_layer_num_units = 32
        self.mlp_hidden_layer_size = prediction_horizon_size
        self.history_size = history_size
        self.mlp_output_layer_size = 1
        self.model = None
        self.action_shape = (prediction_horizon_size,)
        self.batch_size = 32
        self.print_summary = True
        self.draw_statistics = True
        self.validation = validation
        self._build_architecture()

    def _build_architecture2(self):
        input_layer = Input(shape=(self.history_size, 1))  # observation
        action_layer = Input(shape=self.action_shape)
        prev_action = Input(shape=(self.history_size, 1))
        mlp_layer = Concatenate()([input_layer, prev_action])
        mlp_layer = Dense(512, activation="relu")(mlp_layer)
        mlp_layer = Flatten()(mlp_layer)
        mlp_outputs = list()
        for idx in range(self.mlp_hidden_layer_size):
            mlp_inputs = Lambda(lambda x: x[:, :idx + 1])(action_layer)
            mlp_in = Concatenate()([mlp_layer, mlp_inputs])
            mlp = Dense(units=self.mlp_layer_num_units, activation="relu")(mlp_in)
            mlp_output = Dense(units=self.mlp_output_layer_size, activation="relu")(mlp)
            mlp_outputs.append(mlp_output)

        self.model = Model([input_layer, action_layer, prev_action], mlp_outputs)
        self.model.compile(optimizer=Adam(lr=0.00005), loss="mean_squared_error", metrics=["accuracy"])

    def _build_architecture(self):
        input_layer = Input(shape=self.input_shape)
        fov_layer = Input(shape=self.fov_shape)
        input_layer_ = Dense(10, activation="relu")(input_layer)
        fov_layer_ = Dense(10, activation="softmax")(fov_layer)
        action_layer = Input(shape=self.action_shape)
        action_layer_ = Dense(10, activation="relu")(action_layer)
        prev_action = Input(shape=(self.history_size, 1))
        prev_action_ = Dense(10, activation="relu")(prev_action)
        input_layer_ = Concatenate()([input_layer_, fov_layer_])
        gru_input = Concatenate()([input_layer_, prev_action_])
        # input_gru = Reshape(target_shape=(self.history_size, int(gru_input.shape[1])))(gru_input)
        gru = GRU(units=self.gru_layer_num_units)(gru_input)
        mlp_outputs = list()
        for idx in range(self.mlp_hidden_layer_size):
            mlp_inputs = Lambda(lambda x: x[:, :idx+1])(action_layer_)
            mlp_in = Concatenate()([gru, mlp_inputs])
            mlp = Dense(units=self.mlp_layer_num_units, activation="relu")(mlp_in)
            mlp_output = Dense(units=self.mlp_output_layer_size, activation="relu")(mlp)
            mlp_outputs.append(mlp_output)

        self.model = Model([input_layer, fov_layer, action_layer, prev_action], mlp_outputs)
        self.model.compile(optimizer=Adam(lr=0.00005), loss="mean_squared_error", metrics=["mae", "accuracy"])

    def train_network(self, epochs=10, batch_size=256):
        if self.print_summary:
            self.model.summary()

        es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=50)
        fp = self.state_estimation_data_path + "/" + "models" + "/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
        mc = ModelCheckpoint(filepath=fp, save_best_only=True, monitor="val_loss", mode="min")
        rlr = ReduceLROnPlateau(monitor="val_loss", patience=50, factor=0.00001)

        callbacks = [es, mc, rlr]

        generator = StateEstimationDataGenerator(input_file_path=self.state_estimation_data_path,
                                                 batch_size=batch_size,
                                                 history_size=self.history_size,
                                                 prediction_horizon_size=self.mlp_hidden_layer_size,
                                                 shuffle=True,
                                                 normalize=True)
        if self.validation:
            val_generator = StateEstimationDataGenerator(input_file_path=self.state_estimation_data_path,
                                                         batch_size=batch_size,
                                                         history_size=self.history_size,
                                                         prediction_horizon_size=self.mlp_hidden_layer_size,
                                                         shuffle=False,
                                                         validation=True,
                                                         normalize=True)
            history = self.model.fit_generator(generator=generator,
                                               epochs=epochs,
                                               validation_data=val_generator,
                                               verbose=2,
                                               callbacks=callbacks)
        else:
            history = self.model.fit_generator(generator=generator, epochs=epochs, verbose=2, callbacks=callbacks)

        if self.draw_statistics is True:
            plot_model(self.model, to_file=os.path.join(self.state_estimation_data_path, "perf", "model.png"))

            outputs = ["dense_6", "dense_8", "dense_10", "dense_12", "dense_14",
                       "dense_16", "dense_18", "dense_20", "dense_22", "dense_24"]

            loss_outputs = [x + "_loss" for x in outputs]
            loss_val_outputs = ["val_" + x + "_loss" for x in outputs]
            mae_outputs = [x + "_mean_absolute_error" for x in outputs]
            mae_val_outputs = ["val_" + x + "_mean_absolute_error" for x in outputs]
            # Plot training & validation accuracy values

            for out, mae, mae_val in zip(outputs, mae_outputs, mae_val_outputs):
                plt.plot(history.history[mae])
                if self.validation:
                    plt.plot(history.history[mae_val])
                plt.title("{0} mae".format(out))
                plt.ylabel('Mean Absolute Error')
                plt.xlabel('Epoch')
                if not self.validation:
                    plt.legend(['Mean Absolute Error'], loc='upper left')
                else:
                    plt.legend(["Mean Absolute Error", "Validation Mean Absolute Error"], loc="upper left")
                plt.savefig(os.path.join(self.state_estimation_data_path, "perf", "mae_{0}.png".format(out)))
                plt.clf()

            for out, loss, acc_loss in zip(outputs, loss_outputs, loss_val_outputs):
                plt.plot(history.history[loss])
                if self.validation:
                    plt.plot(history.history[acc_loss])
                plt.title('{0} loss'.format(out))
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                if self.validation:
                    plt.legend(['Train Loss', 'Test Loss'], loc='upper left')
                else:
                    plt.legend(["Train Loss"], loc="upper left")
                plt.savefig(os.path.join(self.state_estimation_data_path, "perf", "loss_{0}.png".format(out)))
                plt.clf()

    def evaluate_network(self, x_test, y_test, batch_size=128):
        self.model.evaluate(x_test, y_test, batch_size=batch_size)

    def predict(self, data_frame):
        return self.model.predict(x=data_frame)

    def save_model(self, model_path):
        return self.model.save(model_path)

    def load_model(self, model_path):
        self.model = load_model(model_path)


if __name__ == "__main__":
    model = WorldModel(prediction_horizon_size=10, history_size=10, validation=True)
    model.train_network(epochs=10000, batch_size=32)
