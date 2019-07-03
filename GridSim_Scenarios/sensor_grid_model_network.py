import os
from keras.layers import Conv2D, Dense, GRU, Concatenate, Lambda, Masking, Input, Reshape, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model
import matplotlib.pyplot as plt
from data_loader import StateEstimationSensorArrayDataGenerator


class WorldModel(object):
    """
    Number of front rays must be equal to number of back rays.
    """
    def __init__(self, prediction_horizon_size, num_rays, validation=False):
        self.state_estimation_data_path = os.path.join(os.path.dirname(__file__),
                                                       "resources",
                                                       "traffic_cars_data",
                                                       "state_estimation_data")
        self.prediction_horizon_size = prediction_horizon_size
        self.validation = validation
        self.input_shape = (None, 2, num_rays)
        self.action_shape = (self.prediction_horizon_size,)
        self.prev_action_shape = (None, 1)
        self.gru_num_units = 128
        self.mlp_layer_num_units = 10
        self.gru_layer_num_units = 32
        self.mlp_hidden_layer_size = prediction_horizon_size
        self.mlp_output_layer_units = 2 * num_rays
        self.model = None
        self.draw_statistics = False
        self.print_summary = True
        self._build_architecture()

    def _build_architecture(self):
        input_layer = Input(shape=self.input_shape)
        conv_layer1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="same")(input_layer)
        conv_layer2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same")(conv_layer1)
        conv_fc = Dense(units=256, activation="relu")(conv_layer2)
        conv_fc = Reshape(target_shape=(int(-1), int(conv_fc.shape[2]) * int(conv_fc.shape[3])))(conv_fc)
        action_layer = Input(shape=self.action_shape)
        prev_action_layer = Input(shape=self.prev_action_shape)
        action_dense = Dense(10, activation="relu")(action_layer)
        prev_action_dense = Dense(10, activation="relu")(prev_action_layer)
        gru_input = Concatenate()([conv_fc, prev_action_dense])
        gru = GRU(units=self.gru_num_units)(gru_input)
        mlp_outputs = list()
        for idx in range(self.mlp_hidden_layer_size):
            mlp_inputs = Lambda(lambda x: x[:, :idx + 1])(action_dense)
            mlp_in = Concatenate()([gru, mlp_inputs])
            mlp = Dense(units=self.mlp_layer_num_units, activation="relu")(mlp_in)
            mlp_output = Dense(units=self.mlp_output_layer_units, activation="relu")(mlp)
            mlp_outputs.append(mlp_output)

        self.model = Model([input_layer, action_layer, prev_action_layer], mlp_outputs)
        self.model.compile(optimizer=Adam(lr=0.00005), loss="mean_squared_error", metrics=["mae", "accuracy"])

    def train_model(self, epochs=100, batch_size=32):
        if self.print_summary:
            self.model.summary()

        es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=50)
        fp = self.state_estimation_data_path + "/" + "models" + "/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
        mc = ModelCheckpoint(filepath=fp, save_best_only=True, monitor="val_loss", mode="min")
        rlr = ReduceLROnPlateau(monitor="val_loss", patience=50, factor=0.00001)

        callbacks = [es, mc, rlr]

        generator = StateEstimationSensorArrayDataGenerator(input_file_path=self.state_estimation_data_path,
                                                            batch_size=batch_size,
                                                            prediction_horizon_size=self.mlp_hidden_layer_size,
                                                            shuffle=True)
        if self.validation:
            val_generator = StateEstimationSensorArrayDataGenerator(input_file_path=self.state_estimation_data_path,
                                                                    batch_size=batch_size,
                                                                    prediction_horizon_size=self.mlp_hidden_layer_size,
                                                                    shuffle=False,
                                                                    validation=True)
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


if __name__ == "__main__":
    model = WorldModel(prediction_horizon_size=10, num_rays=50)
    model.train_model(epochs=100, batch_size=32)
