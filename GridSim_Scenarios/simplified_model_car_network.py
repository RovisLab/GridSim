import numpy as np


class WorldModel(object):
    def __init__(self, h_size, pred_size, validation=False):
        # number of ultrasonic sensors
        self.num_rays = 5

        # history size
        self.h_size = h_size

        # prediction horizon size
        self.pred_size = pred_size

        # perform validation
        self.validation = validation

        self.input_shape = (None, self.num_rays)
        self.action_shape = (self.pred_size,)
        self.prev_action_shape = (None, 1)
        self.gru_num_units = 256
        self.mlp_layer_num_units = 100
        self.num_rays = self.num_rays
        self.mlp_hidden_layer_size = pred_size
        self.mlp_output_layer_units = self.num_rays
        self.model = None
        self.draw_statistics = True
        self.print_summary = True
        self.output_names = list()
        self._build_architecture()

    def load_weights(self, m_p):
        pass

    def _build_architecture(self):
        pass

    def save_model(self):
        pass

    def train_model(self, epochs=100, batch_size=32):
        pass

    def predict_generator(self, generator):
        return np.array(self.model.predict_generator(generator, verbose=1))
