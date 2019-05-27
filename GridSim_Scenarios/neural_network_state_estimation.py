from world_model import WorldModel
from agent_functions import AgentObservation
import random
from agent_functions import AgentAction


possible_actions = [AgentAction.DOWN, AgentAction.RIGHT, AgentAction.UP, AgentAction.LEFT, AgentAction.STAY]

num_moving_obstacles = 5

obstacle_encoders = [
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1]
]


def generate_action_sequences(num_sequences, history_size, prediction_horizon):
    """
    Generate random sequences of actions of fixed length (size of agent history + size of prediction horizon)
    :param num_sequences: number of sequences to generate
    :param history_size: size of agent's history
    :param prediction_horizon: size of prediction horizon
    :return: list of sequences
    """
    seq_list = list()
    for _ in range(num_sequences):
        crt_len = history_size + prediction_horizon
        sequence = list()
        for _ in range(crt_len):
            sequence.append(random.choice(possible_actions, k=1))
        seq_list.append(sequence)
    return seq_list


def build_observations(action_sequence, gridsim_object):
    """
    Simulate using GridSim the vehicle's actions and construct observations based on these actions and the simulation
    environment
    :param action_sequence: sequence of the vehicle's actions
    :param gridsim_object: simulator handle
    :return: observations corresponding to each action from the input sequence
    """
    observations = list()
    for action in action_sequence:
        car_pos, walls, cars = gridsim_object.advance_and_get_observation(action)
        observations.append(AgentObservation(car_pos, walls, cars))
    return observations


def prepare_training_data(sequence_list, history_size, prediction_horizon, gridsim_object):
    """
    Generate observations based on the action (sequence list)
    :param sequence_list: list of action sequences of length history_size + prediction_horizon
    :param history_size: size of history
    :param prediction_horizon: size of future prediction
    :param gridsim_object: simulator handle
    :return: (history_observations, prediction_observations)
    """
    history_observations = list()
    prediction_observations = list()
    for act_seq in sequence_list:
        observations = build_observations(act_seq, gridsim_object)
        h_obs = observations[:history_size]
        pred_obs = observations[-prediction_horizon:]
        history_observations.append(h_obs)
        prediction_observations.append(pred_obs)
    return history_observations, prediction_observations


class WorldModelAlgorithm(object):
    def __init__(self, gridsim_object, history_size, prediction_size, training_set_size):
        self.gridsim_object = gridsim_object
        self.history_size = history_size
        self.prediction_size = prediction_size
        self.training_set_size = training_set_size
        self.gridsim_object.start_simulation()
        self.history_observations, self.prediction_observations = None, None

    def get_training_data(self):
        action_sequences = generate_action_sequences(num_sequences=self.training_set_size,
                                                     history_size=self.history_size,
                                                     prediction_horizon=self.prediction_size)
        self.history_observations, self.prediction_observations = \
            prepare_training_data(sequence_list=action_sequences,
                                  history_size=self.history_size,
                                  prediction_horizon=self.prediction_size,
                                  gridsim_object=self.gridsim_object)

    def train_world_model_network(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass
