from deap import base, creator, tools, algorithms
from keras.layers import Dense
from keras.models import Sequential
from keras.models import model_from_json
import pygame
import pygame.gfxdraw
from car import Car
from math_util import *
import pickle
from car_kinematic_city import CitySimulator
from action_handler import *
import config

population = []


class NeuroEvolutionary:
    def __init__(self, screen, screen_width, screen_height,  # simulator
                 sensor_size,
                 activations,
                 traffic,
                 record_data,
                 replay_data_path,
                 state_buf_path,
                 sensors,
                 shape=31,  # kinematic_ga
                 num_actions=8,
                 population_size=20,  # neuro_trainer
                 num_generations=30):

        self.kinematic_ga = KinematicGA(shape, num_actions)
        self.neuro_trainer = NeuroTrainer(population_size, num_generations, self.kinematic_ga.initInd)
        self.ga_sim = GASimulator(screen, screen_width, screen_height, sensor_size, activations, traffic, record_data,
                                  replay_data_path, state_buf_path, sensors, rays_nr=shape)


class NeuroTrainer(object):
    def __init__(self, population_size=20, num_generations=30, init_ind_func=None):
        self.population_size = population_size
        self.num_generations = num_generations
        self.init_individual = init_ind_func
        self.best_individual = None

    def train(self, eval_function):
        # fitness function, maximizes the accuracy
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        # tie fitness function to individual
        creator.create('Individual', list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()

        # create an individual with the random selection attribute
        toolbox.register("individual", creator.Individual, self.init_individual())
        # declare population of indivduals
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)

        # mate
        toolbox.register('mate', tools.cxBlend, alpha=0)
        # mutate
        toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=3, indpb=0.3)

        # selectia
        toolbox.register('select', tools.selTournament, tournsize=10)
        # evaluare
        toolbox.register('evaluate', eval_function, population_size=self.population_size)

        # training
        global population
        population = toolbox.population(n=self.population_size)
        # self.load_population("./np_population/pop_generation_38.pkl")
        r = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.5, ngen=self.num_generations, verbose=False)
        self.best_individual = tools.selBest(population, k=1)

    def load_population(self, path_to_data):
        with open(path_to_data, 'rb') as input:
            global population
            population = pickle.load(input)

    def test(self, test_features, test_labels, test_function):
        test_function(self.best_individual, test_features, test_labels)

    @staticmethod
    def save_population_data(path_where_to_save, generation_idx):
        global population
        with open(path_where_to_save + 'pop_generation_' + str(generation_idx) + '.pkl', 'w+') as output:
            pass
        with open(path_where_to_save + 'pop_generation_' + str(generation_idx) + '.pkl', 'wb') as output:
            pickle.dump(population, output, pickle.HIGHEST_PROTOCOL)


class GASimulator(CitySimulator):
    def __init__(self, screen, screen_width, screen_height,
                 sensor_size,
                 activations,
                 traffic,
                 record_data,
                 replay_data_path,
                 state_buf_path,
                 sensors,
                 rays_nr):
        super().__init__(screen, screen_width, screen_height, sensor_size=sensor_size, activations=activations,
                         traffic=traffic, record_data=record_data, replay_data_path=replay_data_path,
                         state_buf_path=state_buf_path, sensors=sensors, enabled_menu=True)

        self.rays_nr = rays_nr
        pygame.display.set_caption("Neuro-evolutionary")
        self.bgWidth, self.bgHeight = self.background.get_rect().size

    def run_ga(self, nn_model):
        # place car on road
        # 100,997 for seamless complex road
        self.car.max_steering = 27
        self.car.max_velocity = 30
        # car.velocity[0] = car.max_velocity
        global_distance = 0
        predicted_action = -1
        single_save_elite = False
        sanity_check = 15
        avg_vel_vec = np.array([])
        object_mask = pygame.Surface((self.screen_width, self.screen_height))

        while not self.exit:
            dt = self.clock.get_time() / 1000

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True

            pressed = pygame.key.get_pressed()
            if pressed[pygame.K_ESCAPE]:
                self.return_to_menu()

            current_position = [0, 0]
            current_position[0], current_position[1] = self.car.position[0], self.car.position[1]

            if predicted_action != Actions.reverse.value:
                apply_action(predicted_action, self.car, dt)

            # Logic
            self.car.update(dt)

            if not self.on_road(self.car, self.object_mask):
                break

            self.draw_sim_environment(print_coords=False)

            sensor_distances = self.enable_sensor(self.car, self.screen, self.rays_nr)
            input_data = np.append(sensor_distances, self.car.velocity[0])
            input_data_tensor = np.reshape(input_data, (1, input_data.shape[0]))
            prediction = nn_model.predict(input_data_tensor)
            predicted_action = np.argmax(prediction[0])

            myfont = pygame.font.SysFont(config.font, 30)
            text = myfont.render('Car velocity: ' + str(round(self.car.velocity[0], 2)), True, (255, 0, 255))
            self.screen.blit(text, (20, 20))
            pygame.display.update()

            next_position = self.car.position[0], self.car.position[1]
            local_distance = round(
                np.sqrt((current_position[0] - next_position[0]) ** 2 + (current_position[1] - next_position[1]) ** 2),
                4)
            if local_distance == 0:
                sanity_check -= 1
            else:
                sanity_check = 15
            if sanity_check < 0:
                break

            global_distance += local_distance
            avg_vel_vec = np.append(avg_vel_vec, self.car.velocity[0])
            # print(self.clock.get_fps())

            self.clock.tick(self.ticks)
            if global_distance > 2000 and single_save_elite is False:
                model_json = nn_model.to_json()
                with open("./used_models/ga/model_" + str(global_distance) + ".json", "w") as json_file:
                    json_file.write(model_json)
                # serialize weights to HDF5
                nn_model.save_weights("./used_models/ga/model_" + str(global_distance) + ".h5")
                print("Saved model to disk")
                single_save_elite = True

        avg_vel = np.mean(avg_vel_vec)
        return global_distance, avg_vel


class KinematicGA(object):
    def __init__(self, shape, num_actions):
        self.shape = shape
        self.model = self.build_classifier(shape + 1, num_actions)
        self.valid_layer_names = ['hidden1', 'hidden2', 'hidden3']
        self.layer_weights, self.layer_shapes = self.init_shapes()
        self.individual_idx = 0
        self.generation_idx = 0
        self.generative_ptsX = []
        self.generative_ptsY = []

    def build_classifier(self, shape, num_actions):
        # create classifier to train
        classifier = Sequential()

        classifier.add(
            Dense(units=6, input_dim=shape, activation='relu', name='hidden1', kernel_initializer='glorot_uniform',
                  bias_initializer='zeros'))

        classifier.add(Dense(units=7, activation='relu', kernel_initializer='glorot_uniform', name='hidden2',
                             bias_initializer='zeros'))

        classifier.add(
            Dense(units=int(num_actions), activation='softmax', kernel_initializer='glorot_uniform', name='hidden3',
                  bias_initializer='zeros'))

        # Compile the CNN
        classifier.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return classifier

    def neuro_eval(self, individual, population_size):
        # get the weights, extract weights from individual, change the weights with evo weights
        individual = np.asarray(individual)
        for layer_name, weight, bias in zip(self.valid_layer_names, self.layer_shapes[0::2], self.layer_shapes[1::2]):
            self.model.get_layer(layer_name).set_weights(
                [individual[weight[0]:weight[0] + np.prod(weight[1])].reshape(weight[1]),
                 individual[bias[0]:bias[0] + np.prod(bias[1])].reshape(bias[1])])

        screen = pygame.display.set_mode((1280, 720))
        sim = GASimulator(screen, 1280, 720, sensor_size=100, activations=False, traffic=False, record_data=False,
                          replay_data_path=None, state_buf_path=None, sensors=False, rays_nr=self.shape)
        fitness, avg_vel = sim.run_ga(self.model)
        self.generative_ptsY.append(fitness)
        print("ind %i gen %i distance: %.2f" % (
            self.individual_idx, self.generation_idx, fitness))
        checkpoint_freq = 2
        if (self.generation_idx % checkpoint_freq == 0) and (self.individual_idx == population_size):
            NeuroTrainer.save_population_data("./used_models/ga/np_population/", self.generation_idx)
        if self.individual_idx < population_size:
            self.generative_ptsX.append(self.generation_idx)
            self.individual_idx += 1
        elif self.individual_idx:
            self.generation_idx += 1
            self.generative_ptsX.append(self.generation_idx)
            self.individual_idx = 1
        # write_data("./fitness_vel_gen.csv", fitness, avg_vel, self.generation_idx)
        return fitness,

    def init_shapes(self):
        layer_weights = []
        layer_shapes = []
        # get layer weights
        for layer_name in self.valid_layer_names:
            layer_weights.append(self.model.get_layer(layer_name).get_weights())

        # break up the weights and biases
        # layer_weights = np.concatenate(layer_weights) ???
        layer_wb = []
        for w in layer_weights:
            layer_wb.append(w[0])
            layer_wb.append(w[1])

        # set starting index and shape of weight/bias
        for layer in layer_wb:
            layer_shapes.append(
                [0 if layer_shapes.__len__() == 0 else layer_shapes[-1][0] + np.prod(
                    layer_shapes[-1][1]), layer.shape])

        layer_weights = np.asarray(layer_wb)
        # flatten all the vectors
        layer_weights = [layer_weight.flatten() for layer_weight in layer_weights]

        # make one vector of all weights and biases
        layer_weights = np.concatenate(layer_weights)
        return layer_weights, layer_shapes

    def initInd(self):
        # init individual with w
        ind = self.layer_weights.tolist()
        return ind

    def load_model(self, model_name):
        json_file = open('./used_models/ga/' + model_name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights("./used_models/ga/" + model_name + ".h5")
        print("Loaded model from disk")