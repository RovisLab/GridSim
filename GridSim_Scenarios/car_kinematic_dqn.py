import pygame
from pygame.math import Vector2
from ConvDQNAgent import *
import cv2
from math_util import *
from car import Car
from car_kinematic_city import CitySimulator
from action_handler import *
import time


class DqnSimulator(CitySimulator):
    def __init__(self, screen, screen_width, screen_height,
                 sensor_size=150,
                 activations=False,
                 traffic=False,
                 record_data=False,
                 replay_data_path=None,
                 state_buf_path=None,
                 sensors=False,
                 rays_nr=15):
        super().__init__(screen, screen_width, screen_height, sensor_size=sensor_size, activations=activations,
                         traffic=traffic, record_data=record_data, replay_data_path=replay_data_path,
                         state_buf_path=state_buf_path, sensors=sensors, enabled_menu=True)
        self.checkpoint_path = self.current_dir + '/used_models/dqn/gridsim-dqn'
        self.rays_nr = rays_nr
        pygame.display.set_caption("Deep Reinforcement Learning")
        self.ticks = 10
        self.dqn_x_input_pixel = 220
        self.dqn_y_input_pixel = 470
        self.hw_surface_box = 320
        self.dqn_input = None
        self.global_distance = 0.0

        self.bgWidth, self.bgHeight = self.background.get_rect().size

    def get_sensor_reward(self, car):

        distances = self.enable_sensor(car, self.screen, rays_nr=self.rays_nr)
        min_distance = np.min(distances)

        if min_distance >= 50:
            return 1.0
        elif 50 > min_distance > 40:
            return 0.4
        elif 40 > min_distance > 30:
            return -0.8
        else:
            return -1.0

    def step(self, action, car, dt):
        current_position = [0, 0]
        next_position = [0, 0]
        current_position[0], current_position[1] = car.position[0], car.position[1]

        apply_action(action, car, dt)

        car.update(dt)
        self.draw_sim_environment(print_coords=True)
        sensor_distance_reward = self.get_sensor_reward(car)
        pygame.display.update()

        next_state = self.get_current_state(first_frames=False)
        next_position[0], next_position[1] = car.position[0], car.position[1]
        local_distance = round(
            np.sqrt((current_position[0] - next_position[0]) ** 2 + (current_position[1] - next_position[1]) ** 2), 4)
        if local_distance > 0.0:
            reward = normalize_zero_one(local_distance, 0.9, 0) * 0.15 + normalize_zero_one(car.velocity[0],
                                                                                           car.max_velocity,
                                                                                           0) * 0.15 + sensor_distance_reward * 0.7
        else:
            reward = -0.5
        print("reward " + str(reward))

        on_road = self.on_road(car, self.object_mask)
        if on_road:
            return next_state, reward, False
        else:
            return next_state, -1, True

    def get_current_state(self, first_frames):
        state_resized = self.object_mask.subsurface(
            (self.dqn_y_input_pixel, self.dqn_x_input_pixel, self.hw_surface_box, self.hw_surface_box))
        state_resized = pygame.transform.scale(state_resized,
                                               (int(self.hw_surface_box / 4), int(self.hw_surface_box / 4)))
        state_rgb_arr = pygame.surfarray.array3d(state_resized)
        state_bw = cv2.cvtColor(state_rgb_arr, cv2.COLOR_RGB2GRAY)
        norm = np.array([])
        norm = cv2.normalize(state_bw, dst=norm, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        if first_frames:
            single_state_tensor = np.reshape(norm, (norm.shape[0], norm.shape[1], 1))
            multi_state_tensor = np.concatenate(
                [single_state_tensor, single_state_tensor, single_state_tensor, single_state_tensor], axis=2)
            self.dqn_input = np.reshape(multi_state_tensor, (
                1, multi_state_tensor.shape[0], multi_state_tensor.shape[1], multi_state_tensor.shape[2]))
        else:
            self.dqn_input[0, :, :, 0] = self.dqn_input[0, :, :, 1]
            self.dqn_input[0, :, :, 1] = self.dqn_input[0, :, :, 2]
            self.dqn_input[0, :, :, 2] = self.dqn_input[0, :, :, 3]
            self.dqn_input[0, :, :, 3] = norm

        return self.dqn_input

    def random_reset(self, car):

        reset_posXY_vel_list = [[5, 125, 90.0], [33.5, -130, 0.0], [100.00, -130.0, 0.0],
                                [123.0, 100.0, 90.0]]
        random_state = random.choice(reset_posXY_vel_list)
        car.position = Vector2(random_state[0], random_state[1])
        car.velocity = Vector2(0.0, 0.0)
        car.angle = random_state[2]

    def train_conv_dqn(self, episodes):
        eps = episodes
        state_w = int(self.hw_surface_box / 4)
        state_h = int(self.hw_surface_box / 4)
        state_size = (state_w, state_h, 4)
        action_size = 7
        batch_size = 4
        agent = ConvDQNAgent(state_size, action_size, False)
        state = None
        # car = Car(0, 25)
        self.car.max_steering = 25
        for e in range(eps):
            init = True
            sanity_check = 15
            while not self.exit:
                dt = self.clock.get_time() / 1000

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.exit = True

                pressed = pygame.key.get_pressed()
                if pressed[pygame.K_ESCAPE]:
                    self.return_to_menu()

                if init:
                    self.car.update(dt)
                    self.draw_sim_environment(print_coords=True)
                    self.on_road(self.car, self.object_mask)
                    pygame.display.update()
                    init = False
                    state = self.get_current_state(first_frames=True)
                else:
                    action = agent.act(state)

                    next_state, reward, done = self.step(action, self.car, dt)
                    if reward < -0.2:
                        sanity_check -= 1
                    else:
                        sanity_check = 15
                    if sanity_check < 0:
                        done = True
                        reward = -1

                    agent.remember(state, action, reward, next_state, done)
                    state = next_state

                    if len(agent.memory) > batch_size:
                        agent.replay(batch_size)
                    if done:
                        print("episode: {}/{}, score: {}, e: {:.2}"
                              .format(e, episodes, time, agent.epsilon))
                        # self.reset(car)
                        self.random_reset(self.car)
                        break
                self.clock.tick(self.ticks)
            if e % 100 == 0 and e != 0:
                agent.save(self.checkpoint_path)
            if e % 1000 == 0 and e != 0:
                agent.load_target(self.checkpoint_path)
                agent.save_memory(self.checkpoint_path)

    def predict_conv_dqn(self):
        state_w = int(self.hw_surface_box / 4)
        state_h = int(self.hw_surface_box / 4)
        state_size = (state_w, state_h, 4)
        action_size = 7
        agent = ConvDQNAgent(state_size, action_size, True, self.checkpoint_path)
        agent.load_memory(self.checkpoint_path)
        # car = Car(pygame.display.get_surface().get_width() / self.ppu / 2,
        #           pygame.display.get_surface().get_height() / self.ppu / 2 - 26)
        state = None
        for e in range(1000):
            init = True
            sanity_check = 15
            while not self.exit:
                dt = self.clock.get_time() / 1000

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.exit = True

                pressed = pygame.key.get_pressed()
                if pressed[pygame.K_ESCAPE]:
                    self.return_to_menu()

                if init:
                    self.car.update(dt)
                    self.draw_sim_environment(print_coords=True)
                    self.on_road(self.car, self.object_mask)
                    pygame.display.update()
                    init = False
                    state = self.get_current_state(first_frames=True)
                else:
                    action = agent.act(state)
                    next_state, reward, done = self.step(action, self.car, dt)
                    state = next_state
                    if reward < -0.2:
                        sanity_check -= 1
                    else:
                        sanity_check = 15
                    if done:
                        self.random_reset(self.car)
                        break
                self.clock.tick(self.ticks)


if __name__ == '__main__':
    screen = pygame.display.set_mode((1280, 720))
    game = DqnSimulator(screen, 1280, 720)
    episodes = 100000
    # game.train_conv_dqn(episodes)
    game.predict_conv_dqn()
