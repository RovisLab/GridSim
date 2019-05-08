import pygame
import pygame.gfxdraw
from read_write_trajectory import read_coords
from car import Car
from car_kinematic_city import CitySimulator
from print_activations import print_activations
from read_write_trajectory import save_frame, resize_image
import numpy as np
import os


class Replay(CitySimulator):

    def __init__(self, screen, screen_width, screen_height, car_x=5, car_y=27, sensor_size=50, rays_nr=8,
                 activations=False, traffic=True, record_data=False, replay_data_path=None, state_buf_path=None,
                 sensors=False, distance_sensor=False, enabled_menu=False):
        super().__init__(screen, screen_width, screen_height, car_x, car_y, sensor_size, rays_nr, activations, traffic,
                         record_data, replay_data_path, state_buf_path, sensors, distance_sensor, enabled_menu)

        self.bgWidth, self.bgHeight = self.background.get_rect().size

    @staticmethod
    def drive(car, car_data, index):
        car.position = (car_data[index][0], car_data[index][1])
        car.angle = car_data[index][2]
        index = (index + 1) % len(car_data)
        # index += 1
        return car, index

    def draw_trajectory(self, car, car_data, index):
        center_screen = (int(self.screen_width / 2), int(self.screen_height / 2))
        trajectory_points = []
        self.draw_trajectory_points(car, car_data, index, center_screen, trajectory_points)
        self.draw_trajectory_lines(trajectory_points)

    def draw_trajectory_points(self, car, car_data, index, center_screen, trajectory_points):
        for add_elem in range(5, 30, 5):
            delta_position = (
                car.position[0] - car_data[index + add_elem][0], car.position[1] - car_data[index + add_elem][1])
            traj_point = (center_screen[0] + int(delta_position[0] * self.ppu),
                          center_screen[1] + int(delta_position[1] * self.ppu))
            trajectory_points.append(traj_point)
            # draw each trajectory point
            pygame.draw.circle(self.screen, (255, 255, 0), traj_point, 7, 7)

    def draw_trajectory_lines(self, trajectory_points):
        for traj_point, next_traj_point in zip(trajectory_points, trajectory_points[1:]):
            pygame.draw.aaline(self.screen, (255, 255, 0), traj_point, next_traj_point, 10)

    def key_handler(self):
        # User input
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_ESCAPE]:
            self.return_to_menu()
            quit()

    def replay(self, car_data_path, enable_trajectory=False):
        """
        Deprecated and will be removed
        :param car_data_path:
        :param enable_trajectory:
        :return:
        """
        if os.path.exists(car_data_path) is False:
            raise OSError('car_data_path does not exists')

        if self.traffic is True:
            self.init_traffic_cars()

        if self.print_activations is True:
            desired_layer_output = "convolution0"
            layer_names, image_buf, state_buf, activation_model = self.initialize_activation_model(desired_layer_output)

        # place car on road
        car_data = read_coords(car_data_path)
        print(car_data)
        self.car = Car(car_data[0][0], car_data[0][1])

        index = 1
        activation_mask = pygame.Surface((self.screen_width, self.screen_height))

        while not self.exit:
            if len(car_data) - 50 <= index <= len(car_data) - 10:
                index = 1

            if self.traffic is True:
                collision_list = [False] * len(self.traffic_list)

            # Event queue
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True

            self.key_handler()
            self.car, index = self.drive(self.car, car_data, index)

            stage_pos = self.draw_sim_environment(print_coords=True)
            stage_pos = (stage_pos[0], stage_pos[1])

            if self.traffic is True:
                self.check_collisions(collision_list)
                self.traffic_movement(collision_list, stage_pos)

            if self.sensors is True:
                activation_mask.fill((0, 0, 0))
                self.activate_sensors()

            if self.print_activations is True:
                image_rect = pygame.Rect((390, 110), (500, 500))
                sub = activation_mask.subsurface(image_rect)
                self.input_image = pygame.surfarray.array3d(sub)

                image_buf[0] = self.input_image
                activations = activation_model.predict([image_buf, state_buf])
                print_activations(activations, layer_names, desired_layer_output)

            # draw trajectory
            if enable_trajectory is True:
                self.draw_trajectory(self.car, car_data, index)

            pygame.display.update()

            self.clock.tick(60)

        pygame.quit()

    def record_from_replay(self, replay_csv_path, save_simulator_frame_path=None, save_sensor_frame_path=None,
                           save_debug_frame_path=None, save_sensor_frame=False, save_simulator_frame=False,
                           save_debug_frame=False, draw_trajectory=False, traffic=False, display_obstacle_on_screen=False):
        """
        Record images from runs
        :param replay_csv_path: path to replay csv
        :param save_simulator_frame_path: path where to save simulator frame
        :param save_sensor_frame_path: path where to save sensor frame
        :param save_debug_frame_path: path where to save debug(simulator and sensor) frame
        :param save_sensor_frame: bool
        :param save_simulator_frame: bool
        :param save_debug_frame: bool
        :param draw_trajectory: bool
        :param traffic: bool(if run was with traffic)
        :param display_obstacle_on_screen
        :return:
        """
        if traffic is True:
            self.init_traffic_cars()

        if os.path.exists(replay_csv_path) is False:
            raise OSError("path to replay csv doesn't exists")

        car_data = read_coords(replay_csv_path)

        index = 0
        activation_mask = pygame.Surface((self.screen_width, self.screen_height))

        while not self.exit:
            collision_list = [False] * len(self.traffic_list)
            # Event queue
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True

            self.car.position = (car_data[index][0], car_data[index][1])
            self.car.angle = car_data[index][2]

            # index = (index + 1) % len(car_data)
            index = index + 1
            if index >= len(car_data):
                quit()

            # Logic
            # car.update(dt)

            # Drawing
            stagePosX = self.car.position[0] * self.ppu
            stagePosY = self.car.position[1] * self.ppu

            rel_x = stagePosX % self.bgWidth
            rel_y = stagePosY % self.bgHeight

            # blit (BLock Image Transfer) the seamless background
            self.screen.blit(self.background, (rel_x - self.bgWidth, rel_y - self.bgHeight))
            self.screen.blit(self.background, (rel_x, rel_y))
            self.screen.blit(self.background, (rel_x - self.bgWidth, rel_y))
            self.screen.blit(self.background, (rel_x, rel_y - self.bgHeight))

            rotated = pygame.transform.rotate(self.car_image, self.car.angle)
            center_x = int(self.screen_width / 2) - int(rotated.get_rect().width / 2)
            center_y = int(self.screen_height / 2) - int(rotated.get_rect().height / 2)

            # draw the ego car
            self.screen.blit(rotated, (center_x, center_y))

            # self.optimized_front_sensor(car)

            stagePos = (stagePosX, stagePosY)

            if traffic is True:
                self.check_collisions(collision_list)
                self.traffic_movement(collision_list, stagePos)

            activation_mask.fill((0, 0, 0))
            self.optimized_front_sensor(activation_mask, display_obstacle_on_sensor=display_obstacle_on_screen)
            self.optimized_rear_sensor(activation_mask, display_obstacle_on_sensor=display_obstacle_on_screen)

            image_name = 'image_' + str(index) + '.png'
            image_rect = pygame.Rect((440, 160), (400, 400))

            if draw_trajectory is True:
                center_screen = (int(self.screen_width / 2), int(self.screen_height / 2))
                trajectory_points = []

                for add_elem in range(5, 50, 10):
                    delta_position = (
                        self.car.position[0] - car_data[index + add_elem][0],
                        self.car.position[1] - car_data[index + add_elem][1])

                    traj_point = (center_screen[0] + int(delta_position[0] * self.ppu),
                                  center_screen[1] + int(delta_position[1] * self.ppu))
                    trajectory_points.append(traj_point)

                # draw lines between trajectory points
                for traj_point, next_traj_point in zip(trajectory_points, trajectory_points[1:]):
                    pygame.draw.line(activation_mask, (255, 0, 0), traj_point, next_traj_point, 3)

            if save_sensor_frame is True:
                if save_sensor_frame_path is None:
                    print('no path for sensor images given')
                    quit()

                if os.path.exists(save_sensor_frame_path) is False:
                    os.makedirs(save_sensor_frame_path)

                activation_sub = activation_mask.subsurface(image_rect)
                activation_sub = resize_image(activation_sub, (200, 200))
                save_frame(activation_sub, image_name, save_sensor_frame_path)

            if save_simulator_frame is True:
                if save_simulator_frame_path is None:
                    print('no path for simulator images given')
                    quit()

                if os.path.exists(save_simulator_frame_path) is False:
                    os.makedirs(save_simulator_frame_path)

                sub = self.screen.subsurface(image_rect)
                sub = pygame.transform.scale(sub, (200, 200))
                save_frame(sub, image_name, save_simulator_frame_path)

            if save_debug_frame is True:
                if save_debug_frame_path is None:
                    print('no path for sensor images given')
                    quit()

                if os.path.exists(save_debug_frame_path) is False:
                    os.makedirs(save_debug_frame_path)

                activation_sub = activation_mask.subsurface(image_rect)
                activation_np = pygame.surfarray.array3d(activation_sub)
                sub = self.screen.subsurface(image_rect)
                sub_np = pygame.surfarray.array3d(sub)
                full_image = np.vstack((sub_np, activation_np))
                surf = pygame.surfarray.make_surface(full_image)
                save_frame(surf, image_name, save_debug_frame_path)

            pygame.display.update()

            self.clock.tick(60)

        pygame.quit()
