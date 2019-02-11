import pygame
import pygame.gfxdraw
from read_write_trajectory import read_coords
from car_kinematic_model import Car, Simulator
from print_activations import print_activations
from obstacle_list import update_object_mask
from read_write_trajectory import save_frame, resize_image
import numpy as np
import os
import copy


class Replay(Simulator):
    def __init__(self, screen, screen_width, screen_height, activations, traffic, sensors, sensor_size):
        super().__init__(screen, screen_width, screen_height, activations=activations, traffic=traffic, sensors=sensors,
                         sensor_size=sensor_size)
        # self.background = pygame.image.load("resources/backgrounds/maps_overlay.png").convert()
        # self.background = pygame.transform.scale(self.background, (2500, 1261))
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

        if os.path.exists(car_data_path) is False:
            raise OSError('car_data_path does not exists')

        if self.traffic is True:
            self.init_traffic_cars()

        if self.print_activations is True:
            desired_layer_output = "convolution0"
            layer_names, image_buf, state_buf, activation_model = self.initialize_activation_model(desired_layer_output)

        # place car on road
        car_data = read_coords(car_data_path)
        car = Car(car_data[0][0], car_data[0][1])

        index = 1
        object_mask = pygame.Surface((self.screen_width, self.screen_height))
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
            car, index = self.drive(car, car_data, index)

            # Drawing
            stagePosX = car.position[0] * self.ppu
            stagePosY = car.position[1] * self.ppu

            rel_x = stagePosX % self.bgWidth
            rel_y = stagePosY % self.bgHeight

            # blit (BLock Image Transfer) the seamless background
            self.screen.blit(self.background, (rel_x - self.bgWidth, rel_y - self.bgHeight))
            self.screen.blit(self.background, (rel_x, rel_y))
            self.screen.blit(self.background, (rel_x - self.bgWidth, rel_y))
            self.screen.blit(self.background, (rel_x, rel_y - self.bgHeight))

            rotated = pygame.transform.rotate(self.car_image, car.angle)
            center_x = int(self.screen_width / 2) - int(rotated.get_rect().width / 2)
            center_y = int(self.screen_height / 2) - int(rotated.get_rect().height / 2)

            # draw the ego car
            self.screen.blit(rotated, (center_x, center_y))

            if self.sensors is True:
                # self.optimized_front_sensor(car)
                object_mask.fill((0, 0, 0))
                object_mask.blit(self.screen, (0, 0))
                update_object_mask(object_mask, rel_x, rel_y, self.bgWidth, self.bgHeight)

            stagePos = (stagePosX, stagePosY)

            if self.traffic is True:
                self.check_collisions(collision_list)
                self.traffic_movement(collision_list, object_mask, stagePos)

            if self.sensors is True:
                activation_mask.fill((0, 0, 0))
                self.optimized_front_sensor(car, object_mask, activation_mask)
                self.optimized_rear_sensor(car, object_mask, activation_mask)

            if self.print_activations is True:
                image_rect = pygame.Rect((390, 110), (500, 500))
                sub = activation_mask.subsurface(image_rect)
                self.input_image = pygame.surfarray.array3d(sub)

                image_buf[0] = self.input_image
                activations = activation_model.predict([image_buf, state_buf])
                print_activations(activations, layer_names, desired_layer_output)

            # draw trajectory
            if enable_trajectory is True:
                self.draw_trajectory(car, car_data, index)

            pygame.display.update()

            self.clock.tick(60)

        pygame.quit()

    def record_from_replay(self, replay_csv_path, save_simulator_frame_path=None, save_sensor_frame_path=None,
                           save_debug_frame_path=None, save_sensor_frame=False, save_simulator_frame=False,
                           save_debug_frame=False, draw_trajectory=False, traffic=False):
        if traffic is True:
            self.init_traffic_cars()

        # place car on road
        car = Car(5, 27, 270)
        if self.highway:
            car.angle = 270

        if os.path.exists(replay_csv_path) is False:
            raise OSError("path to replay csv doesn't exists")

        car_data = read_coords(replay_csv_path)

        index = 0
        object_mask = pygame.Surface((self.screen_width, self.screen_height))
        activation_mask = pygame.Surface((self.screen_width, self.screen_height))
        copy_sensor_path = copy.copy(save_sensor_frame_path)
        copy_debug_path = copy.copy(save_debug_frame_path)
        copy_simulator_path = copy.copy(save_simulator_frame_path)

        while not self.exit:
            collision_list = [False] * len(self.traffic_list)
            # Event queue
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True

            car.position = (car_data[index][0], car_data[index][1])
            car.angle = car_data[index][2]

            # index = (index + 1) % len(car_data)
            index = index + 1
            if index >= len(car_data):
                quit()

            # Logic
            # car.update(dt)

            # Drawing
            stagePosX = car.position[0] * self.ppu
            stagePosY = car.position[1] * self.ppu

            rel_x = stagePosX % self.bgWidth
            rel_y = stagePosY % self.bgHeight

            # blit (BLock Image Transfer) the seamless background
            self.screen.blit(self.background, (rel_x - self.bgWidth, rel_y - self.bgHeight))
            self.screen.blit(self.background, (rel_x, rel_y))
            self.screen.blit(self.background, (rel_x - self.bgWidth, rel_y))
            self.screen.blit(self.background, (rel_x, rel_y - self.bgHeight))

            rotated = pygame.transform.rotate(self.car_image, car.angle)
            center_x = int(self.screen_width / 2) - int(rotated.get_rect().width / 2)
            center_y = int(self.screen_height / 2) - int(rotated.get_rect().height / 2)

            # draw the ego car
            self.screen.blit(rotated, (center_x, center_y))

            # self.optimized_front_sensor(car)
            object_mask.fill((0, 0, 0))
            object_mask.blit(self.screen, (0, 0))
            update_object_mask(object_mask, rel_x, rel_y, self.bgWidth, self.bgHeight)

            stagePos = (stagePosX, stagePosY)

            if traffic is True:
                self.check_collisions(collision_list)
                self.traffic_movement(collision_list, object_mask, stagePos)

            activation_mask.fill((0, 0, 0))
            self.optimized_front_sensor(car, object_mask, activation_mask, display_obstacle_on_sensor=False)
            self.optimized_rear_sensor(car, object_mask, activation_mask, display_obstacle_on_sensor=False)

            image_name = 'image_' + str(index) + '.png'
            image_rect = pygame.Rect((440, 160), (400, 400))

            if draw_trajectory is True:
                center_screen = (int(self.screen_width / 2), int(self.screen_height / 2))
                trajectory_points = []

                for add_elem in range(5, 50, 10):
                    delta_position = (
                        car.position[0] - car_data[index + add_elem][0],
                        car.position[1] - car_data[index + add_elem][1])

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
                # activation_sub = pygame.transform.scale(activation_sub, (200, 200))
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
