from car_kinematic_model import Simulator
import os
import pygame
import pygame.gfxdraw
from read_write_trajectory import read_traffic_data
from print_activations import print_activations
from traffic_car import TrafficCar


class CitySimulator(Simulator):

    def __init__(self, screen, screen_width, screen_height, car_x=5, car_y=27, sensor_size=50, rays_nr=8,
                 activations=False, traffic=True, record_data=False, replay_data_path=None, state_buf_path=None,
                 sensors=False, distance_sensor=False, enabled_menu=False):
        object_map_path = "resources/backgrounds/maps_overlay_obj.png"
        background_path = "resources/backgrounds/maps_overlay.png"
        car_image_path = "resources/cars/car_eb_2.png"
        traffic_car_image_path = "resources/cars/car_traffic.png"
        object_car_image_path = "resources/cars/object_car.png"
        super().__init__(screen, screen_width, screen_height, car_x, car_y, sensor_size, rays_nr, activations,
                         record_data, replay_data_path, state_buf_path, sensors, distance_sensor, enabled_menu,
                         object_map_path, background_path, car_image_path, traffic_car_image_path,
                         object_car_image_path)

        self.car_image = pygame.transform.scale(self.car_image, (42, 20))
        self.traffic_car_image = pygame.transform.scale(self.traffic_car_image, (42, 20))
        self.object_car_image = pygame.transform.scale(self.object_car_image, (42, 20))
        self.background = pygame.transform.scale(self.background, (2500, 1261))
        self.bgWidth, self.bgHeight = self.background.get_rect().size
        self.object_map = pygame.transform.scale(self.object_map, (2500, 1261))

        self.traffic_list = []
        self.traffic = traffic

    @staticmethod
    def traffic_car_collision(traffic_car_1, traffic_car_2, collision_list, collision_index):
        """
        loop through traffic car list and check if the positions are overlapped and modify collision list
        :param traffic_car_1:
        :param traffic_car_2:
        :param collision_list:
        :param collision_index:
        :return:
        """
        for x1, y1 in zip(traffic_car_1.data_x[traffic_car_1.index: traffic_car_1.index + 40], traffic_car_1.data_y[
                                                                                               traffic_car_1.index:
                                                                                               traffic_car_1.index + 40]):
            for x2, y2 in zip(traffic_car_2.data_x[traffic_car_2.index: traffic_car_2.index + 40], traffic_car_2.data_y[
                                                                                                   traffic_car_2.index:
                                                                                                   traffic_car_2.index + 40]):
                if abs(x2 - x1) <= 1 and abs(y2 - y1) <= 20:
                    collision_list[collision_index] = True

    def init_traffic_cars(self):
        """
        initialize traffic
        :return:
        """
        trajectories = read_traffic_data(os.path.join(self.current_dir, "resources/traffic_cars_data/traffic_trajectories.csv"))
        for trajectory in trajectories:
            traffic_car = TrafficCar(trajectory[0], int(trajectory[1]))
            self.traffic_list.append(traffic_car)

    def check_collisions(self, collision_list):
        """
        function that checks for traffic car collision
        :param collision_list:
        :return:
        """
        for i in range(0, len(self.traffic_list) - 1):
            for j in range(i + 1, len(self.traffic_list)):
                self.traffic_car_collision(self.traffic_list[i], self.traffic_list[j], collision_list, j)

    def traffic_movement(self, collision_list, stagePos):
        """
        traffic movement
        :param collision_list:
        :param stagePos: position of stage
        :return:
        """
        for i in self.traffic_list:
            if collision_list[self.traffic_list.index(i)]:
                i.index -= 1
            i.trajectory(stagePos, self.screen, self.object_mask, self.traffic_car_image, self.object_car_image,
                         2 * self.bgWidth,
                         2 * self.bgHeight)

    def run(self, human_control=True):
        super().run()
        # initialize traffic
        if self.traffic is True:
            self.init_traffic_cars()

        rs_pos_list = [[6, 27, 0.0], [5, 27, 180.0], [4, 24, 180.0], [4, 23, 0.0], [5, 27, 90.0], [5, 27, 0.0]]

        # boolean variable needed to check for single-click press
        mouse_button_pressed = False

        # ----------------------------- print_activations -------------------------------------
        if self.print_activations is True:
            desired_layer_output = "convolution0"
            layer_names, image_buf, state_buf, activation_model = self.initialize_activation_model(desired_layer_output)
        # ----------------------------------------

        if self.record_data is True:
            index_image = 0

        while not self.exit:
            # VARIABLE_UPDATE
            if self.traffic is True:
                collision_list = [False] * len(self.traffic_list)

            self.dt = self.clock.get_time() / 1000
            self.event_handler(mouse_button_pressed)

            # LOGIC
            if human_control is True:
                self.key_handler(self.dt, rs_pos_list)

            # DRAWING
            stage_pos = self.draw_sim_environment(print_coords=True)
            stage_pos = (stage_pos[0], stage_pos[1])

            # UPDATE
            # ------------------------ traffic car -----------------------------------------------
            if self.traffic is True:
                self.check_collisions(collision_list)
                self.traffic_movement(collision_list, stage_pos)
            # -------------------------------------------------------------------------------------------
            self.car.update(self.dt)

            self.activate_sensors()

            # -------------------------------------------print_activations----------------------------------------
            if self.print_activations is True:
                image_rect = pygame.Rect((390, 110), (500, 500))
                sub = self.screen.subsurface(image_rect)
                self.input_image = pygame.surfarray.array3d(sub)

                image_buf[0] = self.input_image
                activations = activation_model.predict([image_buf, state_buf])
                print_activations(activations, layer_names, desired_layer_output)
            # ----------------------------------------------------------

            # CUSTOM FUNCTION TAB
            self.custom()

            # RECORD TAB
            if self.record_data is True:
                self.record_data_function(index_image)

            pygame.display.update()
            self.clock.tick(self.ticks)

        pygame.quit()


if __name__ == '__main__':
    screen = pygame.display.set_mode((1280, 720))
    city_simulator = CitySimulator(screen, 1280, 720)
    city_simulator.run()
