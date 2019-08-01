from multiple_cars_scenario_model import MultipleCarsSimulator
import os
import pygame
import pygame.gfxdraw
from read_write_trajectory import read_traffic_data
from traffic_car import TrafficCar
from action_handler import apply_action, Actions


class MultipleCarsCity(MultipleCarsSimulator):

    def __init__(self, positions_list, screen_width, screen_height, cars_nr=4, sensor_size=50, rays_nr=8,
                 traffic=True, record_data=False, replay_data_path=None, state_buf_path=None,
                 sensors=False, distance_sensor=False, all_cars_visual_sensors=False):
        object_map_path = "resources/backgrounds/maps_overlay_obj.png"
        background_path = "resources/backgrounds/maps_overlay.png"
        traffic_car_image_path = "resources/cars/car_traffic.png"
        object_car_image_path = "resources/cars/object_car.png"
        super().__init__(positions_list, screen_width, screen_height, cars_nr,
                         sensor_size, rays_nr, record_data, replay_data_path, state_buf_path, sensors, distance_sensor,
                         all_cars_visual_sensors, object_map_path, background_path, traffic_car_image_path,
                         object_car_image_path)

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
                         2 * self.bgHeight,
                         self.screen_width,
                         self.screen_height)

    def custom(self, *args):
        super().custom(args)

    def examples(self, *args):
        super().custom()
        car_tags = args[0]
        example = args[1]

        # USAGE EXAMPLES
        # EXAMPLE 1
        # control all 4 cars example with direct actions:
        # if you want to control the current car too, you have to place all the inputs in between the key_handler and
        # the update tab
        if example == 'example_1':
            for tag in car_tags:
                if tag == self.current_car:
                    self.kinematic_cars[tag].accelerate(1)
                else:
                    self.kinematic_cars[tag].accelerate(self.dt)

        # EXAMPLE 2
        # control all 4 cars example with indirect actions:
        # see action_handler.Actions for more info
        if example == 'example_2':
            for tag in car_tags:
                if tag == self.current_car:
                    apply_action(0, self.kinematic_cars[tag], 1)
                else:
                    apply_action(0, self.kinematic_cars[tag], self.dt)

        # EXAMPLE 3
        # example on how to get a specific sensor mask for a specific car
        if example == 'example_3':
            test_mask = self.get_sensor_mask_for_car(self.kinematic_cars['car_2'])
            if test_mask is not None:
                self.screen.blit(test_mask, (0, 0))

        # EXAMPLE 4
        # how to access data from the car/cars
        if example == 'example_4':
            car_data, visual_data, distance_data = self.access_simulator_data(car_tags, car_data_bool=True,
                                                                              visual_sensor_data_bool=True)
            visual_data_resized = []
            if type(visual_data) == list:
                for image in visual_data:
                    image = pygame.transform.scale(image, (100, 100))
                    visual_data_resized.append(image)

                if visual_data_resized.__len__() > 0:
                    self.screen.blit(visual_data_resized[0], (10, 10))
                    self.screen.blit(visual_data_resized[1], (110, 10))
            else:
                visual_data_resized = pygame.transform.scale(visual_data, (100, 100))
                self.screen.blit(visual_data_resized, (10, 10))

    def run(self, car_tags=[]):
        super().run()

        # initialize cars
        self.init_kinematic_cars()

        # initialize traffic
        if self.traffic is True:
            self.init_traffic_cars()

        rs_pos_list = [[6, 27, 0.0], [5, 27, 180.0], [4, 24, 180.0], [4, 23, 0.0], [5, 27, 90.0], [5, 27, 0.0]]

        # boolean variable needed to check for single-click press
        mouse_button_pressed = False

        if self.record_data is True:
            index_image = 0

        while not self.exit:
            # VARIABLE_UPDATE
            if self.traffic is True:
                collision_list = [False] * len(self.traffic_list)

            self.dt = self.clock.get_time() / 1000
            self.event_handler(mouse_button_pressed)

            # determine the current car, in case you need to control a specific one from the pool
            self.determine_current_car()
            car = self.kinematic_cars[self.current_car]

            # LOGIC
            self.key_handler(car, self.dt, rs_pos_list)

            # FOR EXAMPLE 1 AND 2
            # self.examples(car_tags, 'example_')  # run examples 1, 2 from here

            # DRAWING
            stage_pos = self.draw_sim_environment(car, print_coords=True, print_other_cars_coords=False)
            stage_pos = (stage_pos[0], stage_pos[1])

            # UPDATE
            # ------------------------ traffic car -----------------------------------------------
            if self.traffic is True:
                self.check_collisions(collision_list)
                self.traffic_movement(collision_list, stage_pos)
            # -------------------------------------------------------------------------------------------
            # update the current and the others cars
            car.update(self.dt)
            self.update_cars()

            self.activate_sensors(car)
            self.activate_sensors_for_all_cars()

            # EXAMPLES FUNCTION TAB -> CHECK FUNCTION FOR EXAMPLES
            # self.examples(car_tags, 'example_4')  # run examples 3, 4 from here

            self.custom()

            # RECORD TAB
            if self.record_data is True:
                self.record_data_function(car_tags, index_image)

            pygame.display.update()
            self.clock.tick(self.ticks)

        pygame.quit()


if __name__ == '__main__':
    # give positions with angle if necessary
    postions_list = [(5, 27), (25, 27), (5, 45, -90), (-15, 27, -180)]
    city_simulator = MultipleCarsCity(positions_list=postions_list, screen_width=1280, screen_height=720, traffic=False,
                                      cars_nr=2, all_cars_visual_sensors=False)
    city_simulator.run(['car_1', 'car_2'])
