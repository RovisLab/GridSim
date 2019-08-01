import pygame
import pygame.gfxdraw
from multiple_cars_city import MultipleCarsCity
from multiple_cars_highway import MultipleCarsHighwaySimulator
import threading
import numpy as np
from read_write_trajectory import read_coords


class MA_DL_NMPC_city(MultipleCarsCity):

    def __init__(self, positions_list, screen_width, screen_height, cars_nr=4, sensor_size=50, rays_nr=8, traffic=True,
                 record_data=False, replay_data_path=None, state_buf_path=None, sensors=False, distance_sensor=False,
                 all_cars_visual_sensors=False):
        super().__init__(positions_list, screen_width, screen_height, cars_nr, sensor_size, rays_nr, traffic,
                         record_data, replay_data_path, state_buf_path, sensors, distance_sensor,
                         all_cars_visual_sensors)

    def run(self, car_tags=[]):
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

        """
        TEMP
        """
        self.mpc_coords_car1[0] = self.kinematic_cars['car_1'].position[0]
        self.mpc_coords_car1[1] = self.kinematic_cars['car_1'].position[1]
        self.mpc_coords_car2[0] = self.kinematic_cars['car_2'].position[0]
        self.mpc_coords_car2[1] = self.kinematic_cars['car_2'].position[1]
        print(self.mpc_coords_car1)
        print(self.mpc_coords_car2)

        MUL_FACTOR = 1
        car1_data = read_coords("resources/replay_car_1.csv")
        car2_data = read_coords("resources/replay_car_2.csv")
        car1_data = np.multiply(car1_data, MUL_FACTOR)
        car2_data = np.multiply(car2_data, MUL_FACTOR)

        thread = threading.Thread(target=self.mpc_thread, args=())
        thread.start()

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
            # self.custom(car_tags, 'example_')  # run examples 1, 2 from here

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
            self.kinematic_cars['car_1'].steering = np.rad2deg(self.mpc_delta_car1)
            self.kinematic_cars['car_2'].steering = np.rad2deg(self.mpc_delta_car2)

            self.kinematic_cars['car_1'].acceleration = self.mpc_acc_car1
            self.kinematic_cars['car_2'].acceleration = self.mpc_acc_car2

            car.update(self.dt)
            self.update_cars()

            self.draw_trajectory(self.kinematic_cars['car_1'], car1_data)
            self.draw_trajectory(self.kinematic_cars['car_2'], car2_data)

            self.mpc_coords_car1[0] = self.kinematic_cars['car_1'].position[0]
            self.mpc_coords_car1[1] = self.kinematic_cars['car_1'].position[1]
            self.mpc_coords_car2[0] = self.kinematic_cars['car_2'].position[0]
            self.mpc_coords_car2[1] = self.kinematic_cars['car_2'].position[1]
            self.mpc_angle_car1 = self.kinematic_cars['car_1'].angle
            self.mpc_angle_car2 = self.kinematic_cars['car_2'].angle

            self.activate_sensors(car)
            self.activate_sensors_for_all_cars()

            # CUSTOM FUNCTION TAB -> CHECK FUNCTION FOR EXAMPLES
            # self.custom(car_tags, 'example_4')  # run examples 3, 4 from here

            # RECORD TAB
            if self.record_data is True:
                self.record_data_function(car_tags, index_image)

            pygame.display.update()
            self.clock.tick(self.ticks)

        pygame.quit()


class MA_DL_NMPC_highway(MultipleCarsHighwaySimulator):

    def __init__(self, positions_list, screen_width, screen_height, cars_nr=4, sensor_size=50, rays_nr=8,
                 highway_traffic=True, highway_traffic_cars_nr=5, record_data=False, replay_data_path=None,
                 state_buf_path=None, sensors=False, distance_sensor=False, all_cars_visual_sensors=False,
                 ego_car_collisions=True, traffic_collisions=True):
        super().__init__(positions_list, screen_width, screen_height, cars_nr, sensor_size, rays_nr, highway_traffic,
                         highway_traffic_cars_nr, record_data, replay_data_path, state_buf_path, sensors,
                         distance_sensor, all_cars_visual_sensors, ego_car_collisions, traffic_collisions)

    def run(self):

        rs_pos_list = [[6, 27, 0.0], [5, 27, 180.0], [4, 24, 180.0], [4, 23, 0.0], [5, 27, 90.0], [5, 27, 0.0]]

        # boolean variable needed to check for single-click press
        mouse_button_pressed = False

        if self.record_data is True:
            index_image = 0

        """
        TEMP
        """
        self.mpc_coords_car1[0] = self.kinematic_cars['car_1'].position[0]
        self.mpc_coords_car1[1] = self.kinematic_cars['car_1'].position[1]
        self.mpc_coords_car2[0] = self.kinematic_cars['car_2'].position[0]
        self.mpc_coords_car2[1] = self.kinematic_cars['car_2'].position[1]

        MUL_FACTOR = 1
        car1_data = read_coords("resources/highway_replay_car_1.csv")
        car2_data = read_coords("resources/highway_replay_car_2.csv")
        car1_data = np.multiply(car1_data, MUL_FACTOR)
        car2_data = np.multiply(car2_data, MUL_FACTOR)

        thread = threading.Thread(target=self.mpc_thread, args=())
        thread.start()

        while not self.exit:
            # VARIABLE_UPDATE

            self.dt = self.clock.get_time() / 1000
            self.event_handler(mouse_button_pressed)

            # determine the current car, in case you need to control a specific one from the pool
            self.determine_current_car()
            car = self.kinematic_cars[self.current_car]

            # LOGIC
            self.key_handler(car, self.dt, rs_pos_list)

            # DRAWING
            stage_pos = self.draw_sim_environment(car, print_coords=True, print_other_cars_coords=False)
            stage_pos = (stage_pos[0], stage_pos[1])

            self.kinematic_cars['car_1'].steering = np.rad2deg(self.mpc_delta_car1)
            self.kinematic_cars['car_2'].steering = np.rad2deg(self.mpc_delta_car2)

            self.kinematic_cars['car_1'].acceleration = self.mpc_acc_car1
            self.kinematic_cars['car_2'].acceleration = self.mpc_acc_car2

            # UPDATE
            car.update(self.dt)
            self.update_cars()

            self.draw_trajectory(self.kinematic_cars['car_1'], car1_data)
            self.draw_trajectory(self.kinematic_cars['car_2'], car2_data)

            self.mpc_coords_car1[0] = self.kinematic_cars['car_1'].position[0]
            self.mpc_coords_car1[1] = self.kinematic_cars['car_1'].position[1]
            self.mpc_coords_car2[0] = self.kinematic_cars['car_2'].position[0]
            self.mpc_coords_car2[1] = self.kinematic_cars['car_2'].position[1]
            self.mpc_angle_car1 = self.kinematic_cars['car_1'].angle
            self.mpc_angle_car2 = self.kinematic_cars['car_2'].angle

            self.avoid_collisions()
            self.generate_new_cars()
            self.update_highway_traffic()
            self.correct_traffic()

            self.activate_sensors(car)
            self.activate_sensors_for_all_cars()

            # CUSTOM FUNCTION TAB FOR FURTHER ADDITIONS
            self.custom()

            # RECORD TAB
            if self.record_data is True:
                self.record_data_function(index_image)

            pygame.display.update()
            self.clock.tick(self.ticks)

        pygame.quit()


class MA_DL_NMPC():
    def __init__(self, city=False, highway=False):
        if city is False and highway is False:
            raise ValueError("Type of simulator not specified.")
        elif city is True:
            self.simulator = MA_DL_NMPC_city(positions_list=[(5, 27), (25, 27), (5, 45, -90), (-15, 27, -180)],
                                             screen_width=1280, screen_height=720, traffic=False,
                                             cars_nr=2, all_cars_visual_sensors=False, rays_nr=20)
        elif highway is True:
            self.simulator = MA_DL_NMPC_highway([(3, 27), (6, 27)], 1280, 720, highway_traffic_cars_nr=5, cars_nr=2)

    def run(self):
        self.simulator.run()


if __name__ == '__main__':
    ma_dl_nmpc = MA_DL_NMPC(highway=True)
    ma_dl_nmpc.run()


