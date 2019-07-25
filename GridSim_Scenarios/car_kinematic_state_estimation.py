from car_kinematic_model import Simulator
from car import Car
import pygame
import random
import numpy as np
import math
from agent_functions import AgentAccelerationPattern, GridSimScenario
import os
import csv
import datetime
from math import sin, cos, radians
from math_util import get_equidistant_points, get_arc_points, euclidean_norm


class TrainingDataType(object):
    SIMPLIFIED = 0
    SENSOR_RAYS = 1


class StateEstimatorKinematicModel(Simulator):
    def __init__(self, screen, screen_width, screen_height, num_cars, max_veh_vel, base_velocity,
                 scenario=GridSimScenario.USER_CONTROL_SINE,
                 mode=TrainingDataType.SIMPLIFIED):
        # choose your backgrounds
        object_map_path = "resources/backgrounds/highway_fixed_obj.png"
        background_path = "resources/backgrounds/highway_fixed_bigger.png"
        car_image_path = "resources/cars/car_eb_2.png"
        traffic_car_image_path = "resources/cars/car_traffic.png"
        object_car_image_path = "resources/cars/object_car.png"
        car_x = 0
        car_y = 27

        super(StateEstimatorKinematicModel, self).__init__(screen=screen, screen_width=screen_width,
                                                           screen_height=screen_height, car_x=car_x, car_y=car_y,
                                                           sensor_size=50, rays_nr=32,
                                                           activations=False, record_data=True, replay_data_path=None,
                                                           state_buf_path=os.path.join(os.path.dirname(__file__),
                                                                                       "resources",
                                                                                       "traffic_cars_data",
                                                                                       "state_estimation_data"),
                                                           sensors=False, distance_sensor=True, enabled_menu=False,
                                                           object_map_path=object_map_path,
                                                           background_path=background_path,
                                                           car_image_path=car_image_path,
                                                           traffic_car_image_path=traffic_car_image_path,
                                                           object_car_image_path=object_car_image_path)
        self.car_image = pygame.transform.scale(self.car_image, (42, 20))
        self.traffic_car_image = pygame.transform.scale(self.traffic_car_image, (42, 20))
        self.object_car_image = pygame.transform.scale(self.object_car_image, (42, 20))
        self.object_map = pygame.transform.scale(self.object_map, (self.bgWidth, self.bgHeight))
        self.traffic_offset_value = 3 - car_x

        self.car.angle = -90
        self.initial_car_position = car_x
        self.num_cars = num_cars
        self.highway_traffic = list()
        self.max_veh_vel = max_veh_vel
        self.traffic_safe_space = 25
        self.init_highway_traffic()
        self.position_delta = [0.0, 0.0]
        self.base_velocity = base_velocity
        self.scenario = scenario
        self.cycle_num = 0
        self.mode = mode

        self._rename_training_files()

    def _rename_training_files(self):
        if self.mode == TrainingDataType.SIMPLIFIED:
            if os.path.exists(os.path.join(self.state_buf_path, "tmp.npy")):
                parts = os.path.splitext(os.path.join(self.state_buf_path, "tmp.npy"))
                new_name = parts[0] + "_{0}".format(
                    str(datetime.datetime.now().time()).replace(":", "_").replace(".", "_")) + parts[1]
                os.rename(os.path.join(self.state_buf_path, "tmp.npy"), new_name)
        elif self.mode == TrainingDataType.SENSOR_RAYS:
            if os.path.exists(os.path.join(self.state_buf_path, "front_sensor_distances.npy")):
                idx = 0
                while os.path.exists(os.path.join(self.state_buf_path, "front_sensor_distances_{0}.npy".format(idx))):
                    idx += 1
                os.rename(os.path.join(self.state_buf_path, "front_sensor_distances.npy"),
                          os.path.join(self.state_buf_path, "front_sensor_distances_{0}.npy".format(idx)))
                os.rename(os.path.join(self.state_buf_path, "rear_sensor_distances.npy"),
                          os.path.join(self.state_buf_path, "rear_sensor_distances_{0}.npy".format(idx)))
                os.rename(os.path.join(self.state_buf_path, "velocity.npy"),
                          os.path.join(self.state_buf_path, "velocity_{0}.npy".format(idx)))

    def _get_available_fov_vehicles(self):
        x = np.arange(31, 52, 3)
        x = np.delete(x, np.where(x == 46))
        available_positions_in_fov = list()
        y = np.arange(35, 50, 2 * self.traffic_safe_space + self.car_image.get_width())
        for xx in x:
            for yy in y:
                position = (xx, yy)
                available_positions_in_fov.append(position)
        return available_positions_in_fov

    def _object_in_sensor_fov(self):
        max_dist = max(self.rays_sensor_distances)
        for dist in self.rays_sensor_distances:
            if not math.isclose(dist, max_dist):
                return True
        return False

    def init_highway_traffic(self):
        available_traffic_car_positions = self._get_available_fov_vehicles()
        for idx in range(self.num_cars):
            pos = available_traffic_car_positions[random.randint(0, len(available_traffic_car_positions) - 1)]
            available_traffic_car_positions.remove(pos)
            traffic_car = Car(pos[0], pos[1],
                              None, 0.0, 4, 30, self.max_veh_vel,
                              AgentAccelerationPattern(AgentAccelerationPattern.SINUSOIDAL))
            traffic_car.angle = -90
            traffic_car.include_next_lane_mechanic = False
            self.highway_traffic.append(traffic_car)

    def _adjust_velocity(self, traffic_car):
        crt_sine_value = traffic_car.acc_pattern.get_current_acc()
        if self.base_velocity <= traffic_car.velocity.x < self.max_veh_vel:
            traffic_car.accelerate_variable(self.dt, 0.8 * crt_sine_value * (self.max_veh_vel - self.base_velocity))
        elif traffic_car.velocity.x >= self.max_veh_vel:
            traffic_car.cruise(self.dt)

    def update_traffic(self):
        for traffic_car in self.highway_traffic:
            # Acceleration logic
            if traffic_car.velocity.x < self.base_velocity:
                traffic_car.accelerate(self.dt)
            else:
                self._adjust_velocity(traffic_car)
            traffic_car.update(self.dt)

    def correct_drawing(self, diff):
        for traffic_car in self.highway_traffic:
            traffic_car.position.x -= diff[0]
            traffic_car.position.y -= diff[1]

    def scenario_handler(self):
        num_sin_cycles = self.cycle_num // self.highway_traffic[0].acc_pattern.get_num_samples()
        if self.scenario == GridSimScenario.FOLLOW_LEFT_BEHIND_CATCH_UP:
            if 0 <= num_sin_cycles <= 3:
                self.car.follow(self.highway_traffic[0])
            elif 3 < num_sin_cycles <= 6:
                self.car.stay_behind(self.highway_traffic[0], self.dt)
            elif 6 < num_sin_cycles <= 10:
                self.car.overtake(self.highway_traffic[0])
            else:
                self.exit = True
        elif self.scenario == GridSimScenario.TRAIL_OVERTAKE_STOP:
            if 0 <= num_sin_cycles <= 3:
                self.car.follow(self.highway_traffic[0])
            elif 3 < num_sin_cycles <= 6:
                self.car.overtake(self.highway_traffic[0])
            elif 6 < num_sin_cycles <= 10:
                self.car.stop(self.dt)
            else:
                self.exit = True
        elif self.scenario == GridSimScenario.OVERTAKE_LEFT_BEHIND_OVERTAKE:
            if 0 <= num_sin_cycles <= 1:
                self.car.overtake(self.highway_traffic[0])
            elif 1 < num_sin_cycles <= 7:
                self.car.stay_behind(self.highway_traffic[0], self.dt, 10)
            elif 7 < num_sin_cycles <= 12:
                self.car.overtake(self.highway_traffic[0])
            else:
                self.exit = True
        elif self.scenario == GridSimScenario.BACK_AND_FORWARD:
            if num_sin_cycles > 12:
                self.exit = True
            if num_sin_cycles % 2 == 0:
                self.car.accelerate_to_speed(self.dt, self.car.max_velocity)
            else:
                self.car.stop(self.dt)
        elif self.scenario == GridSimScenario.USER_CONTROL_SINE:
            self.key_handler(dt=self.dt, rs_pos_list=[])

    def _write_data(self):
        if os.path.exists(self.state_buf_path) and os.path.isdir(self.state_buf_path):
            with open(os.path.join(self.state_buf_path, "tmp.npy"), "a") as tmp_f:
                '''actual_delta, perceived_delta, in_fov, velocity'''
                delta = abs(self.car.position.y - self.highway_traffic[0].position.y)
                tmp_f.write("{0},{1},{2},{3}\n".format(delta,
                                                       delta if self._object_in_sensor_fov() else 0.0,
                                                       1.0 if self._object_in_sensor_fov() else 0.0,
                                                       self.car.velocity.x))

    def _write_sensor_array_data(self):
        if os.path.exists(self.state_buf_path) and os.path.isdir(self.state_buf_path):
            with open(os.path.join(self.state_buf_path, "front_sensor_distances.npy"), "a") as tmp_ff:
                with open(os.path.join(self.state_buf_path, "rear_sensor_distances.npy"), "a") as tmp_fr:
                    for x in self.front_sensor_distances:
                        tmp_ff.write("{0},".format(x))
                    for x in self.rear_sensor_distances:
                        tmp_fr.write("{0},".format(x))
                    tmp_ff.write("\n")
                    tmp_fr.write("\n")
            with open(os.path.join(self.state_buf_path, "velocity.npy"), "a") as vel_f:
                vel_f.write("{0}\n".format(self.car.velocity.x))

    def record_data_function(self, index):
        if self.mode == TrainingDataType.SIMPLIFIED:
            self._write_data()
        elif self.mode == TrainingDataType.SENSOR_RAYS:
            self._write_sensor_array_data()

    def run(self):
        super().run()

        # we need this bool to check if the sensors are turned on
        # this field should be in any scenario
        mouse_button_pressed = False

        while not self.exit:
            # the flow should be split in 2 parts
            # first part should be the simulator construction which should be like this:
            # 1. CONSTRUCTION
            # update time
            self.dt = self.clock.get_time() / 1000
            # check the mouse click events
            self.event_handler(mouse_button_pressed)

            # take user input for our car
            # self.key_handler(self.dt, [])

            # draw the environment
            self.draw_sim_environment(print_coords=True)

            self.update_traffic()
            self.scenario_handler()
            self.car.update(self.dt)

            # check the sensors for activations
            self.activate_sensors()

            # save information from frames
            if self.record_data is True:
                self.record_data_function(self.cycle_num)

            # leave the custom function tab always open in case you want to add something from another simulator
            # that implements this simulator
            self.custom()

            # and in the last place update the screen and frames
            pygame.display.update()
            self.clock.tick(self.ticks)
            self.cycle_num += 1

        # when the user quits the simulator end the pygame process too
        pygame.quit()

    def find_out_drawing_coordinates_highway_traffic(self, traffic_car):
        ego_pos_x, ego_pos_y = self.car.position.x, self.car.position.y

        diff_y = (ego_pos_y - traffic_car.position.y)

        rotated = pygame.transform.rotate(self.car_image, self.car.angle)
        rot_rect = rotated.get_rect()

        center_x = int(self.screen_width / 2) - int(rot_rect.width / 2)
        center_y = int(self.screen_height / 2) - int(rot_rect.height / 2)
        return traffic_car.position.x * self.ppu, center_y + diff_y * self.ppu

    def _is_in_view(self, traffic_car):
        traffic_car_pos_x = traffic_car.position[0]
        traffic_car_pos_y = traffic_car.position[1]
        diff_y = abs(traffic_car_pos_y - self.car.position.y) * self.ppu
        return diff_y < self.screen_height / 2

    def draw_highway_traffic(self):
        for traffic_car in self.highway_traffic:
            if self._is_in_view(traffic_car):
                pos_x, pos_y = self.find_out_drawing_coordinates_highway_traffic(traffic_car)
                car_img = pygame.transform.rotate(self.traffic_car_image, traffic_car.angle)
                car_obj_img = pygame.transform.rotate(self.object_car_image, traffic_car.angle)
                self.screen.blit(car_img, (pos_x, pos_y))
                self.object_mask.blit(car_obj_img, (pos_x, pos_y))

    def custom_drawing(self, *args):
        super().custom(args)
        self.draw_highway_traffic()

if __name__ == "__main__":
    w, h = 1280, 768
    screen = pygame.display.set_mode((w, h))
    game = StateEstimatorKinematicModel(screen=screen, screen_width=w,
                                        screen_height=h, num_cars=1,
                                        max_veh_vel=20, base_velocity=10,
                                        scenario=GridSimScenario.USER_CONTROL_SINE,
                                        mode=TrainingDataType.SIMPLIFIED)
    game.run()
