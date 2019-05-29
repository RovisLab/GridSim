from car_kinematic_model import Simulator
from car import Car
import pygame
import random
import numpy as np
from agent_functions import AgentAccelerationPattern, GridSimScenario


class StateEstimatorKinematicModel(Simulator):
    def __init__(self, screen, screen_width, screen_height, num_cars, max_veh_vel, base_velocity,
                 scenario=GridSimScenario.FOLLOW_LEFT_BEHIND_CATCH_UP):
        # choose your backgrounds
        object_map_path = "resources/backgrounds/highway_fixed_obj.png"
        background_path = "resources/backgrounds/highway_fixed_bigger.png"
        car_image_path = "resources/cars/car_eb_2.png"
        traffic_car_image_path = "resources/cars/car_traffic.png"
        object_car_image_path = "resources/cars/object_car.png"
        car_x = 5
        car_y = 27

        super(StateEstimatorKinematicModel, self).__init__(screen, screen_width, screen_height, car_x, car_y, 50, 8, False,
                                                           False, None, None, True, True, False, object_map_path,
                                                           background_path, car_image_path, traffic_car_image_path,
                                                           object_car_image_path)
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

    def _get_available_fov_vehicles(self):
        x = np.arange(31, 46, 3)
        available_positions_in_fov = list()
        y = np.arange(30, 80, 4 * self.traffic_safe_space + self.car_image.get_width())
        for xx in x:
            for yy in y:
                position = (xx, yy)
                available_positions_in_fov.append(position)
        return available_positions_in_fov

    def init_highway_traffic(self):
        available_traffic_car_positions = self._get_available_fov_vehicles()
        for idx in range(self.num_cars):
            pos = available_traffic_car_positions[random.randint(0, len(available_traffic_car_positions) - 1)]
            available_traffic_car_positions.remove(pos)
            traffic_car = Car(pos[0], pos[1],
                              None, 0.0, 4, 30, self.max_veh_vel,
                              AgentAccelerationPattern(AgentAccelerationPattern.SINUSOIDAL))
            traffic_car.angle = -90
            traffic_car.include_next_lane_mechanic = True
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
        if self.scenario == GridSimScenario.FOLLOW_LEFT_BEHIND_CATCH_UP:
            num_sin_cycles = self.cycle_num // self.highway_traffic[0].acc_pattern.get_num_samples()
            if 0 <= num_sin_cycles <= 3:
                self.car.follow(self.highway_traffic[0])
            elif 3 < num_sin_cycles <= 6:
                self.car.stay_behind(self.highway_traffic[0], self.dt)
            elif 6 < num_sin_cycles <= 10:
                self.car.catch_up(self.highway_traffic[0])
            else:
                self.exit = True

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

            # update the car behaviour
            prev_pos = [self.car.position.x, self.car.position.y]

            self.update_traffic()
            self.scenario_handler()
            self.car.update(self.dt)
            diff_pos = [0, self.car.position.y - prev_pos[1]]
            self.correct_drawing(diff_pos)

            # check the sensors for activations
            self.activate_sensors()

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
        distance = self.car.position.x - self.initial_car_position - self.traffic_offset_value
        pos_x = (traffic_car.position[0] * self.ppu)
        pos_x += distance * self.ppu

        pos_y = (traffic_car.position[1] * self.ppu)
        pos_y = self.screen_height - pos_y
        return pos_x, pos_y

    def draw_highway_traffic(self):
        for traffic_car in self.highway_traffic:
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
    game = StateEstimatorKinematicModel(screen=screen, screen_width=w, screen_height=h, num_cars=1, max_veh_vel=20,
                                        base_velocity=10)
    game.run()
