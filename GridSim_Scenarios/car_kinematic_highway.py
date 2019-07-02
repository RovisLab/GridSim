from car_kinematic_model import Simulator
from car import Car
import copy
from print_activations import print_activations
import pygame
import numpy as np
import random
from agent_functions import AgentAction, Obstacle, GridSimScenario, AgentAccelerationPattern


class HighwaySimulator(Simulator):

    def __init__(self, screen, screen_width, screen_height, scenario, car_x=3, car_y=27, sensor_size=50, rays_nr=8,
                 activations=False, highway_traffic=True, record_data=False, replay_data_path=None, state_buf_path=None, sensors=False,
                 distance_sensor=False, enabled_menu=False, highway_traffic_cars_nr=5, ego_car_collisions=False, traffic_collisions=False):
        object_map_path = "resources/backgrounds/highway_fixed_obj.png"
        background_path = "resources/backgrounds/highway_fixed_bigger.png"
        car_image_path = "resources/cars/car_eb_2.png"
        traffic_car_image_path = "resources/cars/car_traffic.png"
        object_car_image_path = "resources/cars/object_car.png"
        super().__init__(screen, screen_width, screen_height, car_x, car_y, sensor_size, rays_nr, activations,
                         record_data, replay_data_path, state_buf_path, sensors, distance_sensor, enabled_menu,
                         object_map_path, background_path, car_image_path, traffic_car_image_path,
                         object_car_image_path)

        self.crt_car_id = 0

        self.car_image = pygame.transform.scale(self.car_image, (42, 20))
        self.traffic_car_image = pygame.transform.scale(self.traffic_car_image, (42, 20))
        self.object_car_image = pygame.transform.scale(self.object_car_image, (42, 20))
        self.object_map = pygame.transform.scale(self.object_map, (self.bgWidth, self.bgHeight))
        self.traffic_offset_value = 3 - car_x

        self.car.angle = -90
        self.initial_car_position = car_x
        self.traffic_cars_nr = highway_traffic_cars_nr
        self.traffic_safe_space = 25
        self.scenario = scenario
        self.highway_traffic = []
        if highway_traffic is True:
            self.init_highway_traffic()

        self.ego_car_collisions = ego_car_collisions
        self.traffic_collisions = traffic_collisions

        self.traffic_accidents = 0
        self.ego_car_accidents = 0
        self.rs_pos_list = [[6, 27, 0.0], [5, 27, 180.0], [4, 24, 180.0], [4, 23, 0.0], [5, 27, 90.0], [5, 27, 0.0]]
        self.obstacles = list()

    def find_out_drawing_coordinates_highway_traffic(self, traffic_car):
        distance = self.car.position.x - self.initial_car_position - self.traffic_offset_value
        pos_x = (traffic_car.position[0] * self.ppu) % self.bgWidth
        pos_x += distance * self.ppu

        pos_y = (traffic_car.position[1] * self.ppu) % self.bgHeight
        pos_y = self.screen_height - pos_y
        if pos_y > self.screen_height:
            pos_y = -(self.screen_height - pos_y + self.bgHeight - self.screen_height)

        return pos_x, pos_y

    def update_highway_traffic(self):
        if self.scenario.scenario_type == GridSimScenario.FOLLOW_LEFT_BEHIND_CATCH_UP:
            for traffic_car in self.highway_traffic:
                if self.car.velocity.x < traffic_car.max_velocity:
                    traffic_car.accelerate(self.dt)
                elif self.car.velocity.x > traffic_car.max_velocity:
                    traffic_car.brake(self.dt)
                else:
                    traffic_car.cruise(self.dt)
                traffic_car.update(self.dt)
                '''base_speed = traffic_car.max_velocity / 2
                acc_val = traffic_car.acc_pattern.get_current_acc()
                if traffic_car.velocity.x < base_speed:
                    traffic_car.accelerate(self.dt)
                else:
                    if acc_val > 0 and traffic_car.velocity.x < traffic_car.max_velocity:
                        traffic_car.accelerate(self.dt)
                    elif acc_val < 0 and traffic_car.velocity.x > base_speed:
                        traffic_car.brake(self.dt)
                    else:
                        traffic_car.cruise(self.dt)'''
                traffic_car.update(self.dt)
        else:
            for traffic_car in self.highway_traffic:
                '''if self.car.velocity.x < traffic_car.max_velocity:
                    traffic_car.accelerate(self.dt)
                elif self.car.velocity.x > traffic_car.max_velocity:
                    traffic_car.brake(self.dt)
                else:
                    traffic_car.cruise(self.dt)'''
                traffic_car.update(self.dt)

    def draw_highway_traffic(self):
        for traffic_car in self.highway_traffic:
            if traffic_car.hide_car is False:
                pos_x, pos_y = self.find_out_drawing_coordinates_highway_traffic(traffic_car)
                car_img = pygame.transform.rotate(self.traffic_car_image, traffic_car.angle)
                car_obj_img = pygame.transform.rotate(self.object_car_image, traffic_car.angle)
                self.screen.blit(car_img, (pos_x, pos_y))
                self.object_mask.blit(car_obj_img, (pos_x, pos_y))

    def map_relative_traffic_positions(self):

        relative_positions = []
        for traffic_car in self.highway_traffic:
            pos_y = (traffic_car.position.y * self.ppu) % self.bgHeight
            relative_positions.append((traffic_car.position.x, pos_y))
        return relative_positions

    def map_available_lanes(self):

        relative_positions = self.map_relative_traffic_positions()
        occupied_lanes = [pos[0] for pos in relative_positions]
        lanes = [31, 34, 37, 40, 43, 46, 49, 52]

        for elem in occupied_lanes:
            if elem in lanes:
                lanes.remove(elem)

        # lanes = free lanes
        if len(lanes) != 0:
            return lanes
        else:
            # get lanes that are not occupied in bottom part of the screen
            free_lanes = []
            for pos in relative_positions:
                if pos[1] > 200:
                    if pos[0] in free_lanes:
                        pass
                    else:
                        free_lanes.append(pos[0])
                else:
                    if pos[0] in free_lanes:
                        free_lanes.remove(pos[0])
                    else:
                        pass
            return free_lanes

    def generate_new_cars(self):
        index = 0
        for traffic_car in self.highway_traffic:
            pos_y = (traffic_car.position.y * self.ppu) % self.bgHeight
            index += 1
            if pos_y >= 797 and traffic_car.acceleration > 0:
                print("Car ", index, " : ", pos_y)
                free_lanes = self.map_available_lanes()
                if len(free_lanes) > 0:
                    if len(free_lanes) > 1:
                        new_lane = free_lanes[random.randint(0, len(free_lanes) - 1)]
                    else:
                        new_lane = free_lanes[0]
                    traffic_car.position.x = new_lane

    def _generate_positions_in_fov(self):
        fov_points = self.get_sensors_points_distributions()
        fov_points = [fov_pt for fov_pt in fov_points if self._is_safe(fov_pt)]
        return fov_points

    def init_highway_traffic(self):
        # define lanes : (40, 4); (37, 3); (34, 2); (31, 1); (43, 5); (46, 6); (49, 7); (52, 8)
        if self.scenario.scenario_type == GridSimScenario.FOLLOW_LEFT_BEHIND_CATCH_UP:
            available_positions_in_fov = list()
            x = np.arange(31, 46, 3)
            y = np.arange(200, 500, 4 * self.traffic_safe_space + self.car_image.get_width())
            for xx in x:
                for yy in y:
                    position = (xx, yy)
                    available_positions_in_fov.append(position)
            for car_index in range(self.scenario.num_cars):

                car_pos = available_positions_in_fov[random.randint(0, len(available_positions_in_fov) - 1)]
                available_positions_in_fov.remove(car_pos)

                traffic_car = Car(x=car_pos[0], y=car_pos[1], onehot_encoding=self.generate_onehot_encoding(),
                                  acc_patern=AgentAccelerationPattern(mode=AgentAccelerationPattern.SINUSOIDAL),
                                  max_acceleration=15,
                                  max_velocity=15)
                traffic_car.angle = -90
                traffic_car.include_next_lane_mechanic = False
                self.highway_traffic.append(traffic_car)

        else:
            available_x = np.arange(31, 53, 3)
            available_y = np.arange(self.traffic_safe_space, self.screen_height, 4 * self.traffic_safe_space
                                    + self.car_image.get_width())
            available_positions = []
            for x in available_x:
                for y in available_y:
                    position = (x, y)
                    available_positions.append(position)

            random.shuffle(available_positions)

            for car_index in range(self.traffic_cars_nr):
                # generate random position
                car_position = available_positions[random.randint(0, len(available_positions) - 1)]
                # remove that position from the available position list
                available_positions.remove(car_position)
                # init traffic car and add it to traffic list
                traffic_car = Car(car_position[0], car_position[1], self.generate_onehot_encoding())
                traffic_car.angle = -90
                traffic_car.include_next_lane_mechanic = True
                traffic_car.max_velocity = random.randint(10, self.car.max_velocity)
                self.highway_traffic.append(traffic_car)

    def translate_ego_car_position_to_traffic_position(self):
        # we take the fifth lane as reference
        relative_position = int(43 - self.car.position.x)
        return relative_position

    def find_ego_car_lane(self):
        relative_position = self.translate_ego_car_position_to_traffic_position()
        lanes = [31, 34, 37, 40, 43, 46, 49, 52]
        distances = [np.linalg.norm(relative_position - lane) for lane in lanes]
        return lanes[distances.index(min(distances))]

    def draw_traffic_accident(self, traffic_car_1, traffic_car_2, draw_accident=False):
        if draw_accident is True:
            pos_x_1, pos_y_1 = self.find_out_drawing_coordinates_highway_traffic(traffic_car_1)
            pos_x_2, pos_y_2 = self.find_out_drawing_coordinates_highway_traffic(traffic_car_2)
            pygame.draw.rect(self.screen, (255, 0, 0),
                             pygame.Rect(pos_x_1, pos_y_1,
                                         self.traffic_car_image.get_height(), self.traffic_car_image.get_width()))
            pygame.draw.rect(self.screen, (255, 0, 0),
                             pygame.Rect(pos_x_2, pos_y_2,
                                         self.traffic_car_image.get_height(), self.traffic_car_image.get_width()))
            pygame.display.update()
        self.traffic_accidents += 1

    def draw_accident(self, traffic_car, draw_accident=False):
        if draw_accident is True:
            pos_x_1, pos_y_1 = self.find_out_drawing_coordinates_highway_traffic(traffic_car)
            if pos_y_1 < 0:
                print(pos_y_1)
            pygame.draw.rect(self.screen, (255, 0, 0),
                             pygame.Rect(630, 338,
                                         self.traffic_car_image.get_height(), self.traffic_car_image.get_width()))
            pygame.draw.rect(self.screen, (255, 0, 0),
                             pygame.Rect(pos_x_1, pos_y_1,
                                         self.traffic_car_image.get_height(), self.traffic_car_image.get_width()))
            pygame.display.update()
        self.ego_car_accidents += 1

    def accident_avoider(self):
        for traffic_car_1 in self.highway_traffic:
            for traffic_car_2 in self.highway_traffic:
                if traffic_car_1 is not traffic_car_2:
                    if traffic_car_1.position.x == traffic_car_2.position.x:
                        pos_y_1 = (traffic_car_1.position.y * self.ppu) % self.bgHeight
                        pos_y_2 = (traffic_car_2.position.y * self.ppu) % self.bgHeight
                        if abs(pos_y_1 - pos_y_2) + self.traffic_safe_space < self.traffic_car_image.get_width():
                            # immediate brake
                            if pos_y_1 <= pos_y_2:
                                traffic_car_1.brake(1)
                            else:
                                traffic_car_2.brake(1)

    def lane_check(self):
        possible_lanes = [31, 34, 37, 40, 43, 46, 49, 52]
        for traffic_car in self.highway_traffic:
            if traffic_car.position.x not in possible_lanes and traffic_car.next_lane is None:
                closest_lane = [np.linalg.norm(traffic_car.position.x - lane) for lane in possible_lanes]
                closest_lane = possible_lanes[closest_lane.index(min(closest_lane))]
                if closest_lane in possible_lanes:
                    traffic_car.next_lane = closest_lane

    def backwards_detector(self):
        if self.car.acceleration < 0:
            for traffic_car in self.highway_traffic:
                traffic_car.acceleration -= self.car.acceleration

    def correct_traffic(self):
        #self.backwards_detector()
        self.lane_check()
        self.accident_avoider()

    def check_traffic_car_proximity(self, traffic_car):
        if traffic_car.position.x == 52:
            empty_lanes = [traffic_car.position.x - 3]
        elif traffic_car.position.x == 31:
            empty_lanes = [traffic_car.position.x + 3]
        else:
            empty_lanes = [traffic_car.position.x - 3, traffic_car.position.x + 3]

        for _traffic_car in self.highway_traffic:
            if _traffic_car is not traffic_car:
                if _traffic_car.position.x in empty_lanes:
                    empty_lanes.remove(_traffic_car.position.x)

        ego_car_lane = self.find_ego_car_lane()
        if ego_car_lane in empty_lanes:
            empty_lanes.remove(ego_car_lane)

        if len(empty_lanes) > 0:
            return empty_lanes[0]

        if traffic_car.position.x == 52:
            empty_lanes = [traffic_car.position.x - 3]
        elif traffic_car.position.x == 31:
            empty_lanes = [traffic_car.position.x + 3]
        else:
            empty_lanes = [traffic_car.position.x - 3, traffic_car.position.x + 3]

        for _traffic_car in self.highway_traffic:
            if _traffic_car is not traffic_car:
                if _traffic_car.position.x in empty_lanes:
                    pos_y_1 = (_traffic_car.position.y * self.ppu) % self.bgHeight
                    pos_y_2 = (traffic_car.position.y * self.ppu) % self.bgHeight
                    if pos_y_2 - self.traffic_car_image.get_height() - 2 * self.traffic_safe_space <= pos_y_1 <= \
                            pos_y_2 + self.traffic_car_image.get_height() + 2 * self.traffic_safe_space:
                        empty_lanes.remove(_traffic_car.position.x)

        if ego_car_lane in empty_lanes:
            empty_lanes.remove(ego_car_lane)

        if len(empty_lanes) > 0:
            return empty_lanes[0]

        return None

    def avoid_traffic_collision(self, overtake_allowed):
        for traffic_car_1 in self.highway_traffic:
            pos_y_1 = (traffic_car_1.position.y * self.ppu) % self.bgHeight
            for traffic_car_2 in self.highway_traffic:
                # reference check
                if traffic_car_1 is not traffic_car_2:
                    if traffic_car_1.position.x == traffic_car_2.position.x:
                        pos_y_2 = (traffic_car_2.position.y * self.ppu) % self.bgHeight
                        if abs(pos_y_1 - pos_y_2) < self.traffic_safe_space * 2 + self.car_image.get_width():
                            if abs(pos_y_1 - pos_y_2) < self.traffic_car_image.get_width():
                                self.draw_traffic_accident(traffic_car_1, traffic_car_2, draw_accident=True)

                            if pos_y_1 > pos_y_2:
                                # car 1 in front
                                free_lane = self.check_traffic_car_proximity(traffic_car_2)

                                if overtake_allowed is True:
                                    if free_lane is not None:
                                            traffic_car_2.next_lane = free_lane
                                    else:
                                        traffic_car_2.brake(1)
                                        traffic_car_2.max_velocity = copy.copy(traffic_car_1.max_velocity)
                                        traffic_car_1.accelerate(1)
                                else:
                                    traffic_car_2.brake(1)
                                    traffic_car_2.max_velocity = copy.copy(traffic_car_1.max_velocity)
                                    traffic_car_1.accelerate(1)
                            elif pos_y_1 < pos_y_2:
                                # car 2 in front
                                free_lane = self.check_traffic_car_proximity(traffic_car_1)

                                if overtake_allowed is True:
                                    if free_lane is not None:
                                            traffic_car_1.next_lane = free_lane
                                    else:
                                        traffic_car_1.brake(1)
                                        traffic_car_1.max_velocity = copy.copy(traffic_car_2.max_velocity)
                                        traffic_car_2.accelerate(1)
                                else:
                                    traffic_car_1.brake(1)
                                    traffic_car_1.max_velocity = copy.copy(traffic_car_2.max_velocity)
                                    traffic_car_2.accelerate(1)
                            else:
                                traffic_car_1.accelerate(self.dt)
                                traffic_car_2.accelerate(self.dt)

    def avoid_ego_car_collision(self, overtake_allowed):
        # check cars in ego_car_proximity
        ego_car_lane = self.find_ego_car_lane()
        for traffic_car in self.highway_traffic:
            if traffic_car.position.x in range(ego_car_lane-1, ego_car_lane+1):
                pos_y = (traffic_car.position.y * self.ppu) % self.bgHeight
                # fixed values for better results
                if 250 <= pos_y <= 500:
                    # accident

                    if 330 <= pos_y <= 390:
                        self.draw_accident(traffic_car, draw_accident=True)

                    if self.screen_height - pos_y > 375:
                        # traffic car is behind ego_car -> brake to avoid collision
                        free_lane = self.check_traffic_car_proximity(traffic_car)

                        if overtake_allowed is True:
                            if free_lane is not None:
                                    if traffic_car.velocity.x > self.car.velocity.x:
                                        traffic_car.next_lane = free_lane
                            else:
                                traffic_car.brake(self.dt)
                        else:
                            traffic_car.brake(self.dt)
                    else:
                        # traffic car is in front of ego_car -> accelerate to avoid collision
                        traffic_car.accelerate(self.dt)

    def hide_traffic(self):
        for traffic_car in self.highway_traffic:
            pos = self.find_out_drawing_coordinates_highway_traffic(traffic_car)
            print('pos={0}'.format(pos))
            if pos[1] < 0:
                traffic_car.hide_car = True
                '''
                if traffic_car.velocity.x < self.car.velocity.x:
                    traffic_car.hide_car = False
                else:
                    traffic_car.hide_car = True
                '''

    def avoid_collisions(self, overtake_allowed):
        if self.ego_car_collisions is True:
            self.avoid_ego_car_collision(overtake_allowed)
        if self.traffic_collisions is True:
            self.avoid_traffic_collision(overtake_allowed)

    def update_accidents_count(self):
        myfont = pygame.font.SysFont('Arial', 25)
        text1 = myfont.render('Traffic accidents: ' + str(int(self.traffic_accidents/2)), True, (250, 0, 0))
        text2 = myfont.render('Ego car accidents: ' + str(int(self.ego_car_accidents/2)), True, (250, 0, 0))

        self.screen.blit(text1, (1070, 90))
        self.screen.blit(text2, (1070, 120))

    def custom_drawing(self, *args):
        super().custom_drawing(*args)
        self.draw_highway_traffic()
        self.update_accidents_count()

    def start_standby(self):
        super().run()

    def action_handler(self, action, dt):
        # User input
        """
        Action handler that coordinates the car movement with user provided action
        :param car:
        :param dt:
        :param rs_pos_list:
        :return:
        """

        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_ESCAPE]:
            self.return_to_menu()
            quit()

        if action == AgentAction.STAY:
            self.car.cruise(dt)
        elif action == AgentAction.RIGHT:
            self.car.steer_right(dt)
        elif action == AgentAction.LEFT:
            self.car.steer_left(dt)
        elif action == AgentAction.UP:
            self.car.accelerate(dt)
        elif action == AgentAction.DOWN:
            self.car.brake(dt)
        else:
            self.car.no_steering()

    def generate_onehot_encoding(self):
        arr = [0 for _ in range(25)]
        arr[self.crt_car_id] = 1
        self.crt_car_id += 1
        return np.array(arr).reshape((5, 5))

    def get_walls_in_fov(self):
        return self.obstacles

    def get_state(self):
        """
        Gets world state: car position, walls and moving obstacles
        :return: car_pos, walls, obstacles
        """
        # TODO fetch walls and obstacles (cars)
        cars = [Obstacle(onehot_encoding=car.onehot_encoding, global_position=car.position)
                for car in self.highway_traffic]
        return self.car.position, self.get_walls_in_fov(), cars

    def handle_action_and_get_observation(self, action, overtake_allowed):
        self.dt = AgentAction.ACTION_DELTA

        # boolean variable needed to check for single-click press
        mouse_button_pressed = False

        self.event_handler(mouse_button_pressed)
        self.action_handler(action, self.dt)

        # DRAWING
        self.draw_sim_environment(print_coords=True)

        # UPDATE
        self.car.update(self.dt)
        self.avoid_collisions(overtake_allowed=overtake_allowed)
        # self.generate_new_cars()
        self.update_highway_traffic()
        self.correct_traffic()
        self.obstacles = self.activate_sensors()

        # CUSTOM FUNCTION TAB FOR FURTHER ADDITIONS
        self.custom()

        pygame.display.update()
        return self.get_state()

    def run(self, hide_traffic=False, keep_traffic=False, overtake_allowed=True):
        """

        :param hide_traffic: when the traffic exits the main screen, it will not be re-drawn, it will wait for the
        ego-car to catch up
        :param keep_traffic: when this is true, the traffic positions will be kept
        :param overtake_allowed: allow cars to pass each other
        :return:
        """
        super().run()

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

            self.dt = self.clock.get_time() / 1000
            self.event_handler(mouse_button_pressed)

            # LOGIC
            self.key_handler(self.dt, rs_pos_list)

            # DRAWING
            self.draw_sim_environment(print_coords=True)

            # UPDATE
            self.car.update(self.dt)
            self.avoid_collisions(overtake_allowed)
            if keep_traffic is False:
                self.generate_new_cars()
            if hide_traffic is True:
                self.hide_traffic()
            self.update_highway_traffic()
            self.correct_traffic()

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

            # CUSTOM FUNCTION TAB FOR FURTHER ADDITIONS
            self.custom()

            # RECORD TAB
            if self.record_data is True:
                self.record_data_function(index_image)

            pygame.display.update()
            self.clock.tick(self.ticks)

        pygame.quit()


if __name__ == '__main__':
    screen = pygame.display.set_mode((1280, 720))
    highway_sim = HighwaySimulator(screen, 1280, 720, highway_traffic_cars_nr=1,
                                   scenario=GridSimScenario(num_cars=1,
                                                            scenario_type=GridSimScenario.FOLLOW_LEFT_BEHIND_CATCH_UP))
    highway_sim.run(keep_traffic=True, hide_traffic=True, overtake_allowed=False)

