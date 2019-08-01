import os
import pygame
import pygame.gfxdraw
from math import radians
from checkbox import Checkbox
from collision import Collision
from car import Car
from math_util import *
from read_write_trajectory import write_data, write_state_buf
import ctypes
from mpc_development import MPCController
import math
from math import sin, cos


class MultipleCarsSimulator:
    def __init__(self, positions_list, screen_w=1280, screen_h=720, cars_nr=4,
                 sensor_size=50,
                 rays_nr=8,
                 record_data=False,
                 replay_data_path=None,
                 state_buf_path=None,
                 sensors=False,
                 distance_sensor=False,
                 all_cars_visual_sensors=False,
                 # relative paths to the current folder
                 object_map_path=None,
                 background_path=None,
                 traffic_car_image_path=None,
                 object_car_image_path=None):

        pygame.init()
        self.screen_width = screen_w
        self.screen_height = screen_h
        self.screen = pygame.display.set_mode((screen_w, screen_h))

        self.bkd_color = [255, 255, 0, 255]

        self.current_dir = os.path.dirname(os.path.abspath(__file__))

        self.positions_list = positions_list
        self.cars_nr = cars_nr
        self.kinematic_cars = {}

        if traffic_car_image_path is not None:
            self.traffic_image_path = os.path.join(self.current_dir, traffic_car_image_path)
            self.traffic_car_image = pygame.image.load(self.traffic_image_path).convert_alpha()
        else:
            self.traffic_car_image = None

        if object_car_image_path is not None:
            self.object_car_image_path = os.path.join(self.current_dir, object_car_image_path)
            self.object_car_image = pygame.image.load(self.object_car_image_path).convert_alpha()
        else:
            self.object_car_image = None

        if object_map_path is not None:
            self.object_map_path = os.path.join(self.current_dir, object_map_path)
            self.object_map = pygame.image.load(self.object_map_path).convert_alpha()
        else:
            self.object_map = None

        self.background_path = os.path.join(self.current_dir, background_path)
        self.background = pygame.image.load(self.background_path).convert()

        self.bgWidth, self.bgHeight = self.background.get_rect().size

        pygame.font.init()
        self.used_font = pygame.font.SysFont('Arial', 30)

        # start automatically with car_1
        self.current_car = "car_1"
        self.global_cars_positions = {}
        self.sensors = sensors
        self.distance_sensor = distance_sensor
        self.all_cars_visual_sensors = all_cars_visual_sensors
        self.sensor_size = sensor_size
        self.rays_nr = rays_nr
        self.rays_sensor_distances = None
        self.sensor_mask = pygame.Surface((screen_w, screen_h))
        self.object_mask = pygame.Surface((screen_w, screen_h))
        self.sensor_masks = {}

        self.record_data = record_data

        self.replay_data_path = replay_data_path
        self.state_buf_path = state_buf_path

        self.ppu = 16
        self.exit = False
        self.clock = pygame.time.Clock()
        self.ticks = 60
        self.dt = None

        self.cbox_front_sensor = Checkbox(screen_w - 200, 10, 'Enable front sensor', self.sensors, (0, 255, 75))
        self.cbox_rear_sensor = Checkbox(screen_w - 200, 35, 'Enable rear sensor', self.sensors, (0, 255, 75))
        self.cbox_distance_sensor = Checkbox(screen_w - 200, 60, 'Enable distance sensor',
                                             self.distance_sensor, (0, 255, 75))
        self.cbox_all_cars_visual_sensors = Checkbox(screen_w - 200, 85, 'Enable all cars sensors',
                                                     self.all_cars_visual_sensors, (0, 255, 75))

        self.car1_checkbox = Checkbox(10, 180, "Car1", True, (0, 255, 75))
        if self.cars_nr >= 2:
            self.car2_checkbox = Checkbox(10, 210, "Car2", False, (0, 255, 75))
        else:
            self.car2_checkbox = None
        if self.cars_nr >= 3:
            self.car3_checkbox = Checkbox(10, 240, "Car3", False, (0, 255, 75))
        else:
            self.car3_checkbox = None
        if self.cars_nr == 4:
            self.car4_checkbox = Checkbox(10, 270, "Car4", False, (0, 255, 75))
        else:
            self.car4_checkbox = None

        # MPC VARIABLES
        # currently implemented only for 2 cars
        self.mpc_input_data_car1 = None
        self.mpc_trajectory_points_car1 = []
        self.mpc_delta_car1 = 0
        self.mpc_acc_car1 = 0
        self.mpc_angle_car1 = 0
        self.mpc_coords_car1 = [0, 0]

        self.mpc_input_data_car2 = None
        self.mpc_trajectory_points_car2 = []
        self.mpc_delta_car2 = 0
        self.mpc_acc_car2 = 0
        self.mpc_angle_car2 = 0
        self.mpc_coords_car2 = [0, 0]

        self.prev_ref_index_car1 = 40
        self.prev_ref_index_car2 = 40

    def init_kinematic_cars(self):
        for position, car_index in zip(self.positions_list, range(1, self.cars_nr + 1)):
            car_tag = "car_" + str(car_index)
            car_image_path = self.current_dir + '/resources/cars/' + car_tag + '.png'
            car_image = pygame.image.load(car_image_path).convert_alpha()
            car_image = pygame.transform.scale(car_image, (42, 20))
            car = Car(position[0], position[1], car_tag=car_tag, car_image=car_image)
            car.max_velocity = 10
            if len(position) == 3:
                car.angle = position[2]
            self.kinematic_cars[car_tag] = car

    def on_road(self, car, screen):
        Ox = 32
        Oy = 16
        center_world_x = int(self.screen_width / 2)
        center_world_y = int(self.screen_height / 2)

        bot_right_x = center_world_x + int(Ox * cos(radians(-car.angle))) - int(Oy * sin(radians(-car.angle)))
        bot_right_y = center_world_y + int(Ox * sin(radians(-car.angle))) + int(Oy * cos(radians(-car.angle)))

        bot_left_x = center_world_x - int(Ox * cos(radians(-car.angle))) - int(Oy * sin(radians(-car.angle)))
        bot_left_y = center_world_y - int(Ox * sin(radians(-car.angle))) + int(Oy * cos(radians(-car.angle)))

        top_left_x = center_world_x - int(Ox * cos(radians(-car.angle))) + int(Oy * sin(radians(-car.angle)))
        top_left_y = center_world_y - int(Ox * sin(radians(-car.angle))) - int(Oy * cos(radians(-car.angle)))

        top_right_x = center_world_x + int(Ox * cos(radians(-car.angle))) + int(Oy * sin(radians(-car.angle)))
        top_right_y = center_world_y + int(Ox * sin(radians(-car.angle))) - int(Oy * cos(radians(-car.angle)))

        if (np.array_equal(screen.get_at((bot_right_x, bot_right_y)), self.bkd_color) or np.array_equal
            (screen.get_at((bot_left_x, bot_left_y)), self.bkd_color) or
                np.array_equal(screen.get_at((top_left_x, top_left_y)), self.bkd_color) or
                np.array_equal(screen.get_at((top_right_x, top_right_y)), self.bkd_color)):
            Collision.offroad(car)
            return False
        else:
            return True

    @staticmethod
    def compute_end_point(side, base_point, sensor_length, sensor_angle, car_angle):
        if side is 'front':
            end_point_x = base_point[0] + sensor_length * cos(radians(sensor_angle - car_angle))
            end_point_y = base_point[1] + sensor_length * sin(radians(sensor_angle - car_angle))
        elif side is 'rear':
            end_point_x = base_point[0] - sensor_length * cos(radians(sensor_angle - car_angle))
            end_point_y = base_point[1] - sensor_length * sin(radians(sensor_angle - car_angle))
        else:
            raise ValueError("Side not defined.")
        return end_point_x, end_point_y

    def compute_collision_point(self, side, base_point, sensor_length, sensor_angle, car_angle, data_screen, draw_screen,
                                end_point):
        if side is 'front':
            for index in range(0, sensor_length):
                coll_point_x = base_point[0] + index * cos(radians(sensor_angle - car_angle))
                coll_point_y = base_point[1] + index * sin(radians(sensor_angle - car_angle))

                try:
                    if np.array_equal(data_screen.get_at((int(coll_point_x), int(coll_point_y))), self.bkd_color):
                        break
                except:
                    pass

            pygame.draw.line(draw_screen, (0, 255, 0), base_point, (coll_point_x, coll_point_y), True)
            pygame.draw.line(draw_screen, (255, 0, 0), (coll_point_x, coll_point_y), (end_point[0], end_point[1]), True)
        elif side is 'rear':
            for index in range(0, sensor_length):
                coll_point_x = base_point[0] - index * cos(radians(sensor_angle - car_angle))
                coll_point_y = base_point[1] - index * sin(radians(sensor_angle - car_angle))

                try:
                    if np.array_equal(data_screen.get_at((int(coll_point_x), int(coll_point_y))), self.bkd_color):
                        break
                except:
                    pass

            pygame.draw.line(draw_screen, (0, 255, 0), base_point, (coll_point_x, coll_point_y), True)
            pygame.draw.line(draw_screen, (255, 0, 0), (coll_point_x, coll_point_y), (end_point[0], end_point[1]), True)
        else:
            raise ValueError("Side not defined.")

        return coll_point_x, coll_point_y

    def compute_sensor_distance(self, car, base_point, sensor_length, sensor_angle, data_screen, draw_screen, side=None):

        end_point = self.compute_end_point(side, base_point, sensor_length, sensor_angle, car.angle)
        coll_point = self.compute_collision_point(side, base_point, sensor_length, sensor_angle, car.angle, data_screen,
                                                  draw_screen, end_point)

        distance = euclidean_norm(base_point, coll_point)
        # print(distance)

        return distance

    def enable_front_sensor(self, car, draw_screen, rays_nr):
        """
        front distance sensor
        :param car:
        :param data_screen:
        :param draw_screen:
        :param rays_nr:
        :return:
        """

        position = self.global_cars_positions[car.car_tag]
        center = Collision.calculate_center_for_car(car, position)
        if car.car_tag == self.current_car:
            center = Collision.center_rect(self.screen_width, self.screen_height)
        mid_of_front_axle = Collision.point_rotation(car, 0, 16, center)

        distance = np.array([])
        for angle_index in range(120, 240, int(round(120/rays_nr))):
            distance = np.append(distance,
                                 self.compute_sensor_distance(car, mid_of_front_axle, 200, angle_index, self.object_mask,
                                                              draw_screen, side='front'))
        return distance

    def enable_rear_sensor(self, car, draw_screen, rays_nr):
        """
        rear distance sensor
        :param car:
        :param data_screen:
        :param draw_screen:
        :param rays_nr:
        :return:
        """

        position = self.global_cars_positions[car.car_tag]
        center = Collision.calculate_center_for_car(car, position)
        if car.car_tag == self.current_car:
            center = Collision.center_rect(self.screen_width, self.screen_height)
        mid_of_rear_axle = Collision.point_rotation(car, 65, 16, center)

        distance = np.array([])
        for angle_index in range(120, 240, int(round(120/rays_nr))):
            distance = np.append(distance,
                                 self.compute_sensor_distance(car, mid_of_rear_axle, 200, angle_index, self.object_mask,
                                                              draw_screen, side='rear'))
        return distance

    def optimized_front_sensor(self, car, act_mask, display_obstacle_on_sensor=False):
        """
        front visual sensor
        :param act_mask:
        :param display_obstacle_on_sensor:
        :return:
        """
        # act_mask is a separate image where you can only see what the sensor sees
        position = self.global_cars_positions[car.car_tag]
        center = Collision.calculate_center_for_car(car, position)
        if car.car_tag == self.current_car:
            center = Collision.center_rect(self.screen_width, self.screen_height)
        mid_of_front_axle = Collision.point_rotation(car, -1, 16, center)

        arc_points = get_arc_points(mid_of_front_axle, 150, radians(90 + car.angle), radians(270 + car.angle),
                                    self.sensor_size)

        draw_center = Collision.center_rect(self.screen_width, self.screen_height)
        draw_mid_front_axle = Collision.point_rotation(car, -1, 16, draw_center)
        draw_arc_points = get_arc_points(draw_mid_front_axle, 150, radians(90 + car.angle), radians(270 + car.angle),
                                         self.sensor_size)

        offroad_edge_points = []
        draw_offroad_edge_points = []

        for end_point, draw_end_point in zip(arc_points, draw_arc_points):
            points_to_be_checked = list(get_equidistant_points(mid_of_front_axle, end_point, 25))
            draw_points_to_be_checked = list(get_equidistant_points(draw_mid_front_axle, draw_end_point, 25))

            check = False

            for line_point, draw_line_point in zip(points_to_be_checked, draw_points_to_be_checked):
                try:
                    if np.array_equal(self.object_mask.get_at((int(line_point[0]), int(line_point[1]))), self.bkd_color):
                        check = True
                        break
                except:
                    check = True
                    break

            if check is False:
                offroad_edge_points.append(end_point)
                draw_offroad_edge_points.append(draw_end_point)
            else:
                offroad_edge_points.append(line_point)
                draw_offroad_edge_points.append(draw_line_point)

        for index in range(0, len(arc_points)):
            if offroad_edge_points[index] == arc_points[index]:
                pygame.draw.line(self.screen, (0, 255, 0), mid_of_front_axle, arc_points[index], True)
                pygame.draw.line(act_mask, (0, 255, 0), draw_mid_front_axle, draw_arc_points[index], True)
            else:
                pygame.draw.line(self.screen, (0, 255, 0), mid_of_front_axle, offroad_edge_points[index], True)
                pygame.draw.line(act_mask, (0, 255, 0), draw_mid_front_axle, draw_offroad_edge_points[index], True)
                if display_obstacle_on_sensor is True:
                    pygame.draw.line(self.screen, (255, 0, 0), offroad_edge_points[index], arc_points[index], True)
                    pygame.draw.line(act_mask, (255, 0, 0), draw_offroad_edge_points[index], draw_arc_points[index], True)

    def optimized_rear_sensor(self, car, act_mask, display_obstacle_on_sensor=False):
        """
        rear visual sensor
        :param car:
        :param object_mask:
        :param act_mask:
        :param display_obstacle_on_sensor:
        :return:
        """
        # act_mask is a separate image where you can only see what the sensor sees
        position = self.global_cars_positions[car.car_tag]
        center = Collision.calculate_center_for_car(car, position)
        if car.car_tag == self.current_car:
            center = Collision.center_rect(self.screen_width, self.screen_height)
        mid_of_rear_axle = Collision.point_rotation(car, 65, 16, center)

        arc_points = get_arc_points(mid_of_rear_axle, 150, radians(-90 + car.angle), radians(90 + car.angle),
                                    self.sensor_size)

        draw_center = Collision.center_rect(self.screen_width, self.screen_height)
        draw_mid_rear_axle = Collision.point_rotation(car, 65, 16, draw_center)
        draw_arc_points = get_arc_points(draw_mid_rear_axle, 150, radians(-90 + car.angle), radians(90 + car.angle),
                                         self.sensor_size)

        offroad_edge_points = []
        draw_offroad_edge_points = []

        for end_point, draw_end_point in zip(arc_points, draw_arc_points):
            points_to_be_checked = list(get_equidistant_points(mid_of_rear_axle, end_point, 25))
            draw_points_to_be_checked = list(get_equidistant_points(draw_mid_rear_axle, draw_end_point, 25))

            check = False

            for line_point, draw_line_point in zip(points_to_be_checked, draw_points_to_be_checked):
                try:
                    if np.array_equal(self.object_mask.get_at((int(line_point[0]), int(line_point[1]))), self.bkd_color):
                        check = True
                        break
                except:
                    check = True
                    break

            if check is False:
                offroad_edge_points.append(end_point)
                draw_offroad_edge_points.append(draw_end_point)
            else:
                offroad_edge_points.append(line_point)
                draw_offroad_edge_points.append(draw_line_point)

        for index in range(0, len(arc_points)):
            if offroad_edge_points[index] == arc_points[index]:
                pygame.draw.line(self.screen, (0, 255, 0), mid_of_rear_axle, arc_points[index], True)
                pygame.draw.line(act_mask, (0, 255, 0), draw_mid_rear_axle, draw_arc_points[index], True)
            else:
                pygame.draw.line(self.screen, (0, 255, 0), mid_of_rear_axle, offroad_edge_points[index], True)
                pygame.draw.line(act_mask, (0, 255, 0), draw_mid_rear_axle, draw_offroad_edge_points[index], True)
                if display_obstacle_on_sensor is True:
                    pygame.draw.line(self.screen, (255, 0, 0), offroad_edge_points[index], arc_points[index], True)
                    pygame.draw.line(act_mask, (255, 0, 0), draw_offroad_edge_points[index], draw_arc_points[index], True)

    def custom_drawing(self, *args):
        """
        custom drawing function
        add drawings to simulator
        :param args: arguments
        :return:
        """
        pass

    def draw_other_kinematic_cars(self, stagePosX, stagePosY, debug=False):
        current_car = self.kinematic_cars[self.current_car]
        self.global_cars_positions[self.current_car] = (stagePosX, stagePosY)
        current_position = current_car.position
        if debug is True:
            pos_list = []

        for car in self.kinematic_cars:
            car_to_draw = self.kinematic_cars[car]
            if car_to_draw.car_tag != self.current_car:
                draw_pos_x = (car_to_draw.position.x * self.ppu - stagePosX) + car_to_draw.car_image.get_height()/2
                draw_pos_y = (car_to_draw.position.y * self.ppu - stagePosY) + car_to_draw.car_image.get_width()/2

                x = self.screen_width / 2 - draw_pos_x
                y = self.screen_height / 2 - draw_pos_y

                self.global_cars_positions[car_to_draw.car_tag] = (x, y)

                if debug is True:
                    pos_list.append((car_to_draw.car_tag, car_to_draw.position, x, y))

                rotated = pygame.transform.rotate(car_to_draw.car_image, car_to_draw.angle)
                obj_rotated = pygame.transform.rotate(self.object_car_image, car_to_draw.angle)
                self.screen.blit(rotated, (x, y))
                self.object_mask.blit(obj_rotated, (x, y))

        if debug is True:
            text0 = self.used_font.render('current_car_pos ' + str(current_position), True, (250, 0, 0))
            text1 = self.used_font.render(str(pos_list[0]), True, (250, 0, 0))
            text2 = self.used_font.render(str(pos_list[1]), True, (250, 0, 0))
            text3 = self.used_font.render(str(pos_list[2]), True, (250, 0, 0))
            self.screen.blit(text0, (200, 170))
            self.screen.blit(text1, (200, 200))
            self.screen.blit(text2, (200, 230))
            self.screen.blit(text3, (200, 260))

    def draw_sim_environment(self, car, print_coords=False, print_other_cars_coords=False):
        # Drawing
        """
        principal draw function that builds the simulator environment
        :param print_coords: print_coors on screen bool
        :param print_other_cars_coords=False
        :return:
        """
        stagePosX = car.position[0] * self.ppu
        stagePosY = car.position[1] * self.ppu

        rel_x = stagePosX % self.bgWidth
        rel_y = stagePosY % self.bgHeight

        self.object_mask.blit(self.object_map, (rel_x - self.bgWidth, rel_y - self.bgHeight))
        self.object_mask.blit(self.object_map, (rel_x, rel_y))
        self.object_mask.blit(self.object_map, (rel_x - self.bgWidth, rel_y))
        self.object_mask.blit(self.object_map, (rel_x, rel_y - self.bgHeight))

        self.screen.blit(self.background, (rel_x - self.bgWidth, rel_y - self.bgHeight))
        self.screen.blit(self.background, (rel_x, rel_y))
        self.screen.blit(self.background, (rel_x - self.bgWidth, rel_y))
        self.screen.blit(self.background, (rel_x, rel_y - self.bgHeight))

        self.draw_other_kinematic_cars(stagePosX, stagePosY, debug=print_other_cars_coords)

        if self.car1_checkbox is not None:
            self.car1_checkbox.update()
        if self.car2_checkbox is not None:
            self.car2_checkbox.update()
        if self.car3_checkbox is not None:
            self.car3_checkbox.update()
        if self.car4_checkbox is not None:
            self.car4_checkbox.update()

        if self.cbox_front_sensor is not None:
            self.cbox_front_sensor.update()
        if self.cbox_rear_sensor is not None:
            self.cbox_rear_sensor.update()
        if self.cbox_distance_sensor is not None:
            self.cbox_distance_sensor.update()
        if self.cbox_all_cars_visual_sensors is not None:
            self.cbox_all_cars_visual_sensors.update()
        if self.cbox_all_cars_visual_sensors.isChecked() is True:
            self.all_cars_visual_sensors = True
        else:
            self.all_cars_visual_sensors = False
        if self.cbox_front_sensor.isChecked() is True or self.cbox_rear_sensor.isChecked() is True:
            self.sensors = True
        else:
            self.sensors = False
        if self.cbox_distance_sensor.isChecked() is True:
            self.distance_sensor = True
        else:
            self.distance_sensor = False

        rotated = pygame.transform.rotate(car.car_image, car.angle)
        rotated_obj = pygame.transform.rotate(self.object_car_image, car.angle)
        rot_rect = rotated.get_rect()

        center_x = int(self.screen_width / 2) - int(rot_rect.width / 2)
        center_y = int(self.screen_height / 2) - int(rot_rect.height / 2)

        # draw the ego car
        self.screen.blit(rotated, (center_x, center_y))
        self.object_mask.blit(rotated_obj, (center_x, center_y))
        self.custom_drawing()

        if print_coords is True:
            text1 = self.used_font.render('Car pos x: ' + str(round(stagePosX, 2)), True, (250, 0, 0))
            text2 = self.used_font.render('Car pos y: ' + str(round(stagePosY, 2)), True, (250, 0, 0))
            text3 = self.used_font.render('rel x: ' + str(round(rel_x, 2)), True, (250, 0, 0))
            text4 = self.used_font.render('rel y: ' + str(round(rel_y, 2)), True, (250, 0, 0))
            text5 = self.used_font.render('velocity: ' + str(round(car.velocity.x, 2) * self.ppu/4) + ' km/h', True, (250, 0, 0))

            self.screen.blit(text1, (20, 20))
            self.screen.blit(text2, (20, 50))
            self.screen.blit(text3, (20, 80))
            self.screen.blit(text4, (20, 110))
            self.screen.blit(text5, (20, 140))

        return stagePosX, stagePosY, rel_x, rel_y

    def key_handler(self, car, dt, rs_pos_list):
        # User input
        """
        key handler that coordinates the car movement with user keyboard input
        :param car:
        :param dt:
        :param rs_pos_list:
        :return:
        """
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_ESCAPE]:
            quit()
        if pressed[pygame.K_r]:
            car.reset_car(rs_pos_list)
        if pressed[pygame.K_UP]:
            car.accelerate(dt)
        elif pressed[pygame.K_DOWN]:
            car.brake(dt)
        elif pressed[pygame.K_SPACE]:
            car.handbrake(dt)
        else:
            car.cruise(dt)
        if pressed[pygame.K_RIGHT]:
            car.steer_right(dt)
        elif pressed[pygame.K_LEFT]:
            car.steer_left(dt)
        else:
            car.no_steering()

    @staticmethod
    def change_state_for_other_checkboxes(checkbox_list):

        for checkbox in checkbox_list:
            if checkbox is not None and checkbox.isChecked():
                checkbox.changeState()

    def change_car(self, mouse_button_pressed, mouse_pos):

        if self.car1_checkbox.onCheckbox(mouse_pos) and mouse_button_pressed is False:
            self.car1_checkbox.changeState()
            checkbox_list = [self.car2_checkbox, self.car3_checkbox, self.car4_checkbox]
            self.change_state_for_other_checkboxes(checkbox_list)

        if self.cars_nr >= 2:
            if self.car2_checkbox.onCheckbox(mouse_pos) and mouse_button_pressed is False:
                self.car2_checkbox.changeState()
                checkbox_list = [self.car1_checkbox, self.car3_checkbox, self.car4_checkbox]
                self.change_state_for_other_checkboxes(checkbox_list)

        if self.cars_nr >= 3:
            if self.car3_checkbox.onCheckbox(mouse_pos) and mouse_button_pressed is False:
                self.car3_checkbox.changeState()
                checkbox_list = [self.car1_checkbox, self.car2_checkbox, self.car4_checkbox]
                self.change_state_for_other_checkboxes(checkbox_list)

        if self.cars_nr == 4:
            if self.car4_checkbox.onCheckbox(mouse_pos) and mouse_button_pressed is False:
                self.car4_checkbox.changeState()
                checkbox_list = [self.car1_checkbox, self.car2_checkbox, self.car3_checkbox]
                self.change_state_for_other_checkboxes(checkbox_list)

    def event_handler(self, mouse_button_pressed):
        # Event queue
        """
        event handler for sensors check_boxes, exit event or mouse pressing events
        :param mouse_button_pressed:
        :return:
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.exit = True
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                self.change_car(mouse_button_pressed, mouse_pos)
                if self.cbox_front_sensor.onCheckbox(mouse_pos) and mouse_button_pressed is False:
                    self.cbox_front_sensor.changeState()
                if self.cbox_rear_sensor.onCheckbox(mouse_pos) and mouse_button_pressed is False:
                    self.cbox_rear_sensor.changeState()
                if self.cbox_distance_sensor.onCheckbox(mouse_pos) and mouse_button_pressed is False:
                    self.cbox_distance_sensor.changeState()
                if self.cbox_all_cars_visual_sensors.onCheckbox(mouse_pos) and mouse_button_pressed is False:
                    self.cbox_all_cars_visual_sensors.changeState()

                mouse_button_pressed = True
            if event.type == pygame.MOUSEBUTTONUP:
                mouse_button_pressed = False

    def access_simulator_data(self, car_tags, car_data_bool=False, visual_sensor_data_bool=False,
                              distance_sensor_data_bool=False):
        """
        car_data: acceleration -> car_data[0] ; steering -> car_data[1] ; angle -> car_data[2] ; velocity -> car_data[3]
        sensor_data: subsurface with sensor image
        :param car_tags: tags of cars
        :param car_data_bool: if you want to access car_data set to true
        :param visual_sensor_data_bool: if you want to access sensor_data set to true
        :param distance_sensor_data_bool: if you want to access rays_sensor_data set to true and check the cbox in the simulator
        for rays_sensor.
        :return:
        """

        _car_data = []
        if self.sensors is True:
            _visual_data = None
        elif self.all_cars_visual_sensors is True:
            _visual_data = []
        else:
            _visual_data = []
        _distance_data = []

        for tag in car_tags:
            if tag in self.kinematic_cars:
                car = self.kinematic_cars[tag]
                if car_data_bool is True:
                    car_acc = car.acceleration
                    car_steering = car.steering
                    car_angle = car.angle
                    car_velocity = car.velocity.x
                    car_data = [car_acc, car_steering, car_velocity, car_angle]
                    _car_data.append(car_data)
                if visual_sensor_data_bool is True:
                    if self.sensors is True:
                        image_rect = pygame.Rect((390, 110), (500, 500))
                        sub = self.sensor_mask.subsurface(image_rect)
                        _visual_data = sub
                    if self.all_cars_visual_sensors is True:
                        sub = self.sensor_masks[tag]
                        _visual_data.append(sub)
                if distance_sensor_data_bool is True:
                    if self.distance_sensor is True:
                        if self.rays_sensor_distances is not None:
                            _distance_data.append(self.rays_sensor_distances)
            else:
                raise ValueError('car_tag not found, please specify a valid car tag')
        return _car_data, _visual_data, _distance_data

    def update_cars(self):

        for car in self.kinematic_cars:
            if self.kinematic_cars[car].car_tag != self.current_car:
                self.kinematic_cars[car].update(self.dt)

    def determine_current_car(self):

        if self.car1_checkbox.isChecked():
            self.current_car = "car_1"
        elif self.cars_nr >= 2 and self.car2_checkbox.isChecked():
            self.current_car = "car_2"
        elif self.cars_nr >= 3 and self.car3_checkbox.isChecked():
            self.current_car = "car_3"
        elif self.cars_nr == 4 and self.car4_checkbox.isChecked():
            self.current_car = "car_4"
        else:
            self.current_car = "car_1"

    def get_sensor_mask_for_car(self, car):
        """
        Access a specific sensor mask when all the sensors are active.
        all_cars_sensors has to be True.
        :param car:
        :return:
        """
        if car.car_tag in self.sensor_masks:
            return self.sensor_masks[car.car_tag]

    def activate_sensors_for_all_cars(self):

        self.rays_sensor_distances = []
        for car in self.kinematic_cars:
            if self.cbox_all_cars_visual_sensors.isChecked():
                # temp_sensor_mask = pygame.Surface((self.screen_width, self.screen_height))
                # self.optimized_front_sensor(self.kinematic_cars[car], temp_sensor_mask, display_obstacle_on_sensor=True)
                # self.optimized_rear_sensor(self.kinematic_cars[car], temp_sensor_mask, display_obstacle_on_sensor=True)
                # image_rect = pygame.Rect((440, 160), (400, 400))
                # sensor_sub = temp_sensor_mask.subsurface(image_rect)
                # self.sensor_masks[car] = sensor_sub
                self.rays_sensor_distances.append([self.enable_front_sensor(self.kinematic_cars[car], self.screen, self.rays_nr),
                                                   self.enable_rear_sensor(self.kinematic_cars[car], self.screen,
                                                                           self.rays_nr)])

    def activate_sensors(self, car):
        """
        Check if any sensor has been activated.
        :return:
        """
        self.sensor_mask.fill((0, 0, 0))
        if self.cbox_front_sensor.isChecked():
            self.optimized_front_sensor(car, self.sensor_mask, display_obstacle_on_sensor=True)
        if self.cbox_rear_sensor.isChecked():
            self.optimized_rear_sensor(car, self.sensor_mask, display_obstacle_on_sensor=True)
        if self.cbox_distance_sensor.isChecked():
            if self.cbox_rear_sensor.isChecked() is False and self.cbox_front_sensor.isChecked() is False:
                self.rays_sensor_distances = [self.enable_front_sensor(car, self.screen, self.rays_nr),
                                              self.enable_rear_sensor(car, self.screen, self.rays_nr)]

    def record_data_function(self, car_tags, index):
        """
        Data recording.
        :param car_tags: tags of the cars you want to record
        :param index: image index
        :return:
        """
        image_name = 'image_' + str(index) + '.png'
        index += 1

        # check if the car tags exists
        for tag in car_tags:
            if tag in self.kinematic_cars:
                car = self.kinematic_cars[tag]
                if self.state_buf_path is None:
                    raise OSError('state_buf_path is empty.')
                if self.replay_data_path is None:
                    raise OSError('replay_data_path is empty.')

                actions = [car.position.x, car.position.y, float(round(car.angle, 3)),
                           float(round(car.acceleration, 3)),
                           float(round(car.velocity.x, 3)), image_name]

                # Save state_buf
                write_state_buf(self.state_buf_path + '/' + str(tag) + '_state_buf' + '.csv', actions)

                # Save replay
                write_data(self.replay_data_path + '/' + str(tag) + '_replay' + '.csv', car.position, car.angle)
            else:
                raise ValueError('given car_tag does not exists')

    def custom(self, *args):
        """
        custom function in which to modify or access data
        if you want to create a simulation on top of another simulation but modify some things but not the run function
        you can add this custom() function inside of your run function and override it in the child simulator, but do
        not override the parent run() function.
        :param args: custom arguments if needed
        :return:
        """
        pass

    def run(self):
        """
        main run loop
        :return:
        """
        pass

    """
    MPC FUNCTIONS
    """

    @staticmethod
    def rotate_x_point(point_x, point_y, angle):
        return point_x * cos(np.deg2rad(angle)) - point_y * sin(np.deg2rad(angle))

    @staticmethod
    def rotate_y_point(point_x, point_y, angle):
        return point_x * sin(np.deg2rad(angle)) + point_y * cos(np.deg2rad(angle))

    def draw_mpc_prediction(self, *args, mpc_solution, car_tag):
        """

        :param args: args[0] -> mpc_traj_points_carX, args[1] -> mpc_angle_carX
        :param mpc_solution:
        :param car_tag:
        :return:
        """
        if car_tag == self.current_car:
            center_screen = (int(self.screen_width / 2), int(self.screen_height / 2))
        else:
            position = self.global_cars_positions[car_tag]
            center_screen = Collision.calculate_center_for_car(self.kinematic_cars[car_tag], position)

        args[0].clear()
        for index in range(2, 20, 2):
            delta_position = (
                 mpc_solution[index] * cos(np.deg2rad(args[1])) + mpc_solution[index + 1] * sin(np.deg2rad(args[1])),
                 mpc_solution[index] * (-sin(np.deg2rad(args[1]))) + mpc_solution[index + 1] * cos(np.deg2rad(args[1])))
            x_point = center_screen[0] - int(delta_position[0] * self.ppu)
            y_point = center_screen[1] - int(delta_position[1] * self.ppu)
            traj_point = (x_point, y_point)
            args[0].append(traj_point)

    def prepare_mpc_input(self, car, waypoints):
        if car.car_tag == 'car_1':
            self.mpc_input_data_car1 = (ctypes.c_double * 14)()
            for index in range(6):
                self.mpc_input_data_car1[index*2] = waypoints[index][0]
            for index in range(6):
                self.mpc_input_data_car1[index*2+1] = waypoints[index][1]
            self.mpc_input_data_car1[12] = np.deg2rad(car.angle)
            self.mpc_input_data_car1[13] = car.velocity[0]
        elif car.car_tag == 'car_2':
            self.mpc_input_data_car2 = (ctypes.c_double * 14)()
            for index in range(6):
                self.mpc_input_data_car2[index * 2] = waypoints[index][0]
            for index in range(6):
                self.mpc_input_data_car2[index * 2 + 1] = waypoints[index][1]
            self.mpc_input_data_car2[12] = np.deg2rad(car.angle)
            self.mpc_input_data_car2[13] = car.velocity[0]
        else:
            raise ValueError("Car tag not defined.")

    def draw_trajectory(self, car, car_data):
        # draw trajectory
        """

        :param car:
        :param car_data:
        :return:
        """
        if car.car_tag == self.current_car:
            center_screen = (int(self.screen_width / 2), int(self.screen_height / 2))
        else:
            position = self.global_cars_positions[car.car_tag]
            center_screen = Collision.calculate_center_for_car(car, position)

        trajectory_points = []
        waypoints = []
        min = 1000
        idx = -1
        if car.car_tag == 'car_1':
            prev_ref_index = self.prev_ref_index_car1
        elif car.car_tag == 'car_2':
            prev_ref_index = self.prev_ref_index_car2
        else:
            raise ValueError("Car tag not defined")

        for elem in range(prev_ref_index-40, prev_ref_index+40):
            dx = car_data[elem][0] - car.position[0]
            dy = car_data[elem][1] - car.position[1]
            d = abs(math.sqrt(dx**2+dy**2))
            if d < min:
                min = d
                idx = elem
                prev_ref_index = idx

        if car.car_tag == 'car_1':
            self.prev_ref_index_car1 = prev_ref_index
        elif car.car_tag == 'car_2':
            self.prev_ref_index_car2 = prev_ref_index
        else:
            raise ValueError("Car tag not defined")

        for add_elem in range(idx, idx + 150, 15):
            if add_elem < len(car_data):
                delta_position = (
                    car.position[0] - car_data[add_elem][0],
                    car.position[1] - car_data[add_elem][1])
                x_point = center_screen[0] + int(delta_position[0] * self.ppu)
                y_point = center_screen[1] + int(delta_position[1] * self.ppu)
                traj_point = (x_point, y_point)
                trajectory_points.append(traj_point)

                if len(waypoints) < 9:
                    waypoints.append((car_data[add_elem][0], car_data[add_elem][1]))

                # draw each trajectory point
                pygame.draw.circle(self.screen, (255, 255, 0), traj_point, 2, 2)

        # draw lines between trajectory points
        for traj_point, next_traj_point in zip(trajectory_points, trajectory_points[1:]):
            pygame.draw.aaline(self.screen, (255, 255, 0), traj_point, next_traj_point, 10)

        self.prepare_mpc_input(car, waypoints)
        if car.car_tag == 'car_1':
            if len(self.mpc_trajectory_points_car1) > 0:
                for traj_point, next_traj_point in zip(self.mpc_trajectory_points_car1, self.mpc_trajectory_points_car1[1:]):
                    pygame.draw.aaline(self.screen, (0, 255, 0), traj_point, next_traj_point, 10)
        elif car.car_tag == 'car_2':
            if len(self.mpc_trajectory_points_car2) > 0:
                for traj_point, next_traj_point in zip(self.mpc_trajectory_points_car2, self.mpc_trajectory_points_car2[1:]):
                    pygame.draw.aaline(self.screen, (0, 255, 0), traj_point, next_traj_point, 10)

    def mpc_thread(self, mpc_target_speed=30, mpc_dt=0.1):
        controller = MPCController(target_speed=mpc_target_speed, dt=mpc_dt)
        while True:
            mpc_solution_car1 = controller.control(self.mpc_input_data_car1, self.mpc_coords_car1)
            mpc_solution_car2 = controller.control(self.mpc_input_data_car2, self.mpc_coords_car2)
            self.mpc_delta_car1 = mpc_solution_car1[0]
            self.mpc_acc_car1 = mpc_solution_car1[1]
            self.mpc_delta_car2 = mpc_solution_car2[0]
            self.mpc_acc_car2 = mpc_solution_car2[1]
            self.draw_mpc_prediction(self.mpc_trajectory_points_car1, self.mpc_angle_car1, mpc_solution=mpc_solution_car1,
                                     car_tag='car_1')
            self.draw_mpc_prediction(self.mpc_trajectory_points_car2, self.mpc_angle_car2, mpc_solution=mpc_solution_car2,
                                     car_tag='car_2')

