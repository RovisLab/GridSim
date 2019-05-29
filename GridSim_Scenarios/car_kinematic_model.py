import os
import pygame
import pygame.gfxdraw
from math import radians
from checkbox import Checkbox
from collision import Collision
from car import Car
from math_util import *
from keras.models import load_model
from keras import models
from print_activations import init_activations_display_window
from read_write_trajectory import write_data, write_state_buf
import cv2


class Simulator:
    def __init__(self, screen, screen_width, screen_height,
                 car_x=5,
                 car_y=27,
                 sensor_size=50,
                 rays_nr=8,
                 activations=False,
                 record_data=False,
                 replay_data_path=None,
                 state_buf_path=None,
                 sensors=False,
                 distance_sensor=False,
                 enabled_menu=False,
                 # relative paths to the current folder
                 object_map_path=None,
                 background_path=None,
                 car_image_path=None,
                 traffic_car_image_path=None,
                 object_car_image_path=None):

        pygame.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = screen

        # The color of the object mask
        # Yellow
        self.bkd_color = [255, 255, 0, 255]

        self.current_dir = os.path.dirname(os.path.abspath(__file__))

        self.car = Car(x=car_x, y=car_y, max_velocity=25)
        if car_image_path is not None:
            self.car_image_path = os.path.join(self.current_dir, car_image_path)
            self.car_image = pygame.image.load(self.car_image_path).convert_alpha()
        else:
            self.car_image = None

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

        self.print_activations = activations
        self.model_path = os.path.join(self.current_dir, 'used_models/activations_model.h5')
        if os.path.exists(self.model_path) is False:
            raise OSError("model to path doesn't exists")

        self.background_path = os.path.join(self.current_dir, background_path)
        self.background = pygame.image.load(self.background_path).convert()

        self.bgWidth, self.bgHeight = self.background.get_rect().size

        pygame.font.init()
        self.used_font = pygame.font.SysFont('Comic Sans MS', 30)

        self.input_image = pygame.surfarray.array3d(self.screen)

        self.sensors = sensors
        self.distance_sensor = distance_sensor
        self.sensor_size = sensor_size
        self.rays_nr = rays_nr
        self.rays_sensor_distances = None
        self.sensor_mask = pygame.Surface((self.screen_width, self.screen_height))
        self.object_mask = pygame.Surface((self.screen_width, self.screen_height))

        self.record_data = record_data
        self.enabled_menu = enabled_menu

        self.replay_data_path = replay_data_path
        self.state_buf_path = state_buf_path

        self.ppu = 16
        self.exit = False
        self.clock = pygame.time.Clock()
        self.ticks = 60
        self.dt = None

        self.cbox_front_sensor = Checkbox(self.screen_width - 200, 10, 'Enable front sensor', self.sensors)
        self.cbox_rear_sensor = Checkbox(self.screen_width - 200, 35, 'Enable rear sensor', self.sensors)
        self.cbox_distance_sensor = Checkbox(self.screen_width - 200, 60, 'Enable distance sensor', self.distance_sensor)

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

    def compute_sensor_distance(self, car, base_point, sensor_length, sensor_angle, data_screen, draw_screen):
        end_point_x = base_point[0] + sensor_length * cos(radians(sensor_angle - car.angle))
        end_point_y = base_point[1] + sensor_length * sin(radians(sensor_angle - car.angle))

        for index in range(0, sensor_length):
            coll_point_x = base_point[0] + index * cos(radians(sensor_angle - car.angle))
            coll_point_y = base_point[1] + index * sin(radians(sensor_angle - car.angle))

            if np.array_equal(data_screen.get_at((int(coll_point_x), int(coll_point_y))), self.bkd_color):
                break

        pygame.draw.line(draw_screen, (0, 255, 0), base_point, (coll_point_x, coll_point_y), True)
        pygame.draw.line(draw_screen, (255, 0, 0), (coll_point_x, coll_point_y), (end_point_x, end_point_y), True)

        coll_point = (coll_point_x, coll_point_y)

        distance = euclidean_norm(base_point, coll_point)
        # print(distance)

        return distance

    def enable_sensor(self, car, draw_screen, rays_nr):
        """
        distance sensor
        :param car:
        :param data_screen:
        :param draw_screen:
        :param rays_nr:
        :return:
        """
        center_rect = Collision.center_rect(self.screen_width, self.screen_height)
        mid_of_front_axle = Collision.point_rotation(car, 0, 16, center_rect)
        distance = np.array([])
        for angle_index in range(120, 240, int(round(120/rays_nr))):
            distance = np.append(distance,
                                 self.compute_sensor_distance(car, mid_of_front_axle, 200, angle_index, self.object_mask,
                                                              draw_screen))
        return distance

    def get_sensors_points_distributions(self):
        fov_points = list()
        center_rect = Collision.center_rect(self.screen_width, self.screen_height)
        mid_of_front_axle = Collision.point_rotation(self.car, -1, 16, center_rect)
        arc_points = get_arc_points(mid_of_front_axle, 150, radians(90 + self.car.angle), radians(270 + self.car.angle),
                                    self.sensor_size)
        for end_point in arc_points:
            points_to_be_checked = list(get_equidistant_points(mid_of_front_axle, end_point, 25))
            fov_points.extend(points_to_be_checked)

        center_rect = Collision.center_rect(self.screen_width, self.screen_height)
        mid_of_rear_axle = Collision.point_rotation(self.car, 65, 16, center_rect)

        arc_points = get_arc_points(mid_of_rear_axle, 150, radians(-90 + self.car.angle), radians(90 + self.car.angle),
                                    self.sensor_size)

        for end_point in arc_points:
            points_to_be_checked = list(get_equidistant_points(mid_of_rear_axle, end_point, 25))
            fov_points.extend(points_to_be_checked)

        for index in range(len(fov_points)):
            pygame.draw.circle(self.screen, (255, 255, 0),
                                   (int(fov_points[index][0]), int(fov_points[index][1])), 5, 2)

        return fov_points

    def optimized_front_sensor(self, act_mask, display_obstacle_on_sensor=False):
        """
        front visual sensor
        :param act_mask:
        :param display_obstacle_on_sensor:
        :return:
        """
        # act_mask is a separate image where you can only see what the sensor sees
        center_rect = Collision.center_rect(self.screen_width, self.screen_height)
        mid_of_front_axle = Collision.point_rotation(self.car, -1, 16, center_rect)

        arc_points = get_arc_points(mid_of_front_axle, 150, radians(90 + self.car.angle), radians(270 + self.car.angle),
                                    self.sensor_size)
        offroad_edge_points = []

        obstacles = list()

        for end_point in arc_points:
            points_to_be_checked = list(get_equidistant_points(mid_of_front_axle, end_point, 25))
            check = False

            for line_point in points_to_be_checked:
                if np.array_equal(self.object_mask.get_at((int(line_point[0]), int(line_point[1]))), self.bkd_color):
                    check = True
                    break

            if check is False:
                offroad_edge_points.append(end_point)
            else:
                offroad_edge_points.append(line_point)

        for index in range(0, len(arc_points)):
            if offroad_edge_points[index] == arc_points[index]:
                pygame.draw.line(self.screen, (0, 255, 0), mid_of_front_axle, arc_points[index], True)
                pygame.draw.line(act_mask, (0, 255, 0), mid_of_front_axle, arc_points[index], True)
            else:
                obstacles.append(offroad_edge_points[index])
                pygame.draw.line(self.screen, (0, 255, 0), mid_of_front_axle, offroad_edge_points[index], True)
                pygame.draw.line(act_mask, (0, 255, 0), mid_of_front_axle, offroad_edge_points[index], True)
                if display_obstacle_on_sensor is True:
                    pygame.draw.line(self.screen, (255, 0, 0), offroad_edge_points[index], arc_points[index], True)
                    pygame.draw.line(act_mask, (255, 0, 0), offroad_edge_points[index], arc_points[index], True)
        return obstacles

    def optimized_rear_sensor(self, act_mask, display_obstacle_on_sensor=False):
        """
        rear visual sensor
        :param car:
        :param object_mask:
        :param act_mask:
        :param display_obstacle_on_sensor:
        :return:
        """
        # act_mask is a separate image where you can only see what the sensor sees
        center_rect = Collision.center_rect(self.screen_width, self.screen_height)
        mid_of_rear_axle = Collision.point_rotation(self.car, 65, 16, center_rect)

        arc_points = get_arc_points(mid_of_rear_axle, 150, radians(-90 + self.car.angle), radians(90 + self.car.angle),
                                    self.sensor_size)

        offroad_edge_points = []
        obstacles = list()

        for end_point in arc_points:
            points_to_be_checked = list(get_equidistant_points(mid_of_rear_axle, end_point, 25))

            check = False

            for line_point in points_to_be_checked:
                if np.array_equal(self.object_mask.get_at((int(line_point[0]), int(line_point[1]))), self.bkd_color):
                    check = True
                    break

            if check is False:
                offroad_edge_points.append(end_point)
            else:
                offroad_edge_points.append(line_point)

        for index in range(0, len(arc_points)):
            if offroad_edge_points[index] == arc_points[index]:
                pygame.draw.line(self.screen, (0, 255, 0), mid_of_rear_axle, arc_points[index], True)
                pygame.draw.line(act_mask, (0, 255, 0), mid_of_rear_axle, arc_points[index], True)
            else:
                obstacles.append(offroad_edge_points[index])
                pygame.draw.line(self.screen, (0, 255, 0), mid_of_rear_axle, offroad_edge_points[index], True)
                pygame.draw.line(act_mask, (0, 255, 0), mid_of_rear_axle, offroad_edge_points[index], True)
                if display_obstacle_on_sensor is True:
                    pygame.draw.line(self.screen, (255, 0, 0), offroad_edge_points[index], arc_points[index], True)
                    pygame.draw.line(act_mask, (255, 0, 0), offroad_edge_points[index], arc_points[index], True)
        return obstacles

    def initialize_activation_model(self, desired_layer_output):
        """
        activation model
        :param desired_layer_output:
        :return:
        """
        if self.print_activations is True and self.model_path is None:
            raise ValueError('no model_path given.')

        model = load_model(self.model_path)
        print('Using model...')
        model.summary()
        init_activations_display_window(desired_layer_output, 2048, 1024, 0.7)

        layer_names = []

        for layer in model.layers:
            layer_names.append(layer.name)

        image_buf = np.zeros((1, 500, 500, 3))
        state_buf = np.zeros((1, 4))

        layer_outputs = [layer.output for layer in model.layers]
        activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

        return layer_names, image_buf, state_buf, activation_model

    def custom_drawing(self, *args):
        """
        custom drawing function
        add drawings to simulator
        :param args: arguments
        :return:
        """
        pass

    def draw_sim_environment(self, print_coords=False):
        # Drawing
        """
        principal draw function that builds the simulator environment
        :param print_coords: print_coors on screen bool
        :return:
        """
        stagePosX = self.car.position[0] * self.ppu
        stagePosY = self.car.position[1] * self.ppu

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

        if self.cbox_front_sensor is not None:
            self.cbox_front_sensor.update()
        if self.cbox_rear_sensor is not None:
            self.cbox_rear_sensor.update()
        if self.cbox_distance_sensor is not None:
            self.cbox_distance_sensor.update()
        if self.cbox_front_sensor.isChecked() is True or self.cbox_rear_sensor.isChecked() is True:
            self.sensors = True
        else:
            self.sensors = False
        if self.cbox_distance_sensor.isChecked() is True:
            self.distance_sensor = True
        else:
            self.distance_sensor = False

        rotated = pygame.transform.rotate(self.car_image, self.car.angle)
        rot_rect = rotated.get_rect()

        center_x = int(self.screen_width / 2) - int(rot_rect.width / 2)
        center_y = int(self.screen_height / 2) - int(rot_rect.height / 2)

        # draw the ego car
        self.screen.blit(rotated, (center_x, center_y))
        self.custom_drawing()

        if print_coords is True:
            myfont = pygame.font.SysFont('Arial', 30)
            text1 = myfont.render('Car pos x: ' + str(round(stagePosX, 2)), True, (250, 0, 0))
            text2 = myfont.render('Car pos y: ' + str(round(stagePosY, 2)), True, (250, 0, 0))
            text3 = myfont.render('rel x: ' + str(round(rel_x, 2)), True, (250, 0, 0))
            text4 = myfont.render('rel y: ' + str(round(rel_y, 2)), True, (250, 0, 0))
            text5 = myfont.render('velocity: ' + str(round(self.car.velocity.x, 2) * self.ppu/4) + ' km/h', True, (250, 0, 0))

            self.screen.blit(text1, (20, 20))
            self.screen.blit(text2, (20, 50))
            self.screen.blit(text3, (20, 80))
            self.screen.blit(text4, (20, 110))
            self.screen.blit(text5, (20, 140))

        return stagePosX, stagePosY, rel_x, rel_y

    def return_to_menu(self):
        """
        if enabled_menu is True this function returns to main_menu
        :return:
        """
        if self.enabled_menu is True:
            from car_kinematic_city_menu import Menu
            menu = Menu()
            menu.main_menu()
            return
        else:
            return

    def key_handler(self, dt, rs_pos_list):
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
            if self.print_activations is True:
                cv2.destroyAllWindows()
            self.return_to_menu()
            quit()
        if pressed[pygame.K_r]:
            self.car.reset_car(rs_pos_list)
        if pressed[pygame.K_UP]:
            self.car.accelerate(dt)
        elif pressed[pygame.K_DOWN]:
            self.car.brake(dt)
        elif pressed[pygame.K_SPACE]:
            self.car.handbrake(dt)
        else:
            self.car.cruise(dt)
        if pressed[pygame.K_RIGHT]:
            self.car.steer_right(dt)
        elif pressed[pygame.K_LEFT]:
            self.car.steer_left(dt)
        else:
            self.car.no_steering()

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
                if self.cbox_front_sensor.onCheckbox(mouse_pos) and mouse_button_pressed is False:
                    self.cbox_front_sensor.changeState()
                if self.cbox_rear_sensor.onCheckbox(mouse_pos) and mouse_button_pressed is False:
                    self.cbox_rear_sensor.changeState()
                if self.cbox_distance_sensor.onCheckbox(mouse_pos) and mouse_button_pressed is False:
                    self.cbox_distance_sensor.changeState()
                mouse_button_pressed = True
            if event.type == pygame.MOUSEBUTTONUP:
                mouse_button_pressed = False

    @staticmethod
    def convert_surface_to_opencv_img(surface):
        """
        convert a surface to RGB opencv image
        :param surface: surface to be converted
        :return:
        """
        if type(surface) == pygame.Surface:
            sensor_img = pygame.surfarray.array3d(surface)
            sensor_img = np.rot90(sensor_img, axes=(0, 1))
            sensor_img = np.flipud(sensor_img)
            sensor_img = cv2.cvtColor(sensor_img, cv2.COLOR_BGR2RGB)
            return sensor_img
        else:
            raise ValueError("Given surface is not a pygame.Surface")

    def access_simulator_data(self, car_data=False, visual_sensor_data=False, distance_sensor_data=False):
        """
        car_data: acceleration -> car_data[0] ; steering -> car_data[1] ; angle -> car_data[2] ; velocity -> car_data[3]
        sensor_data: subsurface with sensor image
        :param car_data: if you want to access car_data set to true
        :param visual_sensor_data: if you want to access sensor_data set to true
        :param distance_sensor_data: if you want to access rays_sensor_data set to true and check the cbox in the simulator
        for rays_sensor.
        :return:
        """

        if car_data is True:
            car_acc = self.car.acceleration
            car_steering = self.car.steering
            car_angle = self.car.angle
            car_velocity = self.car.velocity.x
            car_data = [car_acc, car_steering, car_velocity, car_angle]
            return car_data
        elif visual_sensor_data is True:
            if self.sensors is True:
                image_rect = pygame.Rect((390, 110), (500, 500))
                sub = self.sensor_mask.subsurface(image_rect)
                return sub
            else:
                return None
        elif distance_sensor_data is True:
            if self.distance_sensor is True:
                if self.rays_sensor_distances is not None:
                    return self.rays_sensor_distances
                else:
                    return []
            else:
                return []
        else:
            raise ValueError("Please specify what data type to be returned.")

    def activate_sensors(self):
        """
        this function checks if any sensor has been activated
        :return:
        """
        obstacles = list()
        if self.cbox_front_sensor.isChecked():
            obstacles.extend(self.optimized_front_sensor(self.sensor_mask, display_obstacle_on_sensor=True))
        if self.cbox_rear_sensor.isChecked():
            obstacles.extend(self.optimized_rear_sensor(self.sensor_mask, display_obstacle_on_sensor=True))
        if self.cbox_distance_sensor.isChecked():
            if self.cbox_rear_sensor.isChecked() is False and self.cbox_front_sensor.isChecked() is False:
                self.rays_sensor_distances = self.enable_sensor(self.car, self.screen, self.rays_nr)
        return obstacles

    def record_data_function(self, index):
        """
        recording tab
        :param index: image index
        :return:
        """
        image_name = 'image_' + str(index) + '.png'
        index += 1

        if self.state_buf_path is None:
            raise OSError('state_buf_path is empty.')
        if self.replay_data_path is None:
            raise OSError('replay_data_path is empty.')

        actions = [self.car.position.x, self.car.position.y, float(round(self.car.angle, 3)),
                   float(round(self.car.acceleration, 3)),
                   float(round(self.car.velocity.x, 3)), image_name]

        # Save state_buf
        write_state_buf(self.state_buf_path, actions)

        # Save replay
        write_data(self.replay_data_path, self.car.position, self.car.angle)

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
