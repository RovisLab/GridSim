import os
import pygame
import pygame.gfxdraw
from math import radians
from checkbox import Checkbox
from collision import Collision
from read_write_trajectory import write_data, write_state_buf, read_traffic_data
from car import Car
from math_util import *
from keras.models import load_model
from keras import models
from print_activations import print_activations, init_activations_display_window
from obstacle_list import update_object_mask
from traffic_car import TrafficCar
import cv2


class Simulator:
    def __init__(self, screen, screen_width, screen_height,
                 car_x=5,
                 car_y=27,
                 sensor_size=50,
                 rays_nr=8,
                 activations=False,
                 traffic=True,
                 record_data=False,
                 replay_data_path=None,
                 state_buf_path=None,
                 sensors=False,
                 distance_sensor=False,
                 enabled_menu=False):
        pygame.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = screen

        # The color of the object mask
        # Yellow
        self.bkd_color = [255, 255, 0, 255]

        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.image_path = os.path.join(self.current_dir, "resources/cars/car_eb_2.png")
        self.car_image = pygame.image.load(self.image_path).convert_alpha()
        self.car_image = pygame.transform.scale(self.car_image, (42, 20))
        self.ppu = 16

        self.traffic_list = []

        self.traffic_image_path = os.path.join(self.current_dir, "resources/cars/car_traffic.png")
        self.traffic_car_image = pygame.image.load(self.traffic_image_path).convert_alpha()
        self.traffic_car_image = pygame.transform.scale(self.traffic_car_image, (42, 20))

        self.object_car_image_path = os.path.join(self.current_dir, "resources/cars/object_car.png")
        self.object_car_image = pygame.image.load(self.object_car_image_path).convert_alpha()
        self.object_car_image = pygame.transform.scale(self.object_car_image, (42, 20))

        self.print_activations = activations
        self.model_path = os.path.join(self.current_dir, 'used_models/activations_model.h5')
        if os.path.exists(self.model_path) is False:
            raise OSError("model to path doesn't exists")

        self.background = pygame.image.load(
            os.path.join(self.current_dir, "resources/backgrounds/maps_overlay.png")).convert()
        self.background = pygame.transform.scale(self.background, (2500, 1261))
        self.bgWidth, self.bgHeight = self.background.get_rect().size

        pygame.font.init()
        self.myfont = pygame.font.SysFont('Comic Sans MS', 30)

        self.input_image = pygame.surfarray.array3d(self.screen)

        self.clock = pygame.time.Clock()
        self.ticks = 60
        self.exit = False
        self.record_data = record_data
        self.traffic = traffic
        self.replay_data_path = replay_data_path
        self.state_buf_path = state_buf_path
        self.sensors = sensors
        self.distance_sensor = distance_sensor
        self.sensor_size = sensor_size
        self.rays_nr = rays_nr
        self.rays_sensor_distances = None
        self.enabled_menu = enabled_menu

        self.car = Car(car_x, car_y)
        self.sensor_mask = pygame.Surface((self.screen_width, self.screen_height))
        self.dt = None

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

    def enable_sensor(self, car, data_screen, draw_screen, rays_nr):
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
                                 self.compute_sensor_distance(car, mid_of_front_axle, 200, angle_index, data_screen,
                                                              draw_screen))
        return distance

    def optimized_front_sensor(self, car, object_mask, act_mask, display_obstacle_on_sensor=False):
        """
        front visual sensor
        :param car:
        :param object_mask:
        :param act_mask:
        :param display_obstacle_on_sensor:
        :return:
        """
        # act_mask is a separate image where you can only see what the sensor sees
        center_rect = Collision.center_rect(self.screen_width, self.screen_height)
        mid_of_front_axle = Collision.point_rotation(car, -1, 16, center_rect)

        arc_points = get_arc_points(mid_of_front_axle, 150, radians(90 + car.angle), radians(270 + car.angle), self.sensor_size)

        offroad_edge_points = []

        for end_point in arc_points:
            points_to_be_checked = list(get_equidistant_points(mid_of_front_axle, end_point, 25))

            check = False

            for line_point in points_to_be_checked:
                if np.array_equal(object_mask.get_at((int(line_point[0]), int(line_point[1]))), self.bkd_color):
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
                pygame.draw.line(self.screen, (0, 255, 0), mid_of_front_axle, offroad_edge_points[index], True)
                pygame.draw.line(act_mask, (0, 255, 0), mid_of_front_axle, offroad_edge_points[index], True)
                if display_obstacle_on_sensor is True:
                    pygame.draw.line(self.screen, (255, 0, 0), offroad_edge_points[index], arc_points[index], True)
                    pygame.draw.line(act_mask, (255, 0, 0), offroad_edge_points[index], arc_points[index], True)

    def optimized_rear_sensor(self, car, object_mask, act_mask, display_obstacle_on_sensor=False):
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
        mid_of_rear_axle = Collision.point_rotation(car, 65, 16, center_rect)

        arc_points = get_arc_points(mid_of_rear_axle, 150, radians(-90 + car.angle), radians(90 + car.angle), self.sensor_size)

        offroad_edge_points = []

        for end_point in arc_points:
            points_to_be_checked = list(get_equidistant_points(mid_of_rear_axle, end_point, 25))

            check = False

            for line_point in points_to_be_checked:
                if np.array_equal(object_mask.get_at((int(line_point[0]), int(line_point[1]))), self.bkd_color):
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
                pygame.draw.line(self.screen, (0, 255, 0), mid_of_rear_axle, offroad_edge_points[index], True)
                pygame.draw.line(act_mask, (0, 255, 0), mid_of_rear_axle, offroad_edge_points[index], True)
                if display_obstacle_on_sensor is True:
                    pygame.draw.line(self.screen, (255, 0, 0), offroad_edge_points[index], arc_points[index], True)
                    pygame.draw.line(act_mask, (255, 0, 0), offroad_edge_points[index], arc_points[index], True)

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

    def draw_sim_environment(self, car, object_mask, cbox_front_sensor=None, cbox_rear_sensor=None,
                             cbox_distance_sensor=None, print_coords=False):
        # Drawing
        """
        principal draw function that builds the simulator environment
        :param car: ego_car
        :param object_mask: object_mask
        :param cbox_front_sensor: front_sensor check_box
        :param cbox_rear_sensor: rear_sensor check_box
        :param print_coords: print_coors on screen bool
        :return:
        """
        stagePosX = car.position[0] * self.ppu
        stagePosY = car.position[1] * self.ppu

        rel_x = stagePosX % self.bgWidth
        rel_y = stagePosY % self.bgHeight

        self.screen.blit(self.background, (rel_x - self.bgWidth, rel_y - self.bgHeight))
        self.screen.blit(self.background, (rel_x, rel_y))
        self.screen.blit(self.background, (rel_x - self.bgWidth, rel_y))
        self.screen.blit(self.background, (rel_x, rel_y - self.bgHeight))

        if cbox_front_sensor is not None:
            cbox_front_sensor.update()
        if cbox_rear_sensor is not None:
            cbox_rear_sensor.update()
        if cbox_distance_sensor is not None:
            cbox_distance_sensor.update()
        if cbox_front_sensor.isChecked() is True or cbox_rear_sensor.isChecked() is True:
            self.sensors = True
        else:
            self.sensors = False
        if cbox_distance_sensor.isChecked() is True:
            self.distance_sensor = True
        else:
            self.distance_sensor = False

        rotated = pygame.transform.rotate(self.car_image, car.angle)
        rot_rect = rotated.get_rect()

        center_x = int(self.screen_width / 2) - int(rot_rect.width / 2)
        center_y = int(self.screen_height / 2) - int(rot_rect.height / 2)

        # draw the ego car
        self.screen.blit(rotated, (center_x, center_y))
        object_mask.fill((0, 0, 0))
        object_mask.blit(self.screen, (0, 0))
        update_object_mask(object_mask, rel_x, rel_y, self.bgWidth, self.bgHeight)

        if print_coords is True:
            myfont = pygame.font.SysFont('Arial', 30)
            text1 = myfont.render('Car pos x: ' + str(round(stagePosX, 2)), True, (250, 0, 0))
            text2 = myfont.render('Car pos y: ' + str(round(stagePosY, 2)), True, (250, 0, 0))
            text3 = myfont.render('rel x: ' + str(round(rel_x, 2)), True, (250, 0, 0))
            text4 = myfont.render('rel y: ' + str(round(rel_y, 2)), True, (250, 0, 0))
            text5 = myfont.render('velocity: ' + str(round(car.velocity.x, 2) * self.ppu/4) + ' km/h', True, (250, 0, 0))

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
            if self.print_activations is True:
                cv2.destroyAllWindows()
            self.return_to_menu()
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

    def event_handler(self, cbox_front_sensor, cbox_rear_sensor, cbox_distance_sensor, mouse_button_pressed):
        # Event queue
        """
        event handler for sensors check_boxes, exit event or mouse pressing events
        :param cbox_front_sensor:
        :param cbox_rear_sensor:
        :param cbox_distance_sensor:
        :param mouse_button_pressed:
        :return:
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.exit = True
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if cbox_front_sensor.onCheckbox(mouse_pos) and mouse_button_pressed is False:
                    cbox_front_sensor.changeState()
                if cbox_rear_sensor.onCheckbox(mouse_pos) and mouse_button_pressed is False:
                    cbox_rear_sensor.changeState()
                if cbox_distance_sensor.onCheckbox(mouse_pos) and mouse_button_pressed is False:
                    cbox_distance_sensor.changeState()
                mouse_button_pressed = True
            if event.type == pygame.MOUSEBUTTONUP:
                mouse_button_pressed = False

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

    def traffic_movement(self, collision_list, object_mask, stagePos):
        """
        traffic movement
        :param collision_list:
        :param object_mask: object mask
        :param stagePos: position of stage
        :return:
        """
        for i in self.traffic_list:
            if collision_list[self.traffic_list.index(i)]:
                i.index -= 1
            i.trajectory(stagePos, self.screen, object_mask, self.traffic_car_image, self.object_car_image,
                         2 * self.bgWidth,
                         2 * self.bgHeight)

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

    def custom(self, *args):
        """
        custom function in which to modify or access data
        :param args: custom arguments if needed
        :return:
        """
        pass

    def run(self):
        """
        main run loop
        :return:
        """
        # initialize traffic
        if self.traffic is True:
            self.init_traffic_cars()

        # sensor checkboxes on top right corner
        cbox_front_sensor = Checkbox(self.screen_width - 200, 10, 'Enable front sensor', self.sensors)
        cbox_rear_sensor = Checkbox(self.screen_width - 200, 35, 'Enable rear sensor', self.sensors)
        cbox_distance_sensor = Checkbox(self.screen_width - 200, 60, 'Enable distance sensor', self.distance_sensor)

        rs_pos_list = [[6, 27, 0.0], [5, 27, 180.0], [4, 24, 180.0], [4, 23, 0.0], [5, 27, 90.0], [5, 27, 0.0]]

        # boolean variable needed to check for single-click press
        mouse_button_pressed = False

        # initialize object mask
        object_mask = pygame.Surface((self.screen_width, self.screen_height))

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
            self.event_handler(cbox_front_sensor, cbox_rear_sensor, cbox_distance_sensor, mouse_button_pressed)

            # LOGIC
            self.key_handler(self.car, self.dt, rs_pos_list)

            # DRAWING
            stagePos = self.draw_sim_environment(self.car, object_mask, cbox_front_sensor, cbox_rear_sensor,
                                                 cbox_distance_sensor, print_coords=True)
            relPos = (stagePos[2], stagePos[3])
            stagePos = (stagePos[0], stagePos[1])

            # UPDATE
            # ------------------------ traffic car -----------------------------------------------
            if self.traffic is True:
                self.check_collisions(collision_list)
                self.traffic_movement(collision_list, object_mask, stagePos)
            # -------------------------------------------------------------------------------------------
            self.car.update(self.dt)

            if cbox_front_sensor.isChecked():
                self.optimized_front_sensor(self.car, object_mask, self.sensor_mask, display_obstacle_on_sensor=True)
            if cbox_rear_sensor.isChecked():
                self.optimized_rear_sensor(self.car, object_mask, self.sensor_mask, display_obstacle_on_sensor=True)
            if cbox_distance_sensor.isChecked():
                if cbox_rear_sensor.isChecked() is False and cbox_front_sensor.isChecked() is False:
                    self.rays_sensor_distances = self.enable_sensor(self.car, object_mask, self.screen, self.rays_nr)

            # -------------------------------------------print_activations----------------------------------------
            if self.print_activations is True:
                image_rect = pygame.Rect((390, 110), (500, 500))
                sub = self.screen.subsurface(image_rect)
                self.input_image = pygame.surfarray.array3d(sub)

                image_buf[0] = self.input_image
                activations = activation_model.predict([image_buf, state_buf])
                print_activations(activations, layer_names, desired_layer_output)
            # ----------------------------------------------------------

            # if not self.on_road(car, object_mask):
            #    car.reset_car(rs_pos_list)

            # CUSTOM FUNCTION TAB
            self.custom()

            # RECORD TAB
            if self.record_data is True:
                image_name = 'image_' + str(index_image) + '.png'
                index_image += 1

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

            pygame.display.update()
            self.clock.tick(self.ticks)

        pygame.quit()
