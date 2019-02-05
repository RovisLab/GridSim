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


class Simulator:
    def __init__(self, screen, screen_width, screen_height,
                 sensor_size=150,
                 activations=False,
                 traffic=True,
                 record_data=False,
                 replay_data_path=None,
                 state_buf_path=None,
                 sensors=False):
        pygame.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = screen

        # The color of the object mask
        # Yellow
        self.bkd_color = [255, 255, 0, 255]
        # Green
        # self.bkd_color = [0, 88, 0, 255]
        # Olive
        # self.bkd_color = [83, 125, 10, 255]

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
        self.sensor_size = sensor_size

    def optimized_front_sensor(self, car, object_mask, act_mask, display_obstacle_on_sensor=False):
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

    def draw_sim_environment(self, car, object_mask, cbox_front_sensor, cbox_rear_sensor, print_coords=False):
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

        cbox_front_sensor.update()
        cbox_rear_sensor.update()

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

    @staticmethod
    def return_to_menu():
        from car_kinematic_city_menu import Menu
        menu = Menu()
        menu.main_menu()
        return

    def key_handler(self, car, dt, rs_pos_list):
        # User input
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_ESCAPE]:
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

    def event_handler(self, cbox_front_sensor, cbox_rear_sensor, mouse_button_pressed):
        # Event queue
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.exit = True
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if cbox_front_sensor.onCheckbox(mouse_pos) and mouse_button_pressed == False:
                    cbox_front_sensor.changeState()
                if cbox_rear_sensor.onCheckbox(mouse_pos) and mouse_button_pressed == False:
                    cbox_rear_sensor.changeState()
                mouse_button_pressed = True
            if event.type == pygame.MOUSEBUTTONUP:
                mouse_button_pressed = False

    @staticmethod
    def traffic_car_collision(traffic_car_1, traffic_car_2, collision_list, collision_index):
        for x1, y1 in zip(traffic_car_1.data_x[traffic_car_1.index: traffic_car_1.index + 40], traffic_car_1.data_y[
                                                                                               traffic_car_1.index:
                                                                                               traffic_car_1.index + 40]):
            for x2, y2 in zip(traffic_car_2.data_x[traffic_car_2.index: traffic_car_2.index + 40], traffic_car_2.data_y[
                                                                                                   traffic_car_2.index:
                                                                                                   traffic_car_2.index + 40]):
                if abs(x2 - x1) <= 1 and abs(y2 - y1) <= 20:
                    collision_list[collision_index] = True

    def init_traffic_cars(self):
        trajectories = read_traffic_data(os.path.join(self.current_dir, "resources/traffic_cars_data/traffic_trajectories.csv"))
        for trajectory in trajectories:
            traffic_car = TrafficCar(trajectory[0], int(trajectory[1]))
            self.traffic_list.append(traffic_car)

    def check_collisions(self, collision_list):
        for i in range(0, len(self.traffic_list) - 1):
            for j in range(i + 1, len(self.traffic_list)):
                self.traffic_car_collision(self.traffic_list[i], self.traffic_list[j], collision_list, j)

    def traffic_movement(self, collision_list, object_mask, stagePos):
        for i in self.traffic_list:
            if collision_list[self.traffic_list.index(i)]:
                i.index -= 1
            i.trajectory(stagePos, self.screen, object_mask, self.traffic_car_image, self.object_car_image,
                         2 * self.bgWidth,
                         2 * self.bgHeight)

    def run(self):
        # place car on road
        car = Car(5, 27)

        # initialize traffic
        self.init_traffic_cars()

        if self.sensors is True:
            sen = True
        else:
            sen = False

        # sensor checkboxes on top right corner
        cbox_front_sensor = Checkbox(self.screen_width - 200, 10, 'Enable front sensor', sen)
        cbox_rear_sensor = Checkbox(self.screen_width - 200, 35, 'Enable rear sensor', sen)

        # reset position list -> to be updated
        rs_pos_list = [[650, 258, 90.0], [650, 258, 270.0], [0, 0, 180.0], [0, 0, 0.0], [302, 200, 45.0],
                       [40, 997, 0.0], [40, 997, 180.0], [100, 997, 0.0], [100, 997, 180.0], [400, 998, 0.0],
                       [400, 998, 180.0], [385, 315, 135.0]]

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

            dt = self.clock.get_time() / 1000
            self.event_handler(cbox_front_sensor, cbox_rear_sensor, mouse_button_pressed)

            # LOGIC
            self.key_handler(car, dt, rs_pos_list)
            car.acceleration = max(-car.max_acceleration, min(car.acceleration, car.max_acceleration))
            car.steering = max(-car.max_steering, min(car.steering, car.max_steering))

            # DRAWING
            stagePos = self.draw_sim_environment(car, object_mask, cbox_front_sensor, cbox_rear_sensor,
                                                 print_coords=True)
            relPos = (stagePos[2], stagePos[3])
            stagePos = (stagePos[0], stagePos[1])

            # UPDATE
            # ------------------------ traffic car -----------------------------------------------
            if self.traffic is True:
                self.check_collisions(collision_list)
                self.traffic_movement(collision_list, object_mask, stagePos)
            # -------------------------------------------------------------------------------------------
            car.update(dt)

            act_mask = pygame.Surface((self.screen_width, self.screen_height))
            if cbox_front_sensor.isChecked():
                self.optimized_front_sensor(car, object_mask, act_mask, display_obstacle_on_sensor=True)
            if cbox_rear_sensor.isChecked():
                self.optimized_rear_sensor(car, object_mask, act_mask, display_obstacle_on_sensor=True)

            # -------------------------------------------print_activations----------------------------------------
            if self.print_activations is True:
                image_rect = pygame.Rect((390, 110), (500, 500))
                sub = self.screen.subsurface(image_rect)
                self.input_image = pygame.surfarray.array3d(sub)

                image_buf[0] = self.input_image
                activations = activation_model.predict([image_buf, state_buf])
                print_activations(activations, layer_names, desired_layer_output)
            # ----------------------------------------------------------

            # RECORD TAB
            if self.record_data is True:
                image_name = 'image_' + str(index_image) + '.png'
                index_image += 1

                if self.state_buf_path is None:
                    raise OSError('state_buf_path is empty.')
                if self.replay_data_path is None:
                    raise OSError('replay_data_path is empty.')

                actions = [car.position.x, car.position.y, float(round(car.angle, 3)), float(round(car.acceleration, 3)),
                           float(round(car.velocity.x, 3)), image_name]

                # Save state_buf
                write_state_buf(self.state_buf_path, actions)

                # Save replay
                write_data(self.replay_data_path, car.position, car.angle)

            pygame.display.update()
            self.clock.tick(self.ticks)

        pygame.quit()


if __name__ == '__main__':
    screen = pygame.display.set_mode((1280, 720))
    STATE_BUF_PATH = '/GridSim Data/state_buf.csv'
    REPLAY_DATA_PATH = '/GridSim Data/replay_data.csv'
    sim = Simulator(screen, 1280, 720, record_data=True, traffic=True, activations=False,
                    replay_data_path=REPLAY_DATA_PATH, state_buf_path=STATE_BUF_PATH)
    sim.run()
