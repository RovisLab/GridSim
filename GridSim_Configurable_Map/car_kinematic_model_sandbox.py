from car import Car
from collision import Collision
import pygame
import os
from copy import copy
from checkbox import Checkbox
from math_util import *
from math import radians
from tracker import Tracker
from print_activations import print_activations, init_activations_display_window
from keras.models import load_model
from keras.models import Model
from read_write_trajectory import write_replay_data, write_state_buf, read_replay_coords, save_frame
import threading


class ConfigurableSimulator:
    def __init__(self, car_starting_position, car_image_path, map_path, object_map_path,
                 recorded_minimap=None,
                 car_size=(0, 0),
                 screen_width=1280,
                 screen_height=720,
                 scaling_factor=0.5,
                 show_activations=False,
                 show_minimap=True):

        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.car = Car(car_starting_position[0], car_starting_position[1], car_starting_position[2])
        if os.path.exists(os.path.join(self.current_dir, car_image_path)) is False:
            raise OSError('car_image_path does not exists.')
        self.car_image = pygame.image.load(os.path.join(self.current_dir, car_image_path)).convert_alpha()
        self.car_image = pygame.transform.scale(self.car_image, car_size)
        print('Car loaded...')
        if os.path.exists(os.path.join(self.current_dir, map_path)) is False:
            raise OSError('map_path does not exists.')
        self.map = pygame.image.load(os.path.join(self.current_dir, map_path)).convert()
        self.map_width, self.map_height = self.create_map_dimensions(scaling_factor)
        self.map = pygame.transform.scale(self.map, (self.map_width, self.map_height))
        print('Map loaded.Size: ', self.map.get_rect().size)

        if os.path.exists(os.path.join(self.current_dir, object_map_path)) is False:
            raise OSError('object_map_path does not exists.')
        self.object_map = pygame.image.load(os.path.join(self.current_dir, object_map_path)).convert()
        self.object_map = pygame.transform.scale(self.object_map, (self.map_width, self.map_height))
        print('Object_map loaded.Size: ', self.object_map.get_rect().size)

        self.screen_width, self.screen_height = pygame.display.get_surface().get_size()
        pygame.font.init()
        self.my_font = pygame.font.SysFont('Comic Sans MS', 30)
        self.obstacle_color = [254, 242, 0, 255]
        self.ppu = 16
        self.exit = False
        self.show_minimap = show_minimap
        self.clock = pygame.time.Clock()
        self.ticks = 60
        self.show_activ = show_activations
        self.received_action = None

        if show_minimap is True:
            self.tracker = Tracker(self.map_width, self.map_height, self.ppu, self.car, self.car_image, map_path,
                                   minimap_type='medium', recorded_minimap=recorded_minimap)
            print('Minimap created...')

        print('Simulator ready.')

    def create_map_dimensions(self, scaling_factor):
        m_width, m_height = self.map.get_width(), self.map.get_height()
        s_width, s_height = int(m_width*scaling_factor), int(m_height*scaling_factor)
        s_width, s_height = m_width + s_width, m_height + s_height
        return s_width, s_height

    def optimized_front_sensor(self):
        # act_mask is a separate image where you can only see what the sensor sees
        center_rect = Collision.center_rect(self.screen_width, self.screen_height)
        mid_of_front_axle = Collision.point_rotation(self.car, -1, 16, center_rect)

        arc_points = get_arc_points(mid_of_front_axle, 150, radians(90 + self.car.angle), radians(270 + self.car.angle), 80)

        offroad_edge_points = []

        for end_point in arc_points:
            points_to_be_checked = list(get_equidistant_points(mid_of_front_axle, end_point, 25))

            check = False

            for line_point in points_to_be_checked:
                if np.array_equal(self.screen.get_at((int(line_point[0]), int(line_point[1]))), self.obstacle_color):
                    check = True
                    break

            if check is False:
                offroad_edge_points.append(end_point)
            else:
                offroad_edge_points.append(line_point)

        return mid_of_front_axle, arc_points, offroad_edge_points

    def optimized_rear_sensor(self):
        # act_mask is a separate image where you can only see what the sensor sees
        center_rect = Collision.center_rect(self.screen_width, self.screen_height)
        mid_of_rear_axle = Collision.point_rotation(self.car, 65, 16, center_rect)

        arc_points = get_arc_points(mid_of_rear_axle, 150, radians(-90 + self.car.angle), radians(90 + self.car.angle), 80)

        offroad_edge_points = []

        for end_point in arc_points:
            points_to_be_checked = list(get_equidistant_points(mid_of_rear_axle, end_point, 25))

            check = False

            for line_point in points_to_be_checked:
                if np.array_equal(self.screen.get_at((int(line_point[0]), int(line_point[1]))), self.obstacle_color):
                    check = True
                    break

            if check is False:
                offroad_edge_points.append(end_point)
            else:
                offroad_edge_points.append(line_point)

        return mid_of_rear_axle, arc_points, offroad_edge_points

    def draw_front_sensor(self, front_arc_points, front_offroad_edge_points, mid_of_front_axle,
                          sensor_mask, display_obstacle_on_sensor=False):
        # draw front sensor
        for index in range(0, len(front_arc_points)):
            if front_offroad_edge_points[index] == front_arc_points[index]:
                pygame.draw.aaline(self.screen, (0, 255, 0), mid_of_front_axle, front_arc_points[index], True)
                pygame.draw.aaline(sensor_mask, (0, 255, 0), mid_of_front_axle, front_arc_points[index], True)
            else:
                pygame.draw.aaline(self.screen, (0, 255, 0), mid_of_front_axle, front_offroad_edge_points[index], True)
                pygame.draw.aaline(sensor_mask, (0, 255, 0), mid_of_front_axle, front_offroad_edge_points[index], True)
                if display_obstacle_on_sensor is True:
                    pygame.draw.aaline(self.screen, (255, 0, 0), front_offroad_edge_points[index], front_arc_points[index], True)
                    pygame.draw.aaline(sensor_mask, (255, 0, 0), front_offroad_edge_points[index], front_arc_points[index], True)

    def draw_rear_sensor(self, rear_arc_points, rear_offroad_edge_points, mid_of_rear_axle,
                         sensor_mask, display_obstacle_on_sensor=False):
        # draw rear sensor
        for index in range(0, len(rear_arc_points)):
            if rear_offroad_edge_points[index] == rear_arc_points[index]:
                pygame.draw.aaline(self.screen, (0, 255, 0), mid_of_rear_axle, rear_arc_points[index], True)
                pygame.draw.aaline(sensor_mask, (0, 255, 0), mid_of_rear_axle, rear_arc_points[index], True)
            else:
                pygame.draw.aaline(self.screen, (0, 255, 0), mid_of_rear_axle, rear_offroad_edge_points[index], True)
                pygame.draw.aaline(sensor_mask, (0, 255, 0), mid_of_rear_axle, rear_offroad_edge_points[index], True)
                if display_obstacle_on_sensor is True:
                    pygame.draw.aaline(self.screen, (255, 0, 0), rear_offroad_edge_points[index], rear_arc_points[index], True)
                    pygame.draw.aaline(sensor_mask, (255, 0, 0), rear_offroad_edge_points[index], rear_arc_points[index], True)

    def action_handler(self, dt):

        if self.received_action is not None:
            received_actions = set(self.received_action)

            if received_actions.intersection(set(['up'])):
                print('up')
                self.car.accelerate(dt)
            elif received_actions.intersection(set(['down'])):
                print('down')
                self.car.brake(dt)
            elif received_actions.intersection(set(['space'])):
                self.car.handbrake(dt)
            else:
                print('cruise')
                self.car.cruise(dt)
            if received_actions.intersection(set(['right'])):
                print('right')
                self.car.steer_right(dt)
            elif received_actions.intersection(set(['left'])):
                print('left')
                self.car.steer_left(dt)
            else:
                self.car.no_steering()

    def key_handler(self, dt, rs_pos_list):
        # User input
        pressed = pygame.key.get_pressed()
        action = []

        if pressed[pygame.K_r]:
            self.car.reset_car(rs_pos_list)
        if pressed[pygame.K_UP] or self.received_action == 'up':
            action.append('up')
            self.car.accelerate(dt)
        elif pressed[pygame.K_DOWN] or self.received_action == 'down':
            action.append('down')
            self.car.brake(dt)
        elif pressed[pygame.K_SPACE] or self.received_action == 'space':
            action.append('space')
            self.car.handbrake(dt)
        else:
            self.car.cruise(dt)
        if pressed[pygame.K_RIGHT] or self.received_action == 'right':
            action.append('right')
            self.car.steer_right(dt)
        elif pressed[pygame.K_LEFT] or self.received_action == 'left':
            action.append('left')
            self.car.steer_left(dt)
        else:
            self.car.no_steering()

        return action

    def event_handler(self, cbox_front_sensor, cbox_rear_sensor, mouse_button_pressed):
        # Event queue
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.exit = True
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if cbox_front_sensor.onCheckbox(mouse_pos) and mouse_button_pressed is False:
                    cbox_front_sensor.changeState()
                if cbox_rear_sensor.onCheckbox(mouse_pos) and mouse_button_pressed is False:
                    cbox_rear_sensor.changeState()
                mouse_button_pressed = True
            if event.type == pygame.MOUSEBUTTONUP:
                mouse_button_pressed = False

    def draw_sim_environment(self, sensor_mask, cbox_front_sensor, cbox_rear_sensor, save_minimap=False, minimap_name=None):

        # print('car_pos: ', self.car.position * self.ppu/8)
        offset_x = self.car.position[0] * self.ppu
        offset_y = self.car.position[1] * self.ppu
        # print('offset: ', offset_x, offset_y)

        # bound drawing to map size
        car_pos_x = 0
        car_pos_y = 0
        # min(0, offset_x)
        if offset_x > 0:
            car_pos_x = copy(-offset_x)
            offset_x = 0
        # min(0, offset_y)
        if offset_y > 0:
            car_pos_y = copy(-offset_y)
            offset_y = 0
        # max(-(map_width - screen_width), offset_x)
        if offset_x < -(self.map_width - self.screen_width):
            car_pos_x = -((self.map_width - self.screen_width) + offset_x)
            offset_x = -(self.map_width - self.screen_width)
        # max(-(map_height - screen_height), offset_y)
        if offset_y < -(self.map_height - self.screen_height):
            car_pos_y = -((self.map_height - self.screen_height) + offset_y)
            offset_y = -(self.map_height - self.screen_height)

        # rotate car
        rotated = pygame.transform.rotate(self.car_image, self.car.angle)
        rot_rect = rotated.get_rect()

        center_x = int(self.screen_width / 2) - int(rot_rect.width / 2) + car_pos_x
        center_y = int(self.screen_height / 2) - int(rot_rect.height / 2) + car_pos_y

        if self.show_minimap is True:
            # self.tracker.show_car_on_minimap(minimap_car_x_offset, minimap_car_y_offset, self.car.angle)
            minimap_car_x_offset, minimap_car_y_offset = self.tracker.scale_car_positions_to_minimap(center_x, center_y)
            self.tracker.track_car_movement(minimap_car_x_offset, minimap_car_y_offset, save_minimap=save_minimap,
                                            minimap_name=minimap_name)

        # draw
        # first draw onto screen the object_map and check the sensor for it
        self.screen.blit(self.object_map, (offset_x, offset_y))
        if cbox_front_sensor.isChecked():
            mid_front_axle, front_arc_points, front_offroad_points = self.optimized_front_sensor()
        if cbox_rear_sensor.isChecked():
            mid_rear_axle, rear_arc_points, rear_offroad_points = self.optimized_rear_sensor()
        # on top draw the actual map
        self.screen.blit(self.map, (offset_x, offset_y))
        cbox_front_sensor.update()
        cbox_rear_sensor.update()
        # draw the car
        self.screen.blit(rotated, (center_x, center_y))
        # and draw the sensor, with data from the object mask
        if cbox_front_sensor.isChecked():
            self.draw_front_sensor(front_arc_points, front_offroad_points, mid_front_axle,
                                   sensor_mask, display_obstacle_on_sensor=True)
        if cbox_rear_sensor.isChecked():
            self.draw_front_sensor(rear_arc_points, rear_offroad_points, mid_rear_axle,
                                   sensor_mask, display_obstacle_on_sensor=True)

    def initialize_activation_model(self, desired_layer_output):

        model = load_model((self.current_dir + '/resources/used_models/activations_model.h5').replace('\\', '/'))
        # model.summary()
        init_activations_display_window(desired_layer_output, 2048, 1024, 0.7)

        layer_names = []

        for layer in model.layers:
            layer_names.append(layer.name)

        image_buf = np.zeros((1, 500, 500, 3))
        state_buf = np.zeros((1, 4))

        layer_outputs = [layer.output for layer in model.layers]
        activation_model = Model(inputs=model.input, outputs=layer_outputs)

        return layer_names, image_buf, state_buf, activation_model

    @staticmethod
    def show_activations(layer_names, image_buf, state_buf, activation_model, sensor_mask):
        image_rect = pygame.Rect((390, 110), (500, 500))
        sub = sensor_mask.subsurface(image_rect)
        input_image = pygame.surfarray.array3d(sub)

        image_buf[0] = input_image
        activations = activation_model.predict([image_buf, state_buf])
        print_activations(activations, layer_names, 'convolution0')

    def run(self, record_data_path=None, replay_data_path=None, save_image=False, save_minimap=False, minimap_name=None):
        """

        :param record_data_path: only path, replay.txt and state.txt generated automatically
        :param replay_data_path: replay path, replay and state are found automatically
        :param save_image: if you want in replay to save the display
        :param save_minimap: if you want to save the minimap
        :param minimap_name: name of the minimap you want to save
        :return:
        """
        if record_data_path is not None:
            if os.path.exists(record_data_path) is False:
                raise OSError(record_data_path + ' does not exists.')

        if replay_data_path is not None:
            if os.path.exists(replay_data_path) is False:
                raise OSError(replay_data_path + ' does not exists.')
            else:
                car_replay_coords = read_replay_coords(replay_data_path + '/replay_data.txt')
                index_replay = 0
                replay = True
        else:
            replay = False

        if self.show_activ is True or save_image is True:
            cbox_front_sensor = Checkbox(self.screen_width - 200, 10, 'Enable front sensor', True)
            cbox_rear_sensor = Checkbox(self.screen_width - 200, 35, 'Enable rear sensor', True)
        else:
            cbox_front_sensor = Checkbox(self.screen_width - 200, 10, 'Enable front sensor', False)
            cbox_rear_sensor = Checkbox(self.screen_width - 200, 35, 'Enable rear sensor', False)

        mouse_button_pressed = False
        # to be updated
        rs_pos_list = []

        sensor_mask = pygame.Surface((self.screen_width, self.screen_height))
        if self.show_activ is True:
            layer_names, image_buf, state_buf, activation_model = self.initialize_activation_model('convolution0')
        index_image = 0

        while not self.exit:
            dt = self.clock.get_time() / 1000
            self.event_handler(cbox_front_sensor, cbox_rear_sensor, mouse_button_pressed)

            if replay is True:
                if index_replay > len(car_replay_coords) - 1:
                    break
                self.car.position.x = car_replay_coords[index_replay][0]
                self.car.position.y = car_replay_coords[index_replay][1]
                self.car.angle = car_replay_coords[index_replay][2]
                index_replay += 1
            else:
                action = self.key_handler(dt, rs_pos_list)
                self.action_handler(dt)

            sensor_mask.fill((0, 0, 0))
            self.draw_sim_environment(sensor_mask, cbox_front_sensor, cbox_rear_sensor, save_minimap=save_minimap,
                                      minimap_name=minimap_name)
            self.car.update(dt)
            if replay is True and save_image is True:
                image_name = 'image_' + str(index_replay) + '.png'
                save_frame(self.screen, image_name, replay_data_path + '/images')
            if self.show_activ is True:
                self.show_activations(layer_names, image_buf, state_buf, activation_model, sensor_mask)

            if record_data_path is not None:
                image_name = 'image_' + str(index_image) + '.png'
                index_image += 1
                actions = [self.car.position.x, self.car.position.y, float(round(self.car.angle, 3)),
                           float(round(self.car.acceleration, 3)),
                           float(round(self.car.velocity.x, 3)), action, image_name]
                write_replay_data(record_data_path + '/replay_data.txt', self.car.position, self.car.angle)
                write_state_buf(record_data_path + '/state_buf.txt', actions)

            pygame.display.update()
            self.clock.tick(self.ticks)

        pygame.quit()


if __name__ == '__main__':
    CAR_IMAGE_PATH = 'resources/cars/car_eb_2.png'
    MAP_PATH = 'resources/backgrounds/scenario_b_4800x3252.jpg'
    OBJECT_MAP_PATH = 'resources/backgrounds/scenario_b_4800x3252_obj_map.jpg'
    RECORDED_MINIMAP = 'resources/backgrounds/minimap.png'
    RECORD_DATA_PATH = 'resources/recorded_data/run5'
    car_size = (32, 15)
    starting_position = (-130, -450, -90)
    scaling_factor = 1.5

    sim = ConfigurableSimulator(starting_position, CAR_IMAGE_PATH, MAP_PATH, OBJECT_MAP_PATH, car_size=car_size,
                                scaling_factor=scaling_factor)
    sim.run(record_data_path=None, replay_data_path=RECORD_DATA_PATH, save_image=True, save_minimap=True,
            minimap_name='run5')
