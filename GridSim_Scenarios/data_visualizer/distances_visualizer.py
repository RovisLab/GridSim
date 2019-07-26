from read_write_trajectory import read_distances, read_coords
import numpy as np
import matplotlib.pyplot as plt
import os
import pygame
import time
from math import radians, sin, cos
from math_util import euclidean_norm
import cv2
from PIL import Image
import math


class DistancesVisualizer(object):
    """
    Class used for input distances visualization.
    """
    def __init__(self, car_data_path=None, front_data_path=None, rear_data_path=None, car_length=42, debug_window=False):
        """
        Constructor
        :param car_data_path: replay.csv path used if car angles are needed, else the default car angle is -90
        :param front_data_path: distances of the front sensor
        :param rear_data_path: distances of the rear sensor
        :param car_length: car length (car image height)
        :param debug_window: if debug is needed a pygame window will pop-up with the intermediate steps
        """
        if front_data_path is None and rear_data_path is None:
            raise ValueError("No data path given, please give at least 1 path(front/rear)")
        if car_data_path is None:
            self.default_car_angle = -90
            # raise ValueError("No car data path given.")
        if debug_window is True:
            self.test_screen = pygame.display.set_mode((1280, 720))
            pygame.display.set_caption("Debug Window")
        else:
            self.test_screen = None

        current_dir = os.path.dirname(os.path.abspath(__file__))

        self.car_image = pygame.image.load(os.path.join(current_dir, "../resources/cars/car_eb_2.png"))
        self.car_image = pygame.transform.scale(self.car_image, (car_length, int(car_length/2) - 1))

        self.car_length = car_length
        if car_data_path is not None:
            self.car_data_path = car_data_path
            self.car_angle_data = np.asanyarray(read_coords(car_data_path))[:, 2]
        else:
            self.car_angle_data = None

        self.front_data_path = front_data_path
        self.rear_data_path = rear_data_path
        if self.front_data_path is not None:
            if os.path.exists(self.front_data_path) is True:
                self.front_sensor_data = read_distances(self.front_data_path)
        else:
            self.front_sensor_data = None
        if self.rear_data_path is not None:
            if os.path.exists(self.rear_data_path) is True:
                self.rear_sensor_data = read_distances(self.rear_data_path)
        else:
            self.rear_sensor_data = None

    def get_front_sensor_size(self):
        """
        :returns front sensor size based on given front sensor data
        """
        if self.front_sensor_data is not None:
            return len(self.front_sensor_data[0])
        else:
            return 0

    def get_rear_sensor_size(self):
        """
        :returns rear sensor size based on given rear sensor data
        """
        if self.rear_sensor_data is not None:
            return len(self.rear_sensor_data[0])
        else:
            return 0

    def modify_distances_path(self, new_distance_path, front=False, rear=False):
        """
        Modify front or rear data path in the middle of the visualization if needed
        :param new_distance_path: path to new distances folder
        :param front: front bool -> modify front data path
        :param rear: rear bool -> modify rear data path
        :return:
        """
        if front is True:
            self.front_data_path = new_distance_path
            return
        if rear is True:
            self.rear_data_path = new_distance_path
            return

    def reload_distances_file(self, front=False, rear=False):
        """
        After modifying the data paths, reload the files
        :param front: front bool -> modify front data path
        :param rear: rear bool -> modify rear data path
        :return:
        """
        if front is True:
            self.front_sensor_data = read_distances(self.front_data_path)
        if rear is True:
            self.rear_sensor_data = read_distances(self.rear_sensor_data)

    def modify_distances_file(self, new_distances_path, front=False, rear=False):
        self.modify_distances_path(new_distances_path, front=front, rear=rear)
        self.reload_distances_file(front=front, rear=rear)

    @staticmethod
    def calculate_corner_points(image_w, image_h, car_angle, ox=20, oy=12):
        """
        Determine corner and middle points of the car
        :param image_w: image width
        :param image_h:  image height
        :param car_angle: car angle at a certain frame
        :param ox: ox
        :param oy: oy
        :return: top_right, top_left, bot_right, bot_left, mid_top, mid_bot coordinates
        """
        center_world_x = int(image_w / 2)
        center_world_y = int(image_h / 2)

        top_left_x = center_world_x - int(ox * cos(radians(-car_angle))) + int(oy * sin(radians(-car_angle)))
        top_left_y = center_world_y - int(ox * sin(radians(-car_angle))) - int(oy * cos(radians(-car_angle)))
        top_left = (top_left_x, top_left_y)

        top_right_x = center_world_x + int(ox * cos(radians(-car_angle))) + int(oy * sin(radians(-car_angle)))
        top_right_y = center_world_y + int(ox * sin(radians(-car_angle))) - int(oy * cos(radians(-car_angle)))
        top_right = (top_right_x, top_right_y)

        bot_right_x = center_world_x + int(ox * cos(radians(-car_angle))) - int(oy * sin(radians(-car_angle)))
        bot_right_y = center_world_y + int(ox * sin(radians(-car_angle))) + int(oy * cos(radians(-car_angle)))
        bot_right = (bot_right_x, bot_right_y)

        bot_left_x = center_world_x - int(ox * cos(radians(-car_angle))) - int(oy * sin(radians(-car_angle)))
        bot_left_y = center_world_y - int(ox * sin(radians(-car_angle))) + int(oy * cos(radians(-car_angle)))
        bot_left = bot_left_x, bot_left_y

        mid_top = (int((top_left_x + bot_left_x) / 2), int((top_left_y + bot_left_y) / 2))
        mid_bot = (int((top_right_x + bot_right_x) / 2), int((top_right_y + bot_right_y) / 2))

        return top_right, top_left, bot_right, bot_left, mid_top, mid_bot

    @staticmethod
    def resize_image(image, new_size):
        """
        Resize image using opencv
        :param image: image to be resized
        :param new_size: new image size
        :return: image resized
        """
        resized_image = cv2.resize(image, new_size)
        return resized_image

    @staticmethod
    def convert_surface_to_opencv_img(surface):
        """
        Convert a pygame surface to opencv image
        :param surface: surface to be converted
        :return: opencv image of the pygame surface
        """
        if type(surface) == pygame.Surface:
            sensor_img = pygame.surfarray.array3d(surface)
            sensor_img = np.rot90(sensor_img, axes=(0, 1))
            sensor_img = np.flipud(sensor_img)
            sensor_img = cv2.cvtColor(sensor_img, cv2.COLOR_BGR2RGB)
            return sensor_img
        else:
            raise ValueError("Given surface is not a pygame.Surface")

    @staticmethod
    def calculate_end_point(base_point, sensor_length, angle_index, car_angle, front=False, rear=False):
        """
        Calculate end points for one ray of the sensor
        :param base_point: base point
        :param sensor_length: sensor length, default 200
        :param angle_index: index of the angle of the ray
        :param car_angle: angle of the car
        :param front: front bool -> calculate for the front sensor
        :param rear: rear bool -> calculate for the rear sensor
        :return: end_point
        """
        if front is True:
            end_point_x = base_point[0] + sensor_length * cos(radians(angle_index - car_angle))
            end_point_y = base_point[1] + sensor_length * sin(radians(angle_index - car_angle))
            end_point = (end_point_x, end_point_y)
        elif rear is True:
            end_point_x = base_point[0] - sensor_length * cos(radians(angle_index - car_angle))
            end_point_y = base_point[1] - sensor_length * sin(radians(angle_index - car_angle))
            end_point = (end_point_x, end_point_y)
        else:
            raise ValueError("Side of the car not specified")

        return end_point

    @staticmethod
    def calculate_collision_point(base_point, point_index, angle_index, car_angle, front=False, rear=False):
        """
        Calculate the current collision point (point that is checked to see if the distance front the file matches)
        :param base_point: base point
        :param point_index: index of the point on the ray
        :param angle_index: index of the angle of the ray
        :param car_angle: angle of the car
        :param front: front bool -> calculate for the front sensor
        :param rear: rear bool -> calculate for the rear sensor
        :return: coll_point
        """
        if front is True:
            coll_point_x = base_point[0] + point_index * cos(radians(angle_index - car_angle))
            coll_point_y = base_point[1] + point_index * sin(radians(angle_index - car_angle))
            coll_point = (coll_point_x, coll_point_y)
        elif rear is True:
            coll_point_x = base_point[0] - point_index * cos(radians(angle_index - car_angle))
            coll_point_y = base_point[1] - point_index * sin(radians(angle_index - car_angle))
            coll_point = (coll_point_x, coll_point_y)
        else:
            raise ValueError("Side of the car not specified")

        return coll_point

    def reconstruct_sensor_image(self, image_w, image_h, car_angle=None, front_sensor_data=None, rear_sensor_data=None,
                                 draw_car=False, sensor_length=200):
        """
        Sensor image reconstruction
        :param image_w: image width
        :param image_h: image height
        :param car_angle: angle of the car at a certain frame
        :param front_sensor_data: front sensor batch data
        :param rear_sensor_data: rear sensor batch data
        :param draw_car: bool -> draw the car on the display grid
        :param sensor_length: length of the sensor, default 200
        :return: sensor_image_surface
        """
        if car_angle is None:
            raise ValueError("No car angle given for reconstruction.")
        _, _, _, _, mid_top, mid_bot = self.calculate_corner_points(image_w, image_h, car_angle)

        sensor_image_surface = pygame.Surface((image_w, image_h))
        sensor_image_surface.fill((0, 0, 0))
        image_center = (int(image_w/2), int(image_h/2))

        rotated_car_image = pygame.transform.rotate(self.car_image, car_angle)
        rotated_rect = rotated_car_image.get_rect()
        rotated_rect.center = image_center
        car_drawing_center = (image_center[0] - int(rotated_rect.width / 2),
                              image_center[1] - int(rotated_rect.height / 2))

        if draw_car is True:
            sensor_image_surface.blit(rotated_car_image, car_drawing_center)

        if self.test_screen is not None:
            pygame.draw.circle(sensor_image_surface, (255, 0, 0), mid_top, 3)
            pygame.draw.circle(sensor_image_surface, (0, 255, 0), mid_bot, 3)

            self.test_screen.blit(sensor_image_surface, (0, 0))
            pygame.display.update()

        if car_angle is not None:
            if front_sensor_data is not None:
                sensor_size = self.get_front_sensor_size()
                front_inside_index = 0
                for angle_index in range(120, 240, int(round(120/sensor_size))):
                    end_point = self.calculate_end_point(mid_top, sensor_length, angle_index, car_angle, front=True)

                    for index in range(0, sensor_length):
                        coll_point = self.calculate_collision_point(mid_top, index, angle_index, car_angle, front=True)
                        distance = euclidean_norm(mid_top, coll_point)

                        if int(float(distance)) == int(float(front_sensor_data[front_inside_index])):
                            pygame.draw.line(sensor_image_surface, (0, 255, 0), mid_top, coll_point, True)
                            pygame.draw.line(sensor_image_surface, (255, 0, 0), coll_point, end_point, True)
                            break
                        else:
                            if int(float(front_sensor_data[front_inside_index])) - int(float(distance)) <= 1:
                                pygame.draw.line(sensor_image_surface, (0, 255, 0), mid_top, coll_point, True)
                                pygame.draw.line(sensor_image_surface, (255, 0, 0), coll_point, end_point, True)
                                break
                            elif index == sensor_length - 1:
                                pygame.draw.line(sensor_image_surface, (0, 255, 0), mid_top, end_point, True)
                                break

                    front_inside_index += 1

            if rear_sensor_data is not None:
                sensor_size = self.get_rear_sensor_size()
                rear_inside_index = 0
                for angle_index in range(120, 240, int(round(120 / sensor_size))):
                    end_point = self.calculate_end_point(mid_bot, sensor_length, angle_index, car_angle, rear=True)

                    for index in range(0, sensor_length):
                        coll_point = self.calculate_collision_point(mid_bot, index, angle_index, car_angle, rear=True)
                        distance = euclidean_norm(mid_bot, coll_point)

                        if int(float(distance)) == int(float(rear_sensor_data[rear_inside_index])):
                            pygame.draw.line(sensor_image_surface, (0, 255, 0), mid_bot, coll_point, True)
                            pygame.draw.line(sensor_image_surface, (255, 0, 0), coll_point, end_point, True)
                            break
                        else:
                            if int(float(rear_sensor_data[rear_inside_index])) - int(float(distance)) <= 1:
                                pygame.draw.line(sensor_image_surface, (0, 255, 0), mid_bot, coll_point, True)
                                pygame.draw.line(sensor_image_surface, (255, 0, 0), coll_point, end_point, True)
                                break
                            elif index == sensor_length - 1:
                                pygame.draw.line(sensor_image_surface, (0, 255, 0), mid_bot, end_point, True)
                                break

                    rear_inside_index += 1

            if self.test_screen is not None:
                self.test_screen.blit(sensor_image_surface, (0, 0))
                pygame.display.update()
                time.sleep(0.02)

            return sensor_image_surface
        return None

    @staticmethod
    def initialize_image_grid(image_w, image_h, imgs_per_col, imgs_per_row):
        """
        Create the display grid empty image
        :param image_w: image width
        :param image_h: image height
        :param imgs_per_col: wanted images per col
        :param imgs_per_row: wanted images per row
        :return: empty display grid
        """
        total_width = image_w * imgs_per_row
        total_height = image_h * imgs_per_col
        display_grid = Image.new('RGB', (total_width, total_height))
        return display_grid

    @staticmethod
    def initialize_opencv_window(window_name, w, h):
        """
        Initialization of the opencv window
        :param window_name: window name
        :param w: window width
        :param h: window height
        :return: None
        """
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow(window_name, int(w), int(h))

    def display_images_grid(self, sensor_images, batch_size, image_w, image_h, viewtime, matplotlib, opencv):
        """

        :param sensor_images: list of the batch of sensor images
        :param batch_size: batch size
        :param image_w: sample image width
        :param image_h: sample image height
        :param viewtime: viewtime for the matplotlib window
        :param matplotlib: bool -> visualization with matplotlib
        :param opencv: bool -> visualization with opencv
        :return: None
        """
        if math.ceil(math.sqrt(batch_size)) == int(math.sqrt(batch_size)):
            images_per_col = int(math.sqrt(batch_size))
            images_per_row = int(math.sqrt(batch_size))
        else:
            images_per_col = int(math.sqrt(batch_size)) + 1
            images_per_row = int(math.sqrt(batch_size)) + 1

        display_grid = self.initialize_image_grid(image_w, image_h, images_per_col, images_per_row)
        row = 0
        col = 0

        for img in sensor_images:
            np_img = self.convert_surface_to_opencv_img(img)
            res_np_img = cv2.cvtColor(self.resize_image(np_img, (image_w, image_h)), cv2.COLOR_BGR2RGB)
            row_offset = row * image_w
            col_offset = col * image_h
            pil_image = Image.fromarray(res_np_img)
            display_grid.paste(pil_image, (row_offset, col_offset))

            col += 1
            if col == images_per_col:
                row += 1
                col = 0
        if matplotlib is True:
            plt.imshow(display_grid)
            plt.show(block=False)
            plt.pause(viewtime)
            plt.close()
        if opencv is True:
            w, h = int(display_grid.size[0] * 1/2), int(display_grid.size[1] * 1/2)
            self.initialize_opencv_window("Sensor Images", w, h)
            open_cv_image = np.array(display_grid)
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            open_cv_image = self.resize_image(open_cv_image, (w, h))
            cv2.imshow("Sensor Images", open_cv_image)
            cv2.waitKey(1)

    def zip_arg_function(self, *args, batch_size, image_w, image_h, draw_car, sensor_length, viewtime, matplotlib, opencv):
        """
        Image batch creation and data visualization
        :param args: possible datas (car data, front sensor data, rear data -> None if empty)
        :param batch_size: batch size
        :param image_w: image width
        :param image_h: image height
        :param draw_car: draw car bool
        :param sensor_length: sensor length
        :param viewtime: viewtime for the matplotlib window
        :param matplotlib: bool -> visualization with matplotlib
        :param opencv: bool -> visualization with opencv
        :return: None
        """
        current_batch_index = 0
        sensor_images = []

        for angle, front_sensor_data, rear_sensor_data in zip(*args):
            if current_batch_index == batch_size:
                self.display_images_grid(sensor_images, batch_size, image_w, image_h, viewtime, matplotlib, opencv)
                current_batch_index = 0
                sensor_images = []

            sensor_images.append(self.reconstruct_sensor_image(image_w, image_h, angle,
                                                               draw_car=draw_car,
                                                               front_sensor_data=front_sensor_data,
                                                               rear_sensor_data=rear_sensor_data,
                                                               sensor_length=sensor_length))
            current_batch_index += 1

    def visualize_sensor_data(self, image_h, image_w, batch_size=8, draw_car=False, sensor_length=200, viewtime=2,
                              matplotlib=False, opencv=False):
        """
        Caller function for the image batch creation and data visualization function
        Check zip_arg_function for more info
        :param image_h:
        :param image_w:
        :param batch_size:
        :param draw_car:
        :param sensor_length:
        :param viewtime:
        :param matplotlib:
        :param opencv:
        :return: None
        """
        if self.car_angle_data is None:
            if self.front_sensor_data is not None:
                self.car_angle_data = [self.default_car_angle] * len(self.front_sensor_data)
            elif self.rear_sensor_data is not None:
                self.car_angle_data = [self.default_car_angle] * len(self.rear_sensor_data)
            else:
                raise ValueError("No sensor data given and no car data given.")

        if self.car_angle_data is not None:
            if self.front_sensor_data is not None and self.rear_sensor_data is None:
                self.zip_arg_function(self.car_angle_data, self.front_sensor_data, [None] * len(self.front_sensor_data),
                                      batch_size=batch_size, image_w=image_w, image_h=image_h, draw_car=draw_car,
                                      sensor_length=sensor_length, viewtime=viewtime, matplotlib=matplotlib,
                                      opencv=opencv)

            if self.rear_sensor_data is not None and self.front_sensor_data is None:
                self.zip_arg_function(self.car_angle_data, [None] * len(self.rear_sensor_data), self.rear_sensor_data,
                                      batch_size=batch_size, image_w=image_w, image_h=image_h, draw_car=draw_car,
                                      sensor_length=sensor_length, viewtime=viewtime, matplotlib=matplotlib,
                                      opencv=opencv)

            if self.front_sensor_data is not None and self.rear_sensor_data is not None:
                self.zip_arg_function(self.car_angle_data, self.front_sensor_data, self.rear_sensor_data,
                                      batch_size=batch_size, image_w=image_w, image_h=image_h, draw_car=draw_car,
                                      sensor_length=sensor_length, viewtime=viewtime, matplotlib=matplotlib,
                                      opencv=opencv)

            if self.front_sensor_data is None and self.rear_sensor_data is None:
                raise ValueError("No sensor data given.")
        else:
            raise ValueError("No car data given.")


if __name__ == '__main__':
    """
    Data visualization example
    """
    front_path = 'PATH TO FRONT SENSOR DATA(DISTANCES)'
    rear_path = 'PATH TO REAR SENSOR DATA(DISTANCES)'
    car_data_path = 'PATH TO CAR DATA(replay.csv)'
    visualizer = DistancesVisualizer(car_data_path=None, front_data_path=front_path, rear_data_path=rear_path,
                                     debug_window=False)
    visualizer.visualize_sensor_data(500, 500, draw_car=True, batch_size=16, opencv=True)
