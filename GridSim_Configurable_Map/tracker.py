# CLASS DESIGNED FOR TRACKING AND DRAWING ON A IMAGE TRAJECTORIES FROM GRIDSIM
import cv2
import os
import pygame
import copy


class Tracker(object):
    def __init__(self, map_width, map_height, ppu, car, car_image, minimap_path, minimap_type, recorded_minimap):
        self.car = car
        self.ppu = ppu
        self.car_image = car_image
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(os.path.join(self.current_dir, minimap_path)) is False:
            raise OSError('minimap_path does not exists.')
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        if recorded_minimap is None:
            self.minimap = cv2.imread(os.path.join(self.current_dir, minimap_path))
            self.minimap = self.scale_minimap(minimap_type)
        elif recorded_minimap is not None:
            self.minimap = cv2.imread(os.path.join(self.current_dir, recorded_minimap))
        self.minimap_car_route = pygame.surfarray.make_surface(self.minimap).convert_alpha()
        if int(map_height/self.minimap.shape[0]) == int(map_width/self.minimap.shape[1]):
            self.map_scaling_factor = int(map_width/self.minimap.shape[1])
        else:
            raise ValueError('map_size not matching')

        self.car_image = self.scale_car_image(minimap_type)
        self.window = cv2.namedWindow('Minimap', cv2.WINDOW_AUTOSIZE)

    def scale_minimap(self, minimap_type):
        if minimap_type not in ['big', 'medium', 'small']:
            raise ValueError('minimap_type not defined, please select following: big, medium, small')

        if minimap_type == 'big':
            big = cv2.resize(self.minimap, (0, 0), fx=0.5, fy=0.5)
            return big
        if minimap_type == 'medium':
            medium = cv2.resize(self.minimap, (0, 0), fx=0.25, fy=0.25)
            return medium
        if minimap_type == 'small':
            small = cv2.resize(self.minimap, (0, 0), fx=0.1, fy=0.1)
            return small

    def scale_car_image(self, minimap_type):

        self.car_image = pygame.transform.rotate(self.car_image, self.car.angle)
        self.car_image = pygame.surfarray.array3d(self.car_image)
        self.car_image = cv2.cvtColor(self.car_image, cv2.COLOR_BGR2RGB)
        if minimap_type == 'big':
            big = cv2.resize(self.car_image, (0, 0), fx=1, fy=1)
            return big
        if minimap_type == 'medium':
            medium = cv2.resize(self.car_image, (0, 0), fx=0.5, fy=0.5)
            return medium
        if minimap_type == 'small':
            small = cv2.resize(self.car_image, (0, 0), fx=0.25, fy=0.25)
            return small

    def track_car_movement(self, x_offset, y_offset, route_width=2, color=(0, 0, 255), save_minimap=False, minimap_name=None):
        pygame.draw.circle(self.minimap_car_route, color, (y_offset, x_offset), route_width, route_width)
        minimap_image = pygame.surfarray.array3d(self.minimap_car_route)
        cv2.imshow('Minimap', minimap_image)
        if save_minimap is True and minimap_name is not None:
            self.save_tracked_data(minimap_image, minimap_name)

    def show_car_on_minimap(self, x_offset, y_offset, angle):
        minimap_surface = pygame.surfarray.make_surface(self.minimap).convert_alpha()
        car_surface = pygame.surfarray.make_surface(self.car_image).convert_alpha()
        car_surface = pygame.transform.rotate(car_surface, -90-angle).convert_alpha()
        minimap_surface.blit(car_surface, (y_offset, x_offset))

        minimap_image = pygame.surfarray.array3d(minimap_surface)
        cv2.imshow('Minimap', minimap_image)

    def scale_car_positions_to_minimap(self, center_x, center_y):
        pos_x = -self.car.position[0] * self.ppu / self.map_scaling_factor + center_x / self.map_scaling_factor
        pos_y = -self.car.position[1] * self.ppu / self.map_scaling_factor + center_y / self.map_scaling_factor
        return int(pos_x), int(pos_y)

    @staticmethod
    def save_tracked_data(minimap, minimap_name):
        cv2.imwrite('resources/' + minimap_name + '.png', minimap)


