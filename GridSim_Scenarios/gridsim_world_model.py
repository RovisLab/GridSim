import pygame
from car_kinematic_highway import HighwaySimulator


class GridSimActionBasedController(object):
    def __init__(self, action_list):
        self.screen = pygame.display.set_mode((1280, 720))
        self.gridsim_object = HighwaySimulator(screen=self.screen, screen_width=1280, screen_height=720)
        self.action_list = action_list

    def start_simulation(self):
        self.gridsim_object.start_standby()

    def advance_and_get_observation(self, action):
        return self.gridsim_object.handle_action_and_get_observation(action=action)
