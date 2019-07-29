from car_kinematic_highway import HighwaySimulator
import pygame
from action_handler import apply_action


class CustomController(HighwaySimulator):
    """
    Controller example for highway simulator
    """
    def __init__(self, screen, screen_width, screen_height, car_x=3, car_y=27, sensor_size=50, rays_nr=8,
                 activations=False, highway_traffic=True, record_data=False, replay_data_path=None, state_buf_path=None,
                 sensors=False, distance_sensor=False, enabled_menu=False, highway_traffic_cars_nr=5,
                 ego_car_collisions=True, traffic_collisions=True):
        super().__init__(screen, screen_width, screen_height, car_x, car_y, sensor_size, rays_nr, activations,
                         highway_traffic, record_data, replay_data_path, state_buf_path, sensors, distance_sensor,
                         enabled_menu, highway_traffic_cars_nr, ego_car_collisions, traffic_collisions)

    def custom(self, *args):
        # CUSTOM FUNCTION IS INSIDE THE MAIN run FUNCTION THAT OVERRIDES
        # THE run FUNCTION FROM THE ABSTRACT Simulator CLASS
        super().custom(*args)
        # FOR MORE ACTIONS CHECK action_handler.py
        apply_action(0, self.car, self.dt)


if __name__ == '__main__':
    screen = pygame.display.set_mode((1280, 720))
    custom_controller = CustomController(screen, 1280, 720)
    # IF human_control IS SET TO FALSE, THE KEYBOARD INPUT IS OMITTED
    # IF human_control IS SET TO TRUE, THE KEYBOARD INPUT OVERRIDES THE CONTROLLER INPUT EVEN IF NO KEY IS PRESSED
    custom_controller.run(human_control=False)

