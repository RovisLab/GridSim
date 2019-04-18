from car_kinematic_model import Simulator
import pygame


class CustomExampleSimulator(Simulator):
    """
    Custom class example
    """
    def __init__(self, screen, screen_width, screen_height, car_x=5, car_y=27, sensor_size=50, rays_nr=8,
                 activations=False, traffic=True, record_data=False, replay_data_path=None, state_buf_path=None,
                 sensors=False, distance_sensor=False, enabled_menu=False):
        super().__init__(screen, screen_width, screen_height, car_x, car_y, sensor_size, rays_nr, activations, traffic,
                         record_data, replay_data_path, state_buf_path, sensors, distance_sensor, enabled_menu)

    def custom(self, *args):
        """
        Here is where you modify/access the simulator data.
        You can modify in the source code as well, but it's higher risk to break the simulator flow.
        :param args:
        :return:
        """
        super().custom(*args)

        # LOOP THROUGH ARGUMENTS
        for arg in args:
            print(arg)

        # ACCESS DATA EXAMPLE
        car_data = self.access_simulator_data(car_data=True)
        sensor_data = self.access_simulator_data(visual_sensor_data=True)
        if sensor_data is not None:
            sensor_img = self.convert_surface_to_opencv_img(sensor_data)
            print("Visual sensor img. shape: ", sensor_img.shape)
        distance_data = self.access_simulator_data(distance_sensor_data=True)
        print("Distance data: ", distance_data)
        print("Car data: ", car_data)

        # MODIFY CAR DATA EXAMPLE
        self.car.velocity.x = 5
        if self.dt is not None:
            self.car.update(self.dt)

    def custom_with_arguments_example(self):
        self.custom("argument1", "argument2")


if __name__ == '__main__':
    screen = pygame.display.set_mode((1280, 720))
    custom_sim = CustomExampleSimulator(screen, 1280, 720, distance_sensor=True)
    custom_sim.custom_with_arguments_example()
    custom_sim.run()

    # IN ORDER TO RECORD DATA PLEASE SPECIFY IN CONSTRUCTOR LIKE THIS
    # REPLAY_PATH = "/path/to/replay/data"
    # STATE_BUF_PATH = "path/to/state_buf/data"
    # custom_sim = CustomExampleSimulator(screen, 1280, 720, record_data=True, replay_data_path=REPLAY_PATH,
    #                                     state_buf_path=STATE_BUF_PATH)

    # FOR PERFORMANCE REASON, IN ORDER TO RECORD IMAGES FROM YOUR RUN PLEASE USE REPLAY
    # EXAMPLE:
    # from replay import Replay
    # replay = Replay(screen, 1280, 720)
    # replay.record_from_replay("D:/GridSim Dataset/run_replays/training_with_reference/run1.csv",
    #                           save_sensor_frame=True,
    #                           save_debug_frame=True,
    #                           save_sensor_frame_path="D:/GridSim Dataset/test_contest/sensor",
    #                           save_debug_frame_path="D:/GridSim Dataset/test_contest/debug",
    #                           display_obstacle_on_screen=True)
