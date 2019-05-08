from car_kinematic_model import Simulator
import pygame


class CustomExampleSimulator(Simulator):
    """
    Custom class example
    """

    def __init__(self, screen, screen_width, screen_height, car_x=5, car_y=27, sensor_size=50, rays_nr=8,
                 activations=False, record_data=False, replay_data_path=None, state_buf_path=None, sensors=False,
                 distance_sensor=False, enabled_menu=False):
        # choose your backgrounds
        object_map_path = "resources/backgrounds/custom_example_obj.png"
        background_path = "resources/backgrounds/custom_example.png"
        car_image_path = "resources/cars/car_eb_2.png"
        # for this example we do not need traffic
        traffic_car_image_path = None
        object_car_image_path = None
        super().__init__(screen, screen_width, screen_height, car_x, car_y, sensor_size, rays_nr, activations,
                         record_data, replay_data_path, state_buf_path, sensors, distance_sensor, enabled_menu,
                         object_map_path, background_path, car_image_path, traffic_car_image_path,
                         object_car_image_path)

    def run(self):
        super().run()

        # we need this bool to check if the sensors are turned on
        # this field should be in any scenario
        mouse_button_pressed = False

        while not self.exit:

            # the flow should be split in 2 parts
            # first part should be the simulator construction which should be like this:
            # 1. CONSTRUCTION
            # update time
            self.dt = self.clock.get_time() / 1000
            # check the mouse click events
            self.event_handler(mouse_button_pressed)

            # take user input for our car
            self.key_handler(self.dt, [])

            # draw the environment
            self.draw_sim_environment(print_coords=True)

            # update the car behaviour
            self.car.update(self.dt)

            # check the sensors for activations
            self.activate_sensors()

            # 2. CUSTOM FUNCTIONALITY
            # the second part should be data management, controller setup or anything that you want to access your
            # simulator
            # let's say we want to print the car and distance sensor data from the simulator:
            print("Car data: ", self.access_simulator_data(car_data=True))
            print("Distance sensor data: ", self.access_simulator_data(distance_sensor_data=True))

            # leave the custom function tab always open in case you want to add something from another simulator that
            # implements this simulator
            self.custom()

            # and in the last place update the screen and frames
            pygame.display.update()
            self.clock.tick(self.ticks)

        # when the user quits the simulator end the pygame process too
        pygame.quit()


# let's make another simulator that implements this simulator but the resolution is smaller
class CustomExampleSimulator2(CustomExampleSimulator):

    def __init__(self, screen, screen_width, screen_height, car_x=5, car_y=27, sensor_size=50, rays_nr=8,
                 activations=False, record_data=False, replay_data_path=None, state_buf_path=None, sensors=False,
                 distance_sensor=False, enabled_menu=False):
        super().__init__(screen, screen_width, screen_height, car_x, car_y, sensor_size, rays_nr, activations,
                         record_data, replay_data_path, state_buf_path, sensors, distance_sensor, enabled_menu)
        # if you run with only the code above it will be exactly like the CustomExampleSimulator
        # let's make it 2 times smaller
        # now let's make the car image and the background smaller to fit the new resolution
        self.car_image = pygame.transform.scale(self.car_image, (int(self.car_image.get_width()/2),
                                                                 int(self.car_image.get_height()/2)))
        # now let's make the background smaller
        self.background = pygame.transform.scale(self.background, (int(self.bgWidth/2), int(self.bgHeight/2)))
        # don't forget about the object map!
        self.object_map = pygame.transform.scale(self.object_map, (int(self.bgWidth/2), int(self.bgHeight/2)))
        # but in order for the scale to be taken in consideration we have to update the bgWidth and bgHeight
        self.bgWidth, self.bgHeight = self.background.get_rect().size
        # this made the pygame screen smaller
        self.screen_height = self.bgHeight
        self.screen_width = self.bgWidth
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

    def custom_drawing(self, *args):
        # if you want to draw something in the simulator override this function
        super().custom_drawing(*args)
        pygame.draw.circle(self.screen, (255, 0, 0), (self.bgWidth - 10, 20), 5)


if __name__ == '__main__':
    # now let's see how the simulator that we've build is running
    # but first initialize a screen
    # define a width and height of the screen
    w, h = 1280, 720
    screen = pygame.display.set_mode((w, h))
    # custom_simulator = CustomExampleSimulator(screen, w, h)
    # custom_simulator.run()
    custom_simulator = CustomExampleSimulator2(screen, w, h)
    custom_simulator.run()


