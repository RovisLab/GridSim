import threading
import socketserver
from asynchronous.client import img_client
from car_kinematic_model_sandbox import ConfigurableSimulator
import pygame

action = None


def decode_action_number(action_nr):
    if action_nr == 0:
        return 'up'
    elif action_nr == 1:
        return 'down'
    elif action_nr == 2:
        return 'left'
    elif action_nr == 3:
        return 'right'
    elif action_nr == 4:
        return 'up, left'
    elif action_nr == 5:
        return 'up, right'
    elif action_nr == 6:
        return 'down, left'
    elif action_nr == 7:
        return 'down, right'


class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):

    def handle(self):
        data = str(self.request.recv(1024), 'ascii')
        actions = decode_action_number(int(data))
        simulator.received_action = str(actions).split(',')


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass


def create_simulator():
    CAR_IMAGE_PATH = 'resources/cars/car_eb_2.png'
    MAP_PATH = 'resources/backgrounds/scenario_b_4800x3252.jpg'
    OBJECT_MAP_PATH = 'resources/backgrounds/scenario_b_4800x3252_obj_map.jpg'
    RECORDED_MINIMAP = 'resources/backgrounds/minimap.png'
    RECORD_DATA_PATH = 'resources/recorded_data/run3'
    car_size = (32, 15)
    starting_position = (-130, -450, -90)
    scaling_factor = 1.5

    sim = ConfigurableSimulator(starting_position, CAR_IMAGE_PATH, MAP_PATH, OBJECT_MAP_PATH, car_size=car_size,
                                scaling_factor=scaling_factor)
    return sim


def send_img(simulator):
    while True:
        try:
            sub_rect = pygame.Rect((385, 150), (500, 445))
            img_to_send = simulator.screen.subsurface(sub_rect)
            img_to_send = pygame.image.tostring(img_to_send, 'RGB')

            car_pos_x = float(simulator.car.position.x)
            car_pos_y = float(simulator.car.position.y)
            car_angle = float(simulator.car.angle)
            car_velocity = float(simulator.car.velocity.x)

            state = bytes(str(car_pos_x) + ", " + str(car_pos_y) + ", " + str(car_angle) + ", " + str(car_velocity), 'ascii')
            img_client(ip, port, img_to_send, state, (sub_rect.width, sub_rect.height))

        except Exception as e:
            print(e)


if __name__ == "__main__":
    # Port 0 means to select an arbitrary unused port
    HOST, PORT = "localhost", 0

    server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)
    ip, port = server.server_address

    # Start a thread with the server -- that thread will then start one
    # more thread for each request
    server_thread = threading.Thread(target=server.serve_forever)
    # Exit the server thread when the main thread terminates
    server_thread.daemon = True
    server_thread.start()
    print("Server loop running in thread:", server_thread.name)

    simulator = create_simulator()
    simulator.car.max_velocity = 15
    simulator.car.max_acceleration = 15
    simulator.car.max_acceleration = 35

    image_thread = threading.Thread(target=send_img, args=(simulator,))
    image_thread.daemon = True
    image_thread.start()

    simulator.run()

    server.shutdown()
    server.server_close()

