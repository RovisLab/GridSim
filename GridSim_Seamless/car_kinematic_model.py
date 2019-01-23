import os
import pygame
import pygame.gfxdraw
from math import radians
from checkbox import Checkbox
from collision import Collision
from read_write_trajectory import write_data, write_state_buf
from car import Car
from math_util import *


class Simulator:
    def __init__(self, screen, screen_width, screen_height,
                 record_data=False):
        pygame.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = screen

        self.bkd_color = [0, 88, 0, 255]

        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.image_path = os.path.join(self.current_dir, "resources/cars/car_eb_2.png")
        self.car_image = pygame.image.load(self.image_path).convert_alpha()
        self.car_image = pygame.transform.scale(self.car_image, (42, 20))
        self.ppu = 16

        self.background = pygame.image.load(
        os.path.join(self.current_dir, "resources/backgrounds/seamless_road_2000_2000_Green.png")).convert()

        self.bgWidth, self.bgHeight = self.background.get_rect().size
        pygame.font.init()
        self.myfont = pygame.font.SysFont('Comic Sans MS', 30)

        self.input_image = pygame.surfarray.array3d(self.screen)

        self.clock = pygame.time.Clock()
        self.ticks = 60
        self.exit = False
        self.record_data = record_data

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

    def optimized_front_sensor(self, car, display_obstacle_on_sensor=False):
        # act_mask is a separate image where you can only see what the sensor sees
        center_rect = Collision.center_rect(self.screen_width, self.screen_height)
        mid_of_front_axle = Collision.point_rotation(car, -1, 16, center_rect)

        arc_points = get_arc_points(mid_of_front_axle, 150, radians(90 + car.angle), radians(270 + car.angle), 200)

        offroad_edge_points = []

        for end_point in arc_points:
            points_to_be_checked = list(get_equidistant_points(mid_of_front_axle, end_point, 25))

            check = False

            for line_point in points_to_be_checked:
                if np.array_equal(self.screen.get_at((int(line_point[0]), int(line_point[1]))), self.bkd_color):
                    check = True
                    break

            if check is False:
                offroad_edge_points.append(end_point)
            else:
                offroad_edge_points.append(line_point)

        for index in range(0, len(arc_points)):
            if offroad_edge_points[index] == arc_points[index]:
                pygame.draw.line(self.screen, (0, 255, 0), mid_of_front_axle, arc_points[index], True)
            else:
                pygame.draw.line(self.screen, (0, 255, 0), mid_of_front_axle, offroad_edge_points[index], True)
                if display_obstacle_on_sensor is True:
                    pygame.draw.line(self.screen, (255, 0, 0), offroad_edge_points[index], arc_points[index], True)

    def optimized_rear_sensor(self, car, display_obstacle_on_sensor=False):
        # act_mask is a separate image where you can only see what the sensor sees
        center_rect = Collision.center_rect(self.screen_width, self.screen_height)
        mid_of_rear_axle = Collision.point_rotation(car, 65, 16, center_rect)

        arc_points = get_arc_points(mid_of_rear_axle, 150, radians(-90 + car.angle), radians(90 + car.angle), 200)

        offroad_edge_points = []

        for end_point in arc_points:
            points_to_be_checked = list(get_equidistant_points(mid_of_rear_axle, end_point, 25))

            check = False

            for line_point in points_to_be_checked:
                if np.array_equal(self.screen.get_at((int(line_point[0]), int(line_point[1]))), self.bkd_color):
                    check = True
                    break

            if check is False:
                offroad_edge_points.append(end_point)
            else:
                offroad_edge_points.append(line_point)

        for index in range(0, len(arc_points)):
            if offroad_edge_points[index] == arc_points[index]:
                pygame.draw.line(self.screen, (0, 255, 0), mid_of_rear_axle, arc_points[index], True)
            else:
                pygame.draw.line(self.screen, (0, 255, 0), mid_of_rear_axle, offroad_edge_points[index], True)
                if display_obstacle_on_sensor is True:
                    pygame.draw.line(self.screen, (255, 0, 0), offroad_edge_points[index], arc_points[index], True)

    def enable_sensor(self, car, data_screen, draw_screen):
        center_rect = Collision.center_rect(self.screen_width, self.screen_height)
        mid_of_front_axle = Collision.point_rotation(car, 0, 16, center_rect)
        # mid_of_rear_axle = Collision.point_rotation(car, 65, 16, center_rect)
        # pygame.draw.circle(self.screen, (255, 255, 0), (mid_of_front_axle[0], mid_of_front_axle[1]), 5)
        distance = np.array([])
        for angle_index in range(120, 240, 24):
            distance = np.append(distance,
                                 self.compute_sensor_distance(car, mid_of_front_axle, 200, angle_index, data_screen,
                                                              self.screen))
        # for angle_index in range(300, 360, 4):
        #     self.compute_sensor_distance(car, mid_of_rear_axle, 200, angle_index, data_screen, draw_screen)
        # for angle_index in range(0, 60, 4):
        #     self.compute_sensor_distance(car, mid_of_rear_axle, 200, angle_index, data_screen, draw_screen)
        return distance

    def draw_sim_environment(self, car, cbox_front_sensor, cbox_rear_sensor, print_coords=False,
                             record_coords=False, file_path=None, file_name=None):
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

        # Record ego_car positions in GridSim
        if record_coords is True:
            if file_name is None:
                print('no file name given')
                quit()
            if file_path is None:
                print('no file path given')
                quit()
            write_data(file_path + '/' + file_name, round(stagePosX, 2), round(stagePosY, 2), round(rel_x, 2),
                       round(rel_y, 2), (round(car.velocity.x, 2) * self.ppu/4))

        return stagePosX, stagePosY, rel_x, rel_y

    def key_handler(self, car, dt, rs_pos_list):
        # User input
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_ESCAPE]:
            self.return_to_menu()
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

    def run(self):
        # place car on road
        car = Car(10, 125)

        # sensor checkboxes on top right corner
        cbox_front_sensor = Checkbox(self.screen_width - 200, 10, 'Enable front sensor', False)
        cbox_rear_sensor = Checkbox(self.screen_width - 200, 35, 'Enable rear sensor', False)

        # reset position list -> to be updated
        rs_pos_list = [[650, 258, 90.0], [650, 258, 270.0], [0, 0, 180.0], [0, 0, 0.0], [302, 200, 45.0],
                       [40, 997, 0.0], [40, 997, 180.0], [100, 997, 0.0], [100, 997, 180.0], [400, 998, 0.0],
                       [400, 998, 180.0], [385, 315, 135.0]]

        # boolean variable needed to check for single-click press
        mouse_button_pressed = False

        if self.record_data is True:
            index_image = 0

        while not self.exit:
            # VARIABLE_UPDATE

            dt = self.clock.get_time() / 1000
            self.event_handler(cbox_front_sensor, cbox_rear_sensor, mouse_button_pressed)

            # LOGIC
            self.key_handler(car, dt, rs_pos_list)
            car.acceleration = max(-car.max_acceleration, min(car.acceleration, car.max_acceleration))
            car.steering = max(-car.max_steering, min(car.steering, car.max_steering))

            # DRAWING
            stagePos = self.draw_sim_environment(car, cbox_front_sensor, cbox_rear_sensor, print_coords=True)
            relPos = (stagePos[2], stagePos[3])
            stagePos = (stagePos[0], stagePos[1])
            car.update(dt)

            act_mask = pygame.Surface((self.screen_width, self.screen_height))
            if cbox_front_sensor.isChecked():
                self.optimized_front_sensor(car, display_obstacle_on_sensor=True)
            if cbox_rear_sensor.isChecked():
                self.optimized_rear_sensor(car, display_obstacle_on_sensor=True)

            if self.record_data is True:
                image_name = 'image_' + str(index_image) + '.png'
                index_image += 1

            if self.record_data is True:
                # RECORD TAB

                actions = [car.position.x, car.position.y, float(round(car.angle, 3)), float(round(car.acceleration, 3)),
                           float(round(car.velocity.x, 3)), image_name]

                # Save state_buf
                write_state_buf(self.current_dir.replace('\\', '/') + '/GridSim data/state_buf.csv', actions)

                # Save replay
                write_data(self.current_dir.replace('\\', '/') + '/GridSim data/replay_data.csv', car.position, car.angle)

            pygame.display.update()
            self.clock.tick(self.ticks)

        pygame.quit()


if __name__ == '__main__':
    screen = pygame.display.set_mode((1280, 720))
    game = Simulator(screen, 1280, 720, record_data=False)
    game.run()
