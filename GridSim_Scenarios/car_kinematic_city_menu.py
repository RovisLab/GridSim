import pygame
from car_kinematic_city import CitySimulator
from replay import Replay
from button import Button
from checkbox import Checkbox
import config
from inputbox_menu import InputBoxMenu
from car_kinematic_ga import NeuroEvolutionary
from car_kinematic_dqn import DqnSimulator
import os
import time
import cv2
import numpy as np
import threading


class Menu:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("GridSim")
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.screen_width = config.screen_width
        self.screen_height = config.screen_height
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.screen.fill([255, 255, 255])
        self.exit = False
        self.manual_driving_button = Button("manual driving", (500, 150), (350, 50), config.inactive_color, config.active_color)

        self.ga_button = Button("neuro-evolutionary", (500, 200), (350, 50), config.inactive_color, config.active_color)
        self.ga_train_button = Button("→ train", (500, 250), (350, 50), config.inactive_color, config.active_color)
        self.ga_predict_button = Button("→ predict", (500, 300), (350, 50), config.inactive_color, config.active_color)

        self.drl_button = Button("deep reinf. learning", (500, 250), (350, 50), config.inactive_color, config.active_color)  # DQN
        self.drl_train_button = Button("→ train", (500, 300), (350, 50), config.inactive_color, config.active_color)
        self.drl_predict_button = Button("→ predict", (500, 350), (350, 50), config.inactive_color, config.active_color)

        self.record_button = Button("record data", (500, 300), (350, 50), config.inactive_color, config.active_color)
        self.replay_button = Button("replay", (500, 350), (350, 50), config.inactive_color, config.active_color)

        self.back_button = Button("back", (500, 400), (350, 50), config.inactive_color, (200, 0, 0))
        self.exit_button = Button("exit", (500, 450), (350, 50), config.inactive_color, (200, 0, 0))

        self.activation_cbox = Checkbox(1000, 50, "Print activations", True)
        self.sensors_cbox = Checkbox(1000, 70, "Enable visual sensors", True)
        self.distance_sensor_cbox = Checkbox(1000, 90, "Enable distance sensor", False)
        self.traffic_cbox = Checkbox(1000, 110, "Enable traffic", True)

        self.ga_buttons_interactions = False
        self.drl_buttons_interactions = False

        self.background_vid = cv2.VideoCapture("resources/backgrounds/zoomed-background-video.mp4")
        if self.background_vid.isOpened() is False:
            raise OSError("Error opening video stream or file")
        self.current_background_frame = None
        self.pause_background_video = False

        pygame.display.update()

    def display_error_message(self, error_message, position=(20, 20), sleep_time=2):
        font_render = pygame.font.SysFont(config.font, 40)
        font_render.set_bold(True)
        error_text = font_render.render(error_message, True, (250, 0, 0))
        self.screen.blit(error_text, position)
        pygame.display.update()
        time.sleep(sleep_time)

    def background_video(self):

        frame_counter = 0
        while self.background_vid.isOpened():

            if self.pause_background_video is True:
                break
            ret, frame = self.background_vid.read(0)
            frame_counter += 1

            if ret is True:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = np.rot90(frame)
                frame = np.flipud(frame)
                self.current_background_frame = pygame.surfarray.make_surface(frame)

                if frame_counter == self.background_vid.get(cv2.CAP_PROP_FRAME_COUNT):
                    frame_counter = 0
                    self.background_vid.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.background_vid.release()

    def checkbox_interactions(self):
        self.activation_cbox_interaction()
        self.sensors_cbox_interaction()
        self.traffic_cbox_interaction()

    def buttons_interactions(self):
        tag = False
        self.manual_driving_button_interaction()

        self.ga_button_interaction()
        if self.ga_buttons_interactions is True:
            tag = True
            self.ga_train_button_interaction()
            self.ga_predict_button_interaction()
            self.back_button_interaction()

        if tag is False:
            self.drl_button_interaction()
        if self.drl_buttons_interactions is True:
            tag = True
            self.drl_train_button_interaction()
            self.drl_predict_button_interaction()
            self.back_button_interaction()

        if tag is False:
            self.record_button_interaction()
            self.replay_button_interaction()
            self.exit_button_interaction()

    def all_interactions(self):
        self.activation_cbox_interaction()
        self.sensors_cbox_interaction()
        self.traffic_cbox_interaction()
        self.manual_driving_button_interaction()
        self.record_button_interaction()
        self.replay_button_interaction()
        self.exit_button_interaction()

    def manual_driving_button_interaction(self):
        mouse_pos = pygame.mouse.get_pos()
        if (self.manual_driving_button.coords[0] < mouse_pos[0] < self.manual_driving_button.coords[0] + self.manual_driving_button.dimensions[0] and
                self.manual_driving_button.coords[1] < mouse_pos[1] < self.manual_driving_button.coords[1] + self.manual_driving_button.dimensions[1]):
            self.manual_driving_button.button_light(self.screen, (75, -3))
            mouse_click = pygame.mouse.get_pressed()
            if mouse_click[0] == 1:
                self.pause_background_video = True
                questions = ['Sensor size']
                input_box = InputBoxMenu(self.screen, len(questions),
                                         (self.manual_driving_button.coords[0] + 25, self.manual_driving_button.coords[1] + 75),
                                         questions, [int])
                input_box.help()
                inputs = input_box.ask_boxes()
                check = input_box.check_inputbox_input()
                error_message_pos = [20, 20]
                while check in input_box.errors:
                    self.display_error_message('Error ' + check, position=tuple(error_message_pos), sleep_time=0)
                    error_message_pos[1] += 40
                    inputs = input_box.ask_boxes()
                    check = input_box.check_inputbox_input()

                sim = CitySimulator(self.screen, self.screen_width, self.screen_height,
                                    activations=self.activation_cbox.isChecked(), record_data=False,
                                    sensor_size=int(inputs[0]),
                                    traffic=self.traffic_cbox.isChecked(),
                                    sensors=self.sensors_cbox.isChecked(),
                                    distance_sensor=self.distance_sensor_cbox.isChecked(),
                                    enabled_menu=True)
                sim.run()
                quit()
        else:
            self.manual_driving_button.draw_button(self.screen, (75, -3))

    def record_button_interaction(self):
        mouse_pos = pygame.mouse.get_pos()
        if (self.record_button.coords[0] < mouse_pos[0] < self.record_button.coords[0] + self.record_button.dimensions[0] and
                self.record_button.coords[1] < mouse_pos[1] < self.record_button.coords[1] + self.record_button.dimensions[1]):
            self.record_button.button_light(self.screen, (100, -3))
            mouse_click = pygame.mouse.get_pressed()
            if mouse_click[0] == 1:
                questions = ['Sensor size', 'State_buf path', 'Replay_data path']
                input_box = InputBoxMenu(self.screen, len(questions),
                                         (self.record_button.coords[0] + 25, self.record_button.coords[1] + 75),
                                         questions, [int, 'path + csv', 'path + csv'])
                input_box.help()
                inputs = input_box.ask_boxes()
                check = input_box.check_inputbox_input()
                error_message_pos = [20, 20]

                while check in input_box.errors:
                    self.display_error_message('Error ' + check, position=tuple(error_message_pos), sleep_time=0)
                    error_message_pos[1] += 40
                    inputs = input_box.ask_boxes()
                    check = input_box.check_inputbox_input()

                rec = CitySimulator(self.screen, self.screen_width, self.screen_height, record_data=True,
                                    sensor_size=int(inputs[0]),
                                    state_buf_path=str(inputs[1]),
                                    replay_data_path=str(inputs[2]),
                                    traffic=self.traffic_cbox.isChecked(),
                                    sensors=self.sensors_cbox.isChecked(),
                                    distance_sensor=self.distance_sensor_cbox.isChecked(),
                                    enabled_menu=True)
                rec.run()
                quit()
        else:
            self.record_button.draw_button(self.screen, (100, -3))

    def replay_button_interaction(self):
        mouse_pos = pygame.mouse.get_pos()
        if (self.replay_button.coords[0] < mouse_pos[0] < self.replay_button.coords[0] + self.replay_button.dimensions[0] and
                self.replay_button.coords[1] < mouse_pos[1] < self.replay_button.coords[1] + self.replay_button.dimensions[1]):
            self.replay_button.button_light(self.screen, (125, -3))
            mouse_click = pygame.mouse.get_pressed()
            if mouse_click[0] == 1:
                questions = ['Sensor size', 'Replay_data path']
                input_box = InputBoxMenu(self.screen, len(questions),
                                         (self.replay_button.coords[0] + 25, self.replay_button.coords[1] + 75),
                                         questions, [int, 'path + csv'])
                input_box.help()
                inputs = input_box.ask_boxes()
                check = input_box.check_inputbox_input()
                error_message_pos = [20, 20]

                while check in input_box.errors:
                    self.display_error_message('Error ' + check, position=tuple(error_message_pos), sleep_time=0)
                    error_message_pos[1] += 40
                    inputs = input_box.ask_boxes()
                    check = input_box.check_inputbox_input()

                replay = Replay(self.screen, self.screen_width, self.screen_height,
                                activations=self.activation_cbox.isChecked(),
                                traffic=self.traffic_cbox.isChecked(),
                                sensors=self.sensors_cbox.isChecked(),
                                distance_sensor=self.distance_sensor_cbox.isChecked(),
                                sensor_size=int(inputs[0]),
                                enabled_menu=True)
                replay.replay(inputs[1], enable_trajectory=True)
                quit()
        else:
            self.replay_button.draw_button(self.screen, (125, -3))

    def ga_button_interaction(self):
        mouse_pos = pygame.mouse.get_pos()
        if (self.ga_button.coords[0] < mouse_pos[0] < self.ga_button.coords[0] + self.ga_button.dimensions[0] and
                self.ga_button.coords[1] < mouse_pos[1] < self.ga_button.coords[1] + self.ga_button.dimensions[1]):
            self.ga_button.button_light(self.screen, (55, -3))
            mouse_click = pygame.mouse.get_pressed()
            if mouse_click[0] == 1:
                self.ga_buttons_interactions = True
        else:
            self.ga_button.draw_button(self.screen, (55, -3))

    def ga_train_button_interaction(self):
        mouse_pos = pygame.mouse.get_pos()
        if (self.ga_train_button.coords[0] < mouse_pos[0] < self.ga_train_button.coords[0] + self.ga_train_button.dimensions[0] and
                self.ga_train_button.coords[1] < mouse_pos[1] < self.ga_train_button.coords[1] + self.ga_train_button.dimensions[1]):
            self.ga_train_button.button_light(self.screen, (110, -3))
            mouse_click = pygame.mouse.get_pressed()
            if mouse_click[0] == 1:
                self.pause_background_video = True
                questions = ['No_population', 'No_generations', 'Rays_nr']
                input_box = InputBoxMenu(self.screen, len(questions),
                                         (self.ga_train_button.coords[0] + 25, self.ga_train_button.coords[1] + 75),
                                         questions, [int, int, int])
                input_box.help()
                inputs = input_box.ask_boxes()
                check = input_box.check_inputbox_input()
                error_message_pos = [20, 20]

                while check in input_box.errors:
                    self.display_error_message('Error ' + check, position=tuple(error_message_pos), sleep_time=0)
                    error_message_pos[1] += 40
                    inputs = input_box.ask_boxes()
                    check = input_box.check_inputbox_input()

                agent = NeuroEvolutionary(self.screen, self.screen_width, self.screen_height, 0, False, False, False,
                                          None, None, False,
                                          population_size=int(inputs[0]),
                                          num_generations=int(inputs[1]),
                                          shape=int(inputs[2]))
                agent.neuro_trainer.train(agent.kinematic_ga.neuro_eval)
                quit()
        else:
            self.ga_train_button.draw_button(self.screen, (110, -3))

    def ga_predict_button_interaction(self):
        mouse_pos = pygame.mouse.get_pos()
        if (self.ga_predict_button.coords[0] < mouse_pos[0] < self.ga_predict_button.coords[0] + self.ga_predict_button.dimensions[0] and
                self.ga_predict_button.coords[1] < mouse_pos[1] < self.ga_predict_button.coords[1] + self.ga_predict_button.dimensions[1]):
            self.ga_predict_button.button_light(self.screen, (110, -3))
            mouse_click = pygame.mouse.get_pressed()
            if mouse_click[0] == 1:
                self.pause_background_video = True
                questions = ['Model name']
                input_box = InputBoxMenu(self.screen, len(questions),
                                         (self.ga_predict_button.coords[0] + 25, self.ga_predict_button.coords[1] + 75),
                                         questions, ['name'])
                input_box.help()
                inputs = input_box.ask_boxes()

                if os.path.exists('./used_models/ga/' + inputs[0] + '.h5') is False:
                    self.display_error_message("Model doesn't exists in /used_models/ga/. Loading default model.")
                    # replace with standard_model after training a good model ↓
                    inputs[0] = 'model_2000'

                agent = NeuroEvolutionary(self.screen, self.screen_width, self.screen_height, 0, False, False, False,
                                          None, None, False)
                agent.kinematic_ga.load_model(inputs[0])

                while pygame.key.get_pressed() != pygame.K_ESCAPE:
                    agent.ga_sim.run_ga(agent.kinematic_ga.model)
                quit()
        else:
            self.ga_predict_button.draw_button(self.screen, (110, -3))

    def drl_button_interaction(self):
        mouse_pos = pygame.mouse.get_pos()
        if (self.drl_button.coords[0] < mouse_pos[0] < self.drl_button.coords[0] + self.drl_button.dimensions[0] and
                self.drl_button.coords[1] < mouse_pos[1] < self.drl_button.coords[1] + self.drl_button.dimensions[1]):
            self.drl_button.button_light(self.screen, (25, -3))
            mouse_click = pygame.mouse.get_pressed()
            if mouse_click[0] == 1:
                self.drl_buttons_interactions = True
        else:
            self.drl_button.draw_button(self.screen, (25, -3))

    def drl_train_button_interaction(self):
        mouse_pos = pygame.mouse.get_pos()
        if (self.drl_train_button.coords[0] < mouse_pos[0] < self.drl_train_button.coords[0] + self.drl_train_button.dimensions[0] and
                self.drl_train_button.coords[1] < mouse_pos[1] < self.drl_train_button.coords[1] + self.drl_train_button.dimensions[1]):
            self.drl_train_button.button_light(self.screen, (110, -3))
            mouse_click = pygame.mouse.get_pressed()
            if mouse_click[0] == 1:
                self.pause_background_video = True
                questions = ['No_episodes', 'Rays_nr']
                input_box = InputBoxMenu(self.screen, len(questions),
                                         (self.drl_train_button.coords[0] + 25, self.drl_train_button.coords[1] + 75),
                                         questions, [int, int])
                input_box.help()
                inputs = input_box.ask_boxes()
                check = input_box.check_inputbox_input()
                error_message_pos = [20, 20]

                while check in input_box.errors:
                    self.display_error_message('Error ' + check, position=tuple(error_message_pos), sleep_time=0)
                    error_message_pos[1] += 40
                    inputs = input_box.ask_boxes()
                    check = input_box.check_inputbox_input()

                agent = DqnSimulator(self.screen, self.screen_width, self.screen_height, 0, False, False, False,
                                          None, None, False, rays_nr=int(inputs[1]))
                agent.train_conv_dqn(int(inputs[0]))
                quit()
        else:
            self.drl_train_button.draw_button(self.screen, (110, -3))

    def drl_predict_button_interaction(self):
        mouse_pos = pygame.mouse.get_pos()
        if (self.drl_predict_button.coords[0] < mouse_pos[0] < self.drl_predict_button.coords[0] + self.drl_predict_button.dimensions[0] and
                self.drl_predict_button.coords[1] < mouse_pos[1] < self.drl_predict_button.coords[1] + self.drl_predict_button.dimensions[1]):
            self.drl_predict_button.button_light(self.screen, (110, -3))
            mouse_click = pygame.mouse.get_pressed()
            if mouse_click[0] == 1:
                self.pause_background_video = True
                questions = ['Rays_nr']
                input_box = InputBoxMenu(self.screen, len(questions),
                                         (self.drl_predict_button.coords[0] + 25, self.drl_predict_button.coords[1] + 75),
                                         questions, [int])
                input_box.help()
                inputs = input_box.ask_boxes()
                check = input_box.check_inputbox_input()
                error_message_pos = [20, 20]

                while check in input_box.errors:
                    self.display_error_message('Error ' + check, position=tuple(error_message_pos), sleep_time=0)
                    error_message_pos[1] += 40
                    inputs = input_box.ask_boxes()
                    check = input_box.check_inputbox_input()

                agent = DqnSimulator(self.screen, self.screen_width, self.screen_height, 0, False, False, False,
                                     None, None, False, rays_nr=int(inputs[0]))
                agent.predict_conv_dqn()
                quit()
        else:
            self.drl_predict_button.draw_button(self.screen, (110, -3))

    def exit_button_interaction(self):
        mouse_pos = pygame.mouse.get_pos()
        if (self.exit_button.coords[0] < mouse_pos[0] < self.exit_button.coords[0] +
                self.exit_button.dimensions[0] and
                self.exit_button.coords[1] < mouse_pos[1] < self.exit_button.coords[1] +
                self.exit_button.dimensions[1]):
            self.exit_button.button_light(self.screen, (140, -3))
            mouse_click = pygame.mouse.get_pressed()
            if mouse_click[0] == 1:
                quit()
        else:
            self.exit_button.draw_button(self.screen, (140, -3))

    def back_button_interaction(self):
        mouse_pos = pygame.mouse.get_pos()
        if (self.back_button.coords[0] < mouse_pos[0] < self.back_button.coords[0] +
                self.back_button.dimensions[0] and
                self.back_button.coords[1] < mouse_pos[1] < self.back_button.coords[1] +
                self.back_button.dimensions[1]):
            self.back_button.button_light(self.screen, (140, -3))
            mouse_click = pygame.mouse.get_pressed()
            if mouse_click[0] == 1:
                self.pause_background_video = False
                if self.ga_buttons_interactions is True:
                    self.ga_buttons_interactions = False
                elif self.drl_buttons_interactions is True:
                    self.drl_buttons_interactions = False
        else:
            self.back_button.draw_button(self.screen, (140, -3))

    def activation_cbox_interaction(self):
        mouse_pos = pygame.mouse.get_pos()
        if pygame.mouse.get_pressed()[0]:
            if self.activation_cbox.onCheckbox(mouse_pos):
                self.activation_cbox.changeState()
        self.activation_cbox.update()

    def sensors_cbox_interaction(self):
        mouse_pos = pygame.mouse.get_pos()
        if pygame.mouse.get_pressed()[0]:
            if self.sensors_cbox.onCheckbox(mouse_pos):
                self.sensors_cbox.changeState()
            elif self.distance_sensor_cbox.onCheckbox(mouse_pos):
                self.distance_sensor_cbox.changeState()
        self.sensors_cbox.update()
        self.distance_sensor_cbox.update()

    def traffic_cbox_interaction(self):
        mouse_pos = pygame.mouse.get_pos()
        if pygame.mouse.get_pressed()[0]:
            if self.traffic_cbox.onCheckbox(mouse_pos):
                self.traffic_cbox.changeState()
        self.traffic_cbox.update()

    def main_menu(self):
        video_thread = threading.Thread(target=self.background_video)
        video_thread.daemon = True
        video_thread.start()

        while not self.exit:
            if self.current_background_frame is not None:
                self.screen.blit(self.current_background_frame, (0, 0))
                # self.screen.fill(config.background_color)  # let menu simple
            self.checkbox_interactions()
            self.buttons_interactions()
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True
        pygame.quit()


if __name__ == '__main__':
    menu = Menu()
    menu.main_menu()
