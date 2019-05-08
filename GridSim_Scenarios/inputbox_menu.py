import inputbox
import pygame
import os
import config


class InputBoxMenu:
    def __init__(self, screen, nr_rows, position, questions, row_types):
        """

        :param screen: screen to be drawn
        :param nr_rows: nr of rows in inputbox
        :param position: position of inputbox
        :param questions: what to be asked in inputbox
        :param row_types: row types -> str: 'int', 'path', 'path + csv'
        """
        self.row_types = row_types
        self.nr_rows = nr_rows
        self.position = position
        self.screen = screen
        self.question_list = questions
        self.errors = ['not_found', 'not_file', 'not_correct', 'already_exists', 'empty_or_wrong_path']

        self.inputbox_w = 15
        self.inputbox_h = 29
        self.inputs = {}
        self.positions = {}
        self.exit = False
        self.fontobject = None
        self.create_inputbox_menu()

    def check_inputbox_input(self):

        for row in range(self.nr_rows):
            if type(self.row_types[row]) is str:
                if self.row_types[row] == 'name':
                    pass
                if self.row_types[row] == 'path':
                    try:
                        os.path.exists(self.inputs[row])
                    except FileNotFoundError:
                        os.makedirs(path=self.inputs[row])
                if self.row_types[row] == 'path + csv':
                    # try and create path to csv

                    split_path = self.inputs[row].rsplit('/', 1)
                    if os.path.exists(self.inputs[row]) is False:
                        try:
                            os.makedirs(split_path[0])
                        except FileNotFoundError:
                            return 'empty_or_wrong_path'
                        except FileExistsError:
                            pass

                        # try and create csv
                        try:
                            open(self.inputs[row], 'x')
                        except FileExistsError:
                            return 'already_exists'
                        except PermissionError:
                            return 'not_file'
                        finally:
                            open(self.inputs[row], 'r')
                    else:
                        if self.inputs[row].find('.csv') == -1 and self.inputs[row].find('.txt') == -1:
                            return 'not_file'

            if self.row_types[row] is int:
                try:
                    int(self.inputs[row])
                except ValueError:
                    return 'not_correct'
            if self.row_types[row] is float:
                try:
                    float(self.inputs[row])
                except ValueError:
                    return 'not_correct'
        return True

    def create_inputbox_menu(self):
        pygame.font.init()
        self.fontobject = pygame.font.SysFont(config.font, 30)
        for i in range(self.nr_rows):
            self.inputs[i] = ''
            self.positions[i] = (self.position[0], self.position[1] + i * self.inputbox_h)

    def display_boxes(self):
        for row in range(self.nr_rows):
            inputbox.display_box(self.screen, self.question_list[row] + ': ' + self.inputs[row],
                                 position=(self.position[0], self.position[1] + row * self.inputbox_h))

    def help(self):
        help_message = ['Enter -> confirm input',
                        'Backspace -> erase from the back of the input box',
                        'Up/Down keys -> move from box to box',
                        'ESC -> return to menu']
        nr_row = 1

        position_rect = pygame.Rect(self.position[0] - 170,
                                    self.position[1] + (self.nr_rows + nr_row) * self.inputbox_h - 5, 650, 135)
        border_rect = pygame.Rect(position_rect.x - 4, position_rect.y - 4, position_rect.width + 8,
                                  position_rect.height + 8)

        active_surface = pygame.Surface((position_rect.width, position_rect.height))
        active_surface.set_alpha(128)

        pygame.draw.rect(active_surface, config.inactive_color, position_rect)
        pygame.draw.rect(self.screen, config.active_color, border_rect, 8)

        self.screen.blit(active_surface, position_rect.topleft)

        for mess in help_message:
            self.screen.blit(self.fontobject.render(mess, 1, (255, 255, 255)),
                             (self.position[0] - 150, self.position[1] + (self.nr_rows + nr_row) * self.inputbox_h))
            nr_row += 1

    def current_box(self, curr_box):
        self.screen.blit(self.fontobject.render('Current box: ' + str(self.question_list[curr_box]), 1, (255, 255, 255)),
                         (self.position[0], self.position[1] - self.inputbox_h))

    def ask_boxes(self):
        current_box = 0

        while self.exit is False:
            self.display_boxes()
            answear = inputbox.ask(self.screen,
                                   self.question_list[current_box],
                                   self.inputs[current_box],
                                   position=self.positions[current_box],
                                   nr_rows=self.nr_rows,
                                   current_box=current_box)

            if type(answear) is tuple:
                if type(answear[1]) is bool:
                    if answear[1] is True:
                        self.inputs[current_box] = answear[0]
                        inputbox.return_to_menu()
                    elif answear[1] is False:
                        self.inputs[current_box] = answear[0]
                        break
                else:
                    self.inputs[current_box] = answear[0]
                    current_box = answear[1]
            elif type(answear) is str:
                self.inputs[current_box] = answear

        return self.inputs


# # EXAMPLE OF USAGE
# if __name__ == '__main__':
#     screen = pygame.display.set_mode((1280, 720))
#     questions = ['Name', 'Address', 'Random']
#     box_menu = InputBoxMenu(screen, len(questions), (300, 300), questions, ['path + csv', 'path', int])
#     inpts = box_menu.ask_boxes()

