import pygame, pygame.font, pygame.event, pygame.draw
from pygame.locals import *
import config


def get_key():
    while 1:
        event = pygame.event.poll()
        if event.type == KEYDOWN:
            return event.key
        else:
            pass


def return_to_menu():
    from car_kinematic_city_menu import Menu
    menu = Menu()
    menu.main_menu()
    return


def display_box(screen, message, position, inside_c=config.background_color, border_c=config.background_color):
    " Print a message in a box in the middle of the screen "
    fontobject = pygame.font.SysFont(config.font, 18)
    position_rect = pygame.Rect(position[0], position[1], 300, 27)
    border_rect = pygame.Rect(position[0] - 1, position[1] - 1, 302, 29)

    active_surface = pygame.Surface((position_rect.width, position_rect.height))
    active_surface.set_alpha(225)

    pygame.draw.rect(active_surface, config.inactive_color, position_rect)
    pygame.draw.rect(screen, config.active_color, border_rect, 1)

    screen.blit(active_surface, position_rect.topleft)

    # pygame.draw.rect(screen, inside_c,
    #                  (position[0], position[1],
    #                   300, 21), 0)
    # pygame.draw.rect(screen, border_c,
    #                  (position[0], position[1],
    #                   304, 24), 1)
    if len(message) != 0:
        screen.blit(fontobject.render(message, 1, (255, 255, 255)),
                    (position[0] + 5, position[1] + 5))
    pygame.display.flip()


def ask(screen, question, input, position, nr_rows, current_box, inside_color=(0, 200, 0), border_color=config.background_color):
    " ask(screen, question) -> answer "
    pygame.font.init()
    display_box(screen, question + ": " + input, position, inside_color, border_color)
    while 1:
        inkey = get_key()
        # if user presses down change current box down
        if inkey == K_DOWN:
            if current_box < nr_rows - 1:
                current_box += 1
                r_tuple = (input, current_box)
                return r_tuple
        # if user presses down change current box up
        elif inkey == K_UP:
            if current_box > 0:
                current_box -= 1
                r_tuple = (input, current_box)
                return r_tuple
            else:
                return 0
        # if user presses enter
        elif inkey == K_RETURN:
            r_tuple = (input, False)
            return r_tuple
        # if user presses backspace clear the last char in input_box
        elif inkey == K_BACKSPACE:
            input = input[0:-1]
        elif inkey == K_MINUS:
            input += '_'
        elif inkey == K_SEMICOLON:
            input += ':'
        elif inkey == K_BACKSLASH:
            input += '/'
        # if user presses ESC return to menu
        elif inkey == K_ESCAPE:
            r_tuple = (input, True)
            return r_tuple
        # if user presses a normal key, update the box
        elif inkey <= 127:
            input += chr(inkey)
        # clear shift key use only the keyboard

        display_box(screen, question + ": " + input, position, inside_color, border_color)
