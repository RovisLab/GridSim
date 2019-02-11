import pygame
import config


class Button:
    def __init__(self, message, coords, dimensions, inactive_color, active_color):
        self.message = message
        self.coords = coords
        self.dimensions = dimensions
        self.inactive_color = inactive_color
        self.active_color = active_color
        self.border_size = 4

    def draw_button(self, screen, text_offset):
        border_rect = pygame.Rect(self.coords[0]-self.border_size, self.coords[1]-self.border_size,
                                  self.dimensions[0]+self.border_size*2, self.dimensions[1]+self.border_size*2)
        rect = pygame.Rect(self.coords[0], self.coords[1], self.dimensions[0], self.dimensions[1])

        active_surface = pygame.Surface((rect.width, rect.height))
        active_surface.set_alpha(128)

        pygame.draw.rect(active_surface, self.inactive_color, rect)
        pygame.draw.rect(screen, self.active_color, border_rect, 8)

        screen.blit(active_surface, rect.topleft)
        font = pygame.font.SysFont(config.font, 40)
        text = font.render(self.message, True, (212, 239, 223))
        screen.blit(text, (self.coords[0] + text_offset[0], self.coords[1] + text_offset[1]))

    def button_light(self, screen, text_offset):
        rect = pygame.Rect(self.coords[0], self.coords[1], self.dimensions[0], self.dimensions[1])
        pygame.draw.rect(screen, self.active_color, rect)
        font = pygame.font.SysFont(config.font, 40)
        text = font.render(self.message, True, (212, 239, 223))
        screen.blit(text, (self.coords[0] + text_offset[0], self.coords[1] + text_offset[1]))
