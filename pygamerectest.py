import pygame
from pygame import mixer
import librosa
import numpy as np

pygame.init()

res = (1024,768)

screen = pygame.display.set_mode([res[0], res[1]]) 

color = (255, 0, 0)
color_blue = (0, 0, 255)
color_green = (0, 255, 0)

running = True
while running:
        # close the program if the x button is pressed
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.draw.rect(screen, color_blue, pygame.Rect(30,284,60,100))
    pygame.draw.rect(screen, color_green, pygame.Rect(30,384,60,100))
    pygame.draw.rect(screen, color, pygame.Rect(30,384,60,30))
    # pygame.draw.rect(screen, color_blue, pygame.Rect(30,384,60,60))
    # pygame.draw.rect(screen, color_green, pygame.Rect(30,30*3,60,120))
    # pygame.draw.rect(screen, color_blue, pygame.Rect(30,30*3,60,60))
    pygame.display.flip()