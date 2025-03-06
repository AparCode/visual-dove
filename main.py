import pygame
import sys
import pyaudio
import math

# song = "winterfall_example.mp3"

# mixer.init()
# mixer.music.load(song)
# mixer.music.set_volume(0.1)
# mixer.music.play()

# while True:
#     print("Press e to exit the program")
#     command = input(" ")

#     if command == 'e':
#         mixer.music.stop()
#         break

  
# initialize constructor
pygame.init() 
  
# screen dimensions
res = (1024, 768) 
pygame.init()

screen = pygame.display.set_mode((res[0], res[1]))
clock = pygame.time.Clock()

# audio
chunk = 1024
f = pyaudio.paInt16
channels = 1
rate = 44100

p = pyaudio.PyAudio()
stream = p.open(format=f, channels = channels, rate = rate, input = True, frames_per_buffer=chunk)

def get_mic_input_level():
    data = stream.read(chunk)
    rms = 0
    for i in range(0, len(data), 2):
        sample = int.from_bytes(data[i:i + 2], byteorder='little', signed=True)
        rms += sample * sample
    rms = math.sqrt(rms/(chunk/2))
    return rms

def draw_sine_wave(amp):
    screen.fill((0,0,0))
    points = []
    if amp > 10:
        for x in range(res[0]):
            y = res[1]/2 + int(amp * math.sin(x * 0.02))
            points.append((x,y))
    else:
        points.append((0, res[1]/2))
        points.append((res[0], res[1]/2))
    
    pygame.draw.lines(screen, (223, 82, 255), False, points, 2)
    pygame.display.flip()

run = True
amp = 100

while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    amp_adj = get_mic_input_level() / 50
    amp = max(10, amp_adj)

    draw_sine_wave(amp)
    clock.tick(60)

# # opens up a window 
# screen = pygame.display.set_mode(res) 
  
# # white color 
# color = (255,255,255) 
  
# # light shade of the button 
# color_light = (170,170,170) 
  
# # dark shade of the button 
# color_dark = (100,100,100) 
  
# # stores the width of the 
# # screen into a variable 
# width = screen.get_width() 
  
# # stores the height of the 
# # screen into a variable 
# height = screen.get_height() 
  
# # defining a font 
# smallfont = pygame.font.SysFont('Corbel',12) 
  
# # rendering a text written in 
# # this font 
# text = smallfont.render('quit' , True , color) 
  
# while True: 
      
#     for ev in pygame.event.get(): 
          
#         if ev.type == pygame.QUIT: 
#             pygame.quit() 
              
#         #checks if a mouse is clicked 
#         if ev.type == pygame.MOUSEBUTTONDOWN: 
              
#             #if the mouse is clicked on the 
#             # button the game is terminated 
#             if width/2 <= mouse[0] <= width/2+140 and height/2 <= mouse[1] <= height/2+40: 
#                 pygame.quit() 
                  
#     # fills the screen with a color 
#     screen.fill((0,0,0)) 
      
#     # stores the (x,y) coordinates into 
#     # the variable as a tuple 
#     mouse = pygame.mouse.get_pos() 
      
#     # if mouse is hovered on a button it 
#     # changes to lighter shade  
#     if width/2 <= mouse[0] <= width/2+140 and height/2 <= mouse[1] <= height/2+40: 
#         pygame.draw.rect(screen,color_light,[width/2,height/2,140,40]) 
          
#     else: 
#         pygame.draw.rect(screen,color_dark,[width/2,height/2,140,40]) 
      
#     # superimposing the text onto our button 
#     screen.blit(text , (width/2+50,height/2)) 
      
#     # updates the frames of the game 
#     pygame.display.update() 