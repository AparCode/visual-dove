# FINAL PROJECT: VisualDove - Main File
# @author: Aparnaa Senthilnathan
# 3.12.7
# ----------------------------------

import pygame
from pygame import mixer
import librosa
import numpy as np

class AudioBar:
    # initalize the audiobar class with x, y, colors, frequencies, decibels, height, and width parameters
    def __init__(self, x, y, freq, c1, c2, width=50, min_height=10, max_height=100, min_decibel=-80, max_decibel=0):
        self.x = x
        self.y = y
        self.freq = freq

        self.color = c1
        self.flip_color = c2

        self.width = width
        self.min_height = min_height
        self.max_height = max_height

        self.height = min_height

        self.min_decibel = min_decibel
        self.max_decibel = max_decibel

        self.decibel_height_ratio = (self.max_height - self.min_height)/(self.max_decibel - self.min_decibel)

    # update the AudioBar class based on the noise level in decibels.
    def update(self, dt, decibel):
        desired_height = decibel * self.decibel_height_ratio + self.max_height
        speed = (desired_height - self.height)/0.1

        self.height += speed * dt
        self.height = clamp(self.min_height, self.max_height, self.height)

    # render the AudioBar class based on the updated values
    # also create the flipped variation of the AudioBar class
    def render(self, screen):
        y = (self.y + self.max_height - self.height)
        he = self.height
        y_flip = 364
        he_flip = y
        pygame.draw.rect(screen, self.color, (self.x, y, self.width, he))
        # TO BE FIXED LATER
        pygame.draw.rect(screen, self.flip_color, (self.x, y_flip, self.width, he_flip))
        # pygame.draw.rect(screen, self.flip_color[int(self.y)//8][int(self.x)//8], (self.x, y_flip, self.width, he_flip))


# clamp the function based on the current height
def clamp(min_n, max_n, n):
    if n < min_n:
        return min_n
    elif n > max_n:
        return max_n
    return n

# read in the color grid from another file
def read_color_grid(filename):
    image_width = 128
    image_height = 96

    with open(filename, 'r') as f:
        lines = f.readlines()
        arr = []
        for li in lines:
            test = li.strip().split(" ")
            if test != ['']:
                te = list(map(int, test))
                arr.append(te)
            else:
                arr.append("--")

        color_grid = []
        for r in range(image_height):
            r_arr = []
            for c in range(image_width):
                if arr[(r*(c-1))+c] != '--':
                    r_arr.append(arr[(r*(c-1))+c])
            if len(r_arr) < 128:
                r_arr.append(r_arr[len(r_arr)-1])
            color_grid.append(r_arr)
    
    return color_grid


# get the decibel values based on target time and frequencies
def get_decibel(target_time, freq, time_index_ratio, frequencies_index_ratio):
    return spectrogram[int(freq * frequencies_index_ratio)][int(target_time * time_index_ratio)]

# main function
# import song file
song = "creamsodaredemo2.wav"

# create a matrix with amplitude values
time_series, sample_rate = librosa.load(song)
n_fft = 2048 * 4

# create a stft matrix which contains amplitude values 
stft = np.abs(librosa.stft(time_series, hop_length=512, n_fft=n_fft))

# create a spectrogram using the values of stft to convert into a matrix of decibels
spectrogram = librosa.amplitude_to_db(stft, ref=np.max) 

# get the array of frequencies
frequencies = librosa.core.fft_frequencies(n_fft=n_fft) 

# get an array of periodic times and index ratios of both time and frequencies
times = librosa.core.frames_to_time(np.arange(spectrogram.shape[1]), sr=sample_rate, hop_length=512, n_fft=n_fft)
time_index_ratio = len(times)/times[len(times) - 1]
frequencies_index_ratio = len(frequencies)/frequencies[len(frequencies)-1]

# initalize the pygame screen
pygame.init()
res = (1024,768)
screen = pygame.display.set_mode([res[0], res[1]]) 

# create an array of bars and a range of frequencies
bars = []
frequencies = np.arange(100, 8000, 100)
r = len(frequencies) # get the length of frequencies

# get the width, x, and y values
width = res[0] / r
x = (res[0] - width*r) / 2
y = 0

# read in the color_grid file
color_grid = read_color_grid("rgbref.txt")

# append bars as AudioBar classes for each frequency
for f in frequencies:
    bars.append(AudioBar(x,y,f,(217, 73, 252), (94, 30, 110), width, max_height=400))
    x += width

# get the time in ticks and the number of ticks in the last frame
t = pygame.time.get_ticks()
tick_last_frame = t

# load the pygame mixer and play the given song
mixer.music.load(song)
mixer.music.play(0)

# run the program
running = True
while running:
    # update time in ticks and the change in time
    t = pygame.time.get_ticks()
    change_in_time = (t - tick_last_frame) / 1000.0
    tick_last_frame = t

    # close the program if the x button is pressed
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # black screen
    screen.fill((0,0,0))

    # update and render each audio bar
    for b in bars:
        b.update(change_in_time, get_decibel(mixer.music.get_pos()/1000.0, b.freq, time_index_ratio, frequencies_index_ratio))
        b.render(screen)

    # flip the display for a more realistic representation (coordinates are displayed as negative)
    pygame.display.flip()

# end the program
pygame.quit()