import pygame
from pygame import mixer
import librosa
import numpy as np

def clamp(min_n, max_n, n):
    if n < min_n:
        return min_n
    if n > max_n:
        return max_n
    return n
class AudioBar:
    def __init__(self, x, y, freq, color, width=50, min_height=10, max_height=100, min_decibel=-80, max_decibel=0):
        self.x = x
        self.y = y
        self.freq = freq

        self.color = color

        self.width = width
        self.min_height = min_height
        self.max_height = max_height

        self.height = min_height

        self.min_decibel = min_decibel
        self.max_decibel = max_decibel

        self.decibel_height_ratio = (self.max_height - self.min_height)/(self.max_decibel - self.min_decibel)

    def update(self, dt, decibel):
        desired_height = decibel * self.decibel_height_ratio + self.max_height
        speed = (desired_height - self.height)/0.1

        self.height += speed * dt
        self.height = clamp(self.min_height, self.max_height, self.height)

    # --- GLOBAL ILLUMINATION STUFF GOES HERE ---
    def render(self, screen):
        pygame.draw.rect(screen, self.color, (self.x, self.y + self.max_height - self.height, self.width, self.height))


song = "winterfall_example.mp3"

# getting info from the file to create a matrix with amplitude values
time_series, sample_rate = librosa.load(song)
# getting a matrix which contains amplitude values according to frequency and time indexes
stft = np.abs(librosa.stft(time_series, hop_length=512, n_fft=2048*4))

spectrogram = librosa.amplitude_to_db(stft, ref=np.max)  # converting the matrix to decibel matrix

frequencies = librosa.core.fft_frequencies(n_fft=2048*4)  # getting an array of frequencies

# getting an array of time periodic
times = librosa.core.frames_to_time(np.arange(spectrogram.shape[1]), sr=sample_rate, hop_length=512, n_fft=2048*4)

time_index_ratio = len(times)/times[len(times) - 1]

frequencies_index_ratio = len(frequencies)/frequencies[len(frequencies)-1]


def get_decibel(target_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][int(target_time * time_index_ratio)]

pygame.init()

# infoObject = pygame.display.Info()

# screen_w = int(infoObject.current_w/2.5)
# screen_h = int(infoObject.current_w/2.5)

res = (1024,768)

screen = pygame.display.set_mode([res[0], res[1]]) 

bars = []
frequencies = np.arange(100, 8000, 100)
r = len(frequencies)

width = res[0] / r
x = (res[0] - width*r) / 2
y = 0


for c in frequencies:
    bars.append(AudioBar(x,y,c,(217, 73, 252), max_height=400, width=width))
    x += width

t = pygame.time.get_ticks()
tick_last_frame = t

mixer.music.load(song)
mixer.music.play(0)

running = True
while running:
    t = pygame.time.get_ticks()
    change_in_time = (t - tick_last_frame) / 1000.0
    tick_last_frame = t

    # close the program if the x button is pressed
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0,0,0))

    for b in bars:
        b.update(change_in_time, get_decibel(mixer.music.get_pos()/1000.0, b.freq))
        b.render(screen)

    pygame.display.flip()

pygame.quit()