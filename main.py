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

    def __init__(self, ka, kd, ks, ke, co):
        self.ka = ka
        self.kd = kd
        self.ks = ks
        self.ke = ke
        self.co = co

    def illuminate(self, id: IntersectData, world, depth):
        # ---- CHECKPOINT 3 ----
        # obj = id.obj
        # co = obj.color
        co = self.co
        light = id.LightList[0]
        in_pt = id.pt
        norm = id.norm

        # spawn shadowRay
        shadowRay = Ray(Vector(in_pt), Vector(light.pos))

        # getting vectors
        N = Vector(norm)
        S = shadowRay.direction
        Re = reflection(S, N, depth)
        V = camera_eyept

        batch = someVectors(N,S,V,Re)

        # normalizing vectors
        N = N.normalize()
        S = S.normalize()
        Re = Re.normalize()
        V = camera_eyept


        # check if the ray hits any other object
        objectHit = False
        for o in world.objectList:
            if o is not id.obj:
                ip, sur_norm, dist = o.intersect(shadowRay)
                if ip != None:
                    objectHit = True
    
        # if the shadowRay hits any other object, we only calculate the ambient
        if objectHit:
            # only calculate the ambient portion of L
            # L = ka(color)La
            ka = self.ka
            la = light.color

            r = ka * co.r * la.r
            g = ka * co.g * la.g
            b = ka * co.b * la.b

            L = Color(r*255,g*255,b*255)

        else:
            # use the full Phong formula!
            # setting up values
            ka = self.ka
            kd = self.kd
            ks = self.ks
            ke = self.ke
            la = light.color
            cs = id.obj.spec_color
            li = light.color


            # calculating ambient colors
            r_am = ka * co.r * la.r
            g_am = ka * co.g * la.g
            b_am = ka * co.b * la.b

            # calculating diffuse colors
            r_di = kd * (li.r * co.r * S.dot(N))
            g_di = kd * (li.g * co.g * S.dot(N))
            b_di = kd * (li.b * co.b * S.dot(N))

            # calculating specular colors
            rv = Re.dot(V)
            if rv == "nan":
                rv = 0
            r_spec = ks * (li.r * cs.r * (rv**ke))
            g_spec = ks * (li.r * cs.r * (rv**ke))
            b_spec = ks * (li.r * cs.r * (rv**ke))

            # adding the values together
            r = r_am + r_di + r_spec
            g = g_am + g_di + g_spec
            b = b_am + b_di + b_spec

            L = Color(r*255,g*255,b*255)

        return L, batch

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
        # self.height = self.max_height

        self.height += speed * dt
        self.height = clamp(self.min_height, self.max_height, self.height)

    # --- GLOBAL ILLUMINATION STUFF GOES HERE ---
    def render(self, screen):
        pygame.draw.rect(screen, self.color, (self.x, self.y + self.max_height - self.height, self.width, self.height))

class AudioBar:
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

    def update(self, dt, decibel):
        desired_height = decibel * self.decibel_height_ratio + self.max_height
        speed = (desired_height - self.height)/0.1
        # self.height = self.max_height

        self.height += speed * dt
        self.height = clamp(self.min_height, self.max_height, self.height)

    # --- GLOBAL ILLUMINATION STUFF GOES HERE ---
    def render(self, screen):
        y = (self.y + self.max_height - self.height)
        he = self.height
        y_flip = 364
        he_flip = y
        pygame.draw.rect(screen, self.color, (self.x, y, self.width, he))
        pygame.draw.rect(screen, self.flip_color, (self.x, y_flip, self.width, he_flip))

song = "creamsodaredemo2.wav"

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
    bars.append(AudioBar(x,y,c,(217, 73, 252), (94, 30, 110), max_height=400, width=width))
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