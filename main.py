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

class Point:
    def __init__(self, p):
        self.p = p
        self.x = p[0]
        self.y = p[1]
        self.z = p[2]
    
    def transform(self):
        pass

    def add(self, p2):
        return np.add(self.p, p2.p)
    
    def subtract(self, p2):
        return np.subtract(self.p, p2.p)

    def dist(self, p2):
        return np.linalg.norm(self.p, p2.p)
class Vector:
    def __init__(self, v):
        self.v = v
        self.x = self.v[0]
        self.y = self.v[1]
        self.z = self.v[2]

    def add(self, v2):
        return np.add(self.v, v2.v)
    
    def subtract(self, v2):
        return np.subtract(self.v, v2.v)

    def cross(self, v2):
        return np.cross(self.v, v2.v)
    
    def dot(self, v2):
        return np.dot(self.v, v2.v)
    
    def length(self):
        return math.sqrt((self.x ** 2) + (self.y ** 2) + (self.z ** 2))

    def normalize(self):
        return Vector(self.v / np.linalg.norm(self.v))

    def scaling(self, n):
        return Vector([self.x * n, self.y * n, self.z * n])

    def power(self, n):
        return Vector([self.x ** n, self.y ** n, self.z ** n])
    
    def reverse(self):
        return self.scaling(-1)

    def transform(self):
        pass
class Ray:
    def __init__(self, o, d):
        self.origin = o
        self.direction = d
    
    def endpt(self, w):
        return self.origin + w*self.direction
class Color:
    def __init__(self, r, g, b):
        self.r = r / 255
        self.g = g / 255
        self.b = b / 255
        # self.color = [r,g,b]

    # translate color values to bytes
    def convert_to_bytes(self):
        return [self.r * 255, self.g * 255, self.b * 255]

    # add a new color by mixing with another color
    def addnewColor(self, c2):
        new_r = self.r + c2.r
        new_g = self.g + c2.g
        new_b = self.b + c2.b

        return Color(new_r * 255, new_g * 255, new_b * 255)

    def multiply(self, s):
        new_r = self.r * s
        new_g = self.g * s
        new_b = self.b * s

        return Color(new_r * 255, new_g * 255, new_b * 255)
class Light:
    def __init__(self, pos, color):
        self.pos = pos
        # save the radiance
        self.color = color
class World:
    def __init__(self):
        self.objectList = []
        self.attributes = []
        self.lightList = []
        # lightColor = Color(255,255,255)
        # ambient = Light(lightColor)

    def add(self, obj):
        self.objectList.append(obj)

    def addLight(self, light):
        self.lightList.append(light)
class Object:
    def __init__(self, m):
        self.material = m
class Sphere:
    def __init__(self, c, r, col, spec, kr, kt):
        self.center = c
        self.radius = r

        # CHECKPOINT 3 -- adding object and specular color
        self.color = col
        self.spec_color = spec

        # Phong Parameters
        # be sure that kd + ks < 1
        self.ka = 0.8
        self.kd = 0.1
        self.ks = 0.1
        self.ke = 1

        # CHECKPOINT 5 -- setting reflection and transmission constants
        self.kr = kr
        self.kt = kt
        

    def intersect(self, ray):
        di = ray.direction.normalize()
        dx = di.x
        dy = di.y
        dz = di.z

        ori = ray.origin
        xo = ori.x
        yo = ori.y
        zo = ori.z

        xc = self.center[0]
        yc = self.center[1]
        zc = self.center[2]

        r = self.radius

        a = dx**2 + dy**2 + dz**2 
        b = 2 * ((dx * (xo - xc)) + (dy * (yo - yc)) + (dz * (zo - zc)))
        c = (xo - xc)**2 + (yo - yc)**2 + (zo - zc)**2 - r**2
        det = b**2 - (4 * a * c)
    
        if det > 0:
            w_pos = ((-1 * b) + (math.sqrt(det)))/ (2 * a)
            w_neg = ((-1 * b) - math.sqrt(det)) / (2 * a)
            root = 0
            if w_pos > 0 and w_neg > 0:
                root = min(w_pos, w_neg)
            elif w_pos > 0:
                root = w_pos
            elif w_neg > 0:
                root = w_neg

            elif w_pos == 0 or w_neg == 0:
                root = 0
                
            else:
                root = -1 # no intersection
        else:
            root = -1
            
        if root < 0:
            return None, None, None
            # return None, None, None
            
        else:
            xi = xo + dx * root
            yi = yo + dy * root
            zi = zo + dz * root

            # normalize the intersection point
            xn = xi - xc
            yn = yi - yc
            zn = zi - zc

            in_pt = [xi, yi, zi]
            sur_norm = [xn, yn, zn]

            # return in_pt, root
            return in_pt, sur_norm, root

    def type(self):
        return "sphere"
class Triangle:
    def __init__(self, a: Point, b: Point, c: Point, col, spec, kr, kt):
        self.a = a
        self.b = b
        self.c = c

        # CHECKPOINT 3 -- adding object and specular color
        self.color = col
        self.spec_color = spec

        # Phong Parameters
        # be sure that kd + ks < 1
        self.ka = 0.8
        self.kd = 0.1
        self.ks = 0.1
        self.ke = 1

        # CHECKPOINT 5 -- setting reflection and transmission constants
        self.kr = kr
        self.kt = kt

    def transform(self, t_matrix):
        a = self.a
        b = self.b
        c = self.c

        coords_a = np.array([a.x, a.y, a.z, 1])
        coords_b = np.array([b.x, b.y, b.z, 1])
        coords_c = np.array([c.x, c.y, c.z, 1])


        res_a = np.matmul(coords_a, t_matrix)
        res_b = np.matmul(coords_b, t_matrix)
        res_c = np.matmul(coords_c, t_matrix)


        # translate back to Euclidean space
        wa = res_a[3]
        wb = res_b[3]
        wc = res_c[3]

        self.a = Vector(res_a[0]/wa, res_a[1]/wa, res_a[2]/wa)
        self.b = Vector(res_b[0]/wb, res_b[1]/wb, res_b[2]/wb)
        self.c = Vector(res_c[0]/wc, res_c[1]/wc, res_c[2]/wc)

    def intersect(self, ray:Ray):
        po = ray.origin
        d = ray.direction

        d = d.normalize()
 
        a = self.a
        b = self.b
        c = self.c

        e1 = Vector(b.subtract(a))
        # print("e1: " + str(e1.v))
        e2 = Vector(c.subtract(a))
        # print("e2: " + str(e2.v))
        T = Vector(po.subtract(a))
        # print("T: " + str(T.v))
        P = Vector(d.cross(e2))
        # print("P: " + str(P.v))
        Q = Vector(T.cross(e1))
        # print("Q: " + str(Q.v))


        den = P.dot(e1)
        # print("den: " + str(den))

        if den != 0:
            w_dist = Q.dot(e2) / den
            u = P.dot(T) / den
            v = Q.dot(d) / den

            if w_dist < 0:
                return None, None, None # behind ray

            if u < 0 or u > 1 or v < 0 or v > 1 or u + v < 0 or u + v > 1:
                return None, None, None # outside of triangle
            
            else:
                w = 1 - u - v
                in_pt = [a.x * u + b.x * v + c.x * w, a.y * u + b.y * v + c.y * w, a.z * u + b.z * v + c.z * w]
                sur_norm = e1.cross(e2)
                return in_pt, sur_norm, w_dist

        else:
            return None, None, None
    
    def type(self):
        return "triangle"

class IntersectData:
    def __init__(self, obj, in_pt, norm, li):
        self.obj = obj
        self.pt = in_pt
        self.norm = norm
        self.LightList = li
        self.s = Vector([0,0,0])
        self.n = Vector([0,0,0])
        self.v = Vector([0,0,0])
        self.r = Vector([0,0,0])

        # self.point = pt # S
        # self.norm = norm # N
        # self.incoming = incoming # V
        # self.reflective = reflective # R
class someVectors:
    def __init__ (self, S, N, V, R):
        self.s = S
        self.n = N
        self.v = V
        self.r = R


class IlluminationModel:
    def __init__(self, material):
        self.material = material
    def illuminate(IntersectData):
        pass

class Phong:
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