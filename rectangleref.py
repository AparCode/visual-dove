# VisualizerDove - Ray Tracing Rectangle Reference Image
# FINAL PROJECT
# @author: Aparnaa Senthilnathan
# 3.12.7
# ----------------------------------

import numpy as np
import math
from PIL import Image
MAX_DEPTH = 2
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

        # taking the averages of all lights
        x = 0
        y = 0
        z = 0

        for li in world.lightList:
            x += li.pos[0]
            y += li.pos[1]
            z += li.pos[2]
        
        x_avg = x / 4.0
        y_avg = y / 4.0
        z_avg = z / 4.0

        light = Light([x_avg, y_avg, z_avg], Color(255,255,255))
            
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

class CookTorrance:
    def __init__(self, material):
        self.material = material

    def illuminate(id: IntersectData):
        print(id.pt)


# returns the reflection vector
# r = S + 2(s*n/|n|^2)n
# ray = d
def reflection(ray, norm, value):
    if value % 2 == 0:
        comp = ray.dot(norm)
        # print(ray.length())
        coe = 2 * comp
        a = norm.scaling(coe)

        r = Vector(ray.add(a))
    
    # incident ray
    else:
        comp = ray.dot(norm)
        # print(ray.length())
        coe = 2 * comp
        a = norm.scaling(coe)

        r = Vector(ray.subtract(a))

    return r

# --- CHECKPOINT 6 ---
# check if the vector is facing forward
def faceForward(A, B):
    # for acute angles, the dot product must be positive
    if A.dot(B) >= 0:
        return A

    # for obtuse angles, reverse the first vector
    V = Vector(A).scaling(-1)

    return V

# calculating the transmission ray
def transmission(ray, norm):
    i = ray

    # if the ray is inside the shape, use -n as the normal
    # vector
    if norm.dot(ray) < 0:
        n = norm.scaling(-1)
    else:
        n = norm

    # getting the refraction values
    re_air_idx = 1.0
    re_ground_idx = 1.0
    
    # calculating sin^2 using Snail's Law
    ni_nt = re_ground_idx/re_air_idx

    neg_i = i.scaling(-1)
    cos_i = neg_i.dot(n)

    sin_i2 = (ni_nt)**2 * (1-(cos_i**2))

    # calculating the t vector now
    t_n = (ni_nt * cos_i) - math.sqrt((1 - sin_i2))
    t = (i.scaling(ni_nt)).add(n.scaling(t_n))

    return Vector(t)


def color_illuminate(ray, depth, world, background_color):
    # print("START")
    # print(depth)
    in_list = []
    for o in world.objectList:
        in_pt, sur_norm, dist = o.intersect(ray)
        if in_pt != None:
            in_list.append((o, in_pt, sur_norm, dist))
        
    if in_list == []:
        return background_color
    else:
        if len(in_list) > 1:
            closest_dist = 9999
            closest_obj = in_list[0][0]
            closest_in_pt = in_list[0][1]
            closest_norm = in_list[0][2]
            for o in in_list:
                if o[3] > closest_dist:
                    closest_obj = o[0]
                    closest_in_pt = o[1]
                    closest_norm = o[2]

        else:
            closest_obj = in_list[0][0]
            closest_in_pt = in_list[0][1]
            closest_norm = in_list[0][2]

        data = IntersectData(closest_obj, closest_in_pt, closest_norm, world.lightList)


        phong = Phong(closest_obj.ka, closest_obj.kd, closest_obj.ks, closest_obj.ke, closest_obj.color)
        L, batch = phong.illuminate(data, world, depth)
              
        reflect_dir = batch.r
        if depth < MAX_DEPTH:
            kr = data.obj.kr
            kt = data.obj.kt

            if kr > 0:
                in_pt = Vector(data.pt)
                reflectRay = Ray(in_pt,reflect_dir)

                L = L.addnewColor(color_illuminate(reflectRay, depth+1, world, background_color)).multiply(kr)

            if kt > 0:
                in_pt = Vector(data.pt)
                norm = Vector(data.norm)

                transmit_dir = transmission(ray.direction, norm)
                transmissionRay = Ray(in_pt,transmit_dir)


                L = L.addnewColor(color_illuminate(transmissionRay, depth+1, world, background_color)).multiply(kt)

        return L

# ---- main function ----
# y - vertical pos
# x - upper diagonal
# z - forwards

# image_width = 198
# image_height = 108

# # working triangles
tri_a = Vector([4,2,1])
tri_b = Vector([-4,2,1])
tri_c = Vector([-4,-0.5,1])

tri_a2 = Vector([-4,-0.5,1])
tri_b2 = Vector([4,-0.5,1])
tri_c2 = Vector([4,2,1])

flip_tri_a = Vector([4,-0.5,1])
flip_tri_b = Vector([-4,-0.5,1])
flip_tri_c = Vector([-4,-3,1])

flip_tri_a2 = Vector([-4,-3,1])
flip_tri_b2 = Vector([4,-3,1])
flip_tri_c2 = Vector([4,-0.5,1])


# ---- other elements ---
camera_eyept = Vector([0,0,0])
color = Color(217, 73, 252)
flip_color = Color(94, 30, 110)
background_color = Color(0, 0, 0)
spec_color = Color(255,255,255)

light = [12.25,2,1]
light_rad_color = Color(255,255,255)
light2 = [-12.25,2,1]
light3 = [12.25,-2,1]
light4 = [12.25,2,-1]



# ---- CHECKPOINT 2 ----

# final dimensioms
image_width = 1024
image_height = 768

# testing dimensions
# image_width = 128
# image_height = 96

frame = np.zeros((image_height, image_width, 3), dtype=np.uint8)

# adding objects
world = World()
world.add(Triangle(tri_a, tri_b, tri_c, color, spec_color, 0.0, 0.0))
world.add(Triangle(tri_a2, tri_b2, tri_c2, color, spec_color, 0.0, 0.0))
world.add(Triangle(flip_tri_a, flip_tri_b, flip_tri_c, color, spec_color, 0.3, 0.3))
world.add(Triangle(flip_tri_a2, flip_tri_b2, flip_tri_c2, flip_color, spec_color, 0.3, 0.3))
world.addLight(Light(light, light_rad_color))
world.addLight(Light(light2, light_rad_color))
world.addLight(Light(light3, light_rad_color))
world.addLight(Light(light4, light_rad_color))

# create film plane
f = 8
film_plane_width = 64
film_plane_height = 48
pixel_height = film_plane_height/image_height
pixel_width = film_plane_width/image_width

# spawn a ray
for h in range(image_height):
    for w in range(image_width):
        w_val = (-1 * film_plane_width/2) + (w * pixel_width)
        h_val = (film_plane_height/2) - (h * pixel_height)
        
        x = w_val + (0.5 * pixel_width)
        y = h_val - (0.5 * pixel_height)
        z = f

        pixel = camera_eyept
        endpoint = Vector([x, y, z])

        ray = Ray(pixel, endpoint)
        L = color_illuminate(ray, 1, world, background_color)

        frame[h,w] = L.convert_to_bytes()

# convert coordinates to an image
img = Image.fromarray(frame, 'RGB')
img.save('ref.png')

# save the RGB values in a file
with open("rgbref.txt", "w") as f:
    for r in range(len(frame)):
        for c in range(len(frame[r])):
            f.write(str(frame[r][c][0]) + " " + str(frame[r][c][1]) + " " + str(frame[r][c][2]) + " ")
            f.write("\n")
        f.write("\n")