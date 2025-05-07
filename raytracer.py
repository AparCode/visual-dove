# FINAL PROJECT: VisualDove - Raytracing for Color Grid File
# @author: Aparnaa Senthilnathan
# 3.12.7
# ----------------------------------

# importing libraries
import numpy as np
import math
from PIL import Image

# defining global variables
MAX_DEPTH = 2


class Point:
    # Initalizes a point in the frame
    def __init__(self, p):
        self.p = p
        self.x = p[0]
        self.y = p[1]
        self.z = p[2]
    
    # transforms the points based on camera
    def transform(self):
        pass

    # takes the sum of two points
    def add(self, p2):
        return np.add(self.p, p2.p)

    # calculates the difference between these two points
    def subtract(self, p2):
        return np.subtract(self.p, p2.p)

    # calculates the distance between the two points
    def dist(self, p2):
        return np.linalg.norm(self.p, p2.p)

class Vector:
    # Initalizes a vector in the frame
    def __init__(self, v):
        self.v = v
        self.x = self.v[0]
        self.y = self.v[1]
        self.z = self.v[2]

    # takes the sum of two vectors
    def add(self, v2):
        return np.add(self.v, v2.v)

    # takes the difference between two vectors
    def subtract(self, v2):
        return np.subtract(self.v, v2.v)

    # takes the cross product of two vectors
    def cross(self, v2):
        return np.cross(self.v, v2.v)
    
    # takes the dot product of two vectors
    def dot(self, v2):
        return np.dot(self.v, v2.v)
    
    # takes the magnitude of the vector
    def length(self):
        return math.sqrt((self.x ** 2) + (self.y ** 2) + (self.z ** 2))

    # normalizes the vector
    def normalize(self):
        return Vector(self.v / np.linalg.norm(self.v))

    # multiplies a vector by a scalar
    def scaling(self, n):
        return Vector([self.x * n, self.y * n, self.z * n])

    # raises a vector to a given power
    def power(self, n):
        return Vector([self.x ** n, self.y ** n, self.z ** n])
    
    # creates a vector in the reverse direction
    def reverse(self):
        return self.scaling(-1)

    # transforms a vector into world space and back
    def transform(self):
        pass

class Ray:
    # initalizes a ray with origin and direction
    def __init__(self, o, d):
        self.origin = o
        self.direction = d
    
    # calculate the end point of the ray given the parameter w
    def endpt(self, w):
        return self.origin + w*self.direction

class Color:
    # initalizes the color (in bytes) by converting its RGB values to RGB(0-1) values
    def __init__(self, r, g, b):
        self.r = r / 255
        self.g = g / 255
        self.b = b / 255

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
    # initalize the world space with list of objects and lights
    def __init__(self):
        self.objectList = []
        self.attributes = []
        self.lightList = []
        self.luminance = 1

    # add an object to the world
    def add(self, obj):
        self.objectList.append(obj)

    # add a light source to the world
    def addLight(self, light):
        self.lightList.append(light)

class Object:
    # initalize an object with the given material
    def __init__(self, m):
        self.material = m

class Triangle:
    # create a triangle of three points
    def __init__(self, a: Point, b: Point, c: Point, col, spec, kr, kt):
        self.a = a
        self.b = b
        self.c = c

        # adding object and specular colors
        self.color = col
        self.spec_color = spec

        # adding Phong Parameters
        # kd + ks < 1
        self.ka = 0.8
        self.kd = 0.1
        self.ks = 0.1
        self.ke = 1

        # setting reflection and transmission constants
        self.kr = kr
        self.kt = kt

    # check to see if the ray hits the triangle object
    # if so, retrieve the intersection point, normalized intersection point, and root
    def intersect(self, ray:Ray):
        po = ray.origin
        d = ray.direction

        d = d.normalize()
 
        a = self.a
        b = self.b
        c = self.c

        e1 = Vector(b.subtract(a))
        e2 = Vector(c.subtract(a))
        T = Vector(po.subtract(a))
        P = Vector(d.cross(e2))
        Q = Vector(T.cross(e1))

        den = P.dot(e1)

        if den != 0:
            w_dist = Q.dot(e2) / den
            u = P.dot(T) / den
            v = Q.dot(d) / den

            if w_dist < 0: # behind ray
                return None, None, None 

            if u < 0 or u > 1 or v < 0 or v > 1 or u + v < 0 or u + v > 1: # outside of triangle
                return None, None, None 
            
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
    # initalize an interface of intersection data if a ray hits an object
    def __init__(self, obj, in_pt, norm, li):
        self.obj = obj
        self.pt = in_pt
        self.norm = norm
        self.LightList = li
        self.s = Vector([0,0,0]) # shadowRay
        self.n = Vector([0,0,0]) # normal vector
        self.v = Vector([0,0,0]) # incoming vector
        self.r = Vector([0,0,0]) # reflection vector

class someVectors:
    # initalizes a cluster of vectors useful for illumination
    def __init__ (self, S, N, V, R):
        self.s = S
        self.n = N
        self.v = V
        self.r = R

# parent class of illumination models
class IlluminationModel:
    # initalizes a type of illumination model
    def __init__(self, material):
        self.material = material
    # illuminates the image with the model
    def illuminate(IntersectData):
        pass

# Phong Model
class Phong:
    # Initalize a Phong model with k values and object color
    def __init__(self, ka, kd, ks, ke, co):
        self.ka = ka
        self.kd = kd
        self.ks = ks
        self.ke = ke
        self.co = co

    # Perform illumination with the Phong model
    def illuminate(self, id: IntersectData, world, depth):
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


# returns the reflection vector
# r = S + 2(s*n/|n|^2)n
# ray = d
def reflection(ray, norm, value):
    if value % 2 == 0:
        comp = ray.dot(norm)
        coe = 2 * comp
        a = norm.scaling(coe)

        r = Vector(ray.add(a))
    
    # incident ray
    else:
        comp = ray.dot(norm)
        coe = 2 * comp
        a = norm.scaling(coe)

        r = Vector(ray.subtract(a))

    return r

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


# the function for performing local and global illumination function on 
# colors
def color_illuminate(ray, depth, world, background_color):
    in_list = []
    for o in world.objectList:
        in_pt, sur_norm, dist = o.intersect(ray)
        if in_pt != None:
            in_list.append((o, in_pt, sur_norm, dist))

    # if there are no intersections, return background color
    if in_list == []:
        return background_color
    
    # find the closest point of intersection if the ray
    # intersects with more than one object
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

        # save intersection data into a frame
        data = IntersectData(closest_obj, closest_in_pt, closest_norm, world.lightList)

        # use the regular Phong model to illuminate
        phong = Phong(closest_obj.ka, closest_obj.kd, closest_obj.ks, closest_obj.ke, closest_obj.color)
        L, batch = phong.illuminate(data, world, depth)
            
        # prepare the program for recursive ray tracing 
        reflect_dir = batch.r
        if depth < MAX_DEPTH:
            kr = data.obj.kr
            kt = data.obj.kt

            # Perform recursive tracing with reflection by spawning reflection rays per depth increment
            if kr > 0:
                in_pt = Vector(data.pt)
                reflectRay = Ray(in_pt,reflect_dir)

                L = L.addnewColor(color_illuminate(reflectRay, depth+1, world, background_color)).multiply(kr)

            # Perform recursive tracing with transmission by spawning transmission rays per depth increment
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


# main function

# dimensions
image_width = 128
image_height = 96

# create frame
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

        # perform ray tracing
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