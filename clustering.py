from random import randint
from math import sin, cos, radians
from numpy import linspace
from numpy.random import normal
from math import sqrt, exp
import numpy as np


def get_cluster(num_of_points, d, X, Y):
    points = []
    norm = normal(0, d, num_of_points)
    for i, k in enumerate(linspace(0, 360, num_of_points)):
        x = X + norm[i] * cos(radians(k))
        y = Y + norm[i] * sin(radians(k))
        points.append((x, y))

    return np.array(points)


def euclid_distance(point_a, point_b):
    x0, y0 = point_a
    x1, y1 = point_b
    return sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)


def schematic_distance(point_a, point_b):
    return abs(point_a[0] - point_b[0]) + abs(point_a[1] - point_b[1])


def calc_centroids(alpha, div_matrix, points, old):
    centroids = []
    for i in range(3):
        sum_ax = 0
        sum_ay = 0
        sum_bx = 0
        sum_by = 0
        for j, (x, y) in enumerate(points):
            sum_ax += pow(div_matrix[i][j], alpha) * x
            sum_ay += pow(div_matrix[i][j], alpha) * y
            sum_bx += pow(div_matrix[i][j], alpha)
            sum_by += pow(div_matrix[i][j], alpha)
        ca = sum_ax / sum_bx
        cb = sum_ay / sum_by
        centroids.append((ca, cb))
    if old:
        for j, o in enumerate(old):
            if euclid_distance(o, centroids[j]) < 1:
                return None
    return centroids


def calc_div_matrix(alpha, centroids, points, dist_fn):
    div_matrix = []
    for i, centr in enumerate(centroids):
        ps = []
        for j, p in enumerate(points):
            s = 0
            for c in centroids:
                s += 1 / (dist_fn(c, p) ** 2)
            res = ((dist_fn(p, centr) ** 2) * s) ** (1 / (alpha - 1))
            ps.append(res)
        div_matrix.append(ps)
    return div_matrix


def sort_by_potential(points):
    return sorted(points, key=lambda x: x[1], reverse=True)






if __name__ == '__main__':
    pass