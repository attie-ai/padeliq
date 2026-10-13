"""Dependency-free identity gating helpers."""
import math

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area = lambda r: max(0, r[2]-r[0]) * max(0, r[3]-r[1])
    return inter / max(1e-9, area(a)+area(b)-inter)

def plausible(previous, current):
    center = lambda b: ((b[0]+b[2])/2, (b[1]+b[3])/2)
    a, b = center(previous), center(current)
    height = max(1, previous[3]-previous[1])
    ratio = (current[3]-current[1])/height
    return math.dist(a, b) <= height * .6 and .5 <= ratio <= 2
