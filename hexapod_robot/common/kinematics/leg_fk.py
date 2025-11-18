import math

def forward(joints, links=(0.05, 0.1, 0.12)):
    q1, q2, q3 = joints
    l1, l2, l3 = links
    x1 = l1 * math.cos(q1)
    y1 = l1 * math.sin(q1)
    a2 = q1 + q2
    x2 = x1 + l2 * math.cos(a2)
    y2 = y1 + l2 * math.sin(a2)
    a3 = a2 + q3
    x3 = x2 + l3 * math.cos(a3)
    y3 = y2 + l3 * math.sin(a3)
    return x3, y3