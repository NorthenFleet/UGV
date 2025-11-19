import math

def solve(target, links=(0.05, 0.1, 0.12)):
    x, y = target
    l1, l2, l3 = links
    r = math.hypot(x, y)
    if r < 1e-6:
        return 0.0, 0.0, 0.0
    q1 = math.atan2(y, x)
    xr = r - l1
    c = (xr * xr - l2 * l2 - l3 * l3) / (2 * l2 * l3)
    c = max(-1.0, min(1.0, c))
    q3 = math.acos(c)
    k1 = l2 + l3 * math.cos(q3)
    k2 = l3 * math.sin(q3)
    q2 = math.atan2(y - l1 * math.sin(q1), x - l1 * math.cos(q1)) - math.atan2(k2, k1)
    return q1, q2, q3