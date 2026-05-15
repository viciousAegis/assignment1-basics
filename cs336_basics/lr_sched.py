import math


def lr_cosine_schedule(t, alpha_max, alpha_min, T_w, T_c):
    if t < T_w:
        return (t / T_w) * alpha_max
    if t >= T_w and t <= T_c:
        return alpha_min + (
            0.5
            * (1 + math.cos((t - T_w) * math.pi / (T_c - T_w)))
            * (alpha_max - alpha_min)
        )
    if t > T_c:
        return alpha_min
