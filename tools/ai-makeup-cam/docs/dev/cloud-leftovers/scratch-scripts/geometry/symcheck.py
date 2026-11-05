import numpy as np
lm = np.load("tmp_geometry/frame_720p.npy")
pairs = {
 "eye_upper": ([133, 173, 157, 158, 159, 160, 161, 246, 33], [362, 398, 384, 385, 386, 387, 388, 466, 263]),
 "eye_lower": ([133, 155, 154, 153, 145, 144, 163, 7, 33], [362, 382, 381, 380, 374, 373, 390, 249, 263]),
 "iris": ([468, 469, 470, 471, 472], [473, 474, 475, 476, 477]),
 "crease": ([190, 56, 28, 27, 29, 30, 247], [414, 286, 258, 257, 259, 260, 467]),
 "ue_top": ([243, 112, 26, 22, 23, 24, 110, 25, 130], [463, 341, 256, 252, 253, 254, 339, 255, 359]),
 "ue_bot": ([245, 128, 121, 120, 119, 118, 117, 111, 143], [465, 357, 350, 349, 348, 347, 346, 340, 372]),
 "brow_up": ([107, 66, 105, 63, 70], [336, 296, 334, 293, 300]),
 "brow_lo": ([55, 65, 52, 53, 46], [285, 295, 282, 283, 276]),
 "cheek": ([50, 101, 36, 205], [280, 330, 266, 425]),
 "wing": ([49, 129, 64, 98, 240, 75, 218, 115, 131], [279, 358, 294, 327, 460, 305, 438, 344, 360]),
 "corners": ([61, 78], [291, 308]),
}
# midline: fit x = a*y + b through center points
mid = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 2, 164, 0, 13, 14, 17, 18, 200, 199, 175, 152]
y = lm[mid, 1]; x = lm[mid, 0]
a, b = np.polyfit(y, x, 1)
ipd = np.linalg.norm(lm[[33,133]].mean(0) - lm[[362,263]].mean(0))
print("ipd", ipd, "midline", a, b)
def reflect(p):
    # reflect across line x = a*y + b
    n = np.array([1.0, -a]) / np.hypot(1, a)
    d = (p[:, 0] - a * p[:, 1] - b) / np.hypot(1, a)
    return p - 2 * d[:, None] * n
for k, (r, l) in pairs.items():
    err = np.linalg.norm(reflect(lm[l]) - lm[r], axis=1) / ipd
    print(f"{k:10s} max {err.max():.3f} mean {err.mean():.3f}", np.round(err, 2))
# side check
print("468 x", lm[468, 0], "473 x", lm[473, 0], "(subject left should be larger x)")
for v in (2, 1, 4, 168, 0, 13, 14, 17):
    print(v, lm[v])
em = (lm[[33,133]].mean(0) + lm[[362,263]].mean(0)) / 2
print("eye_mid->subnasale(2)/ipd", np.linalg.norm(lm[2] - em) / ipd, "eye_mid->0/ipd", np.linalg.norm(lm[0] - em) / ipd)
