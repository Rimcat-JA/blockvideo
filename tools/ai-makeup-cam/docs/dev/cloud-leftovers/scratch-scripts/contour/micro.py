import time
import cv2, numpy as np
small = np.random.rand(87, 74, 3).astype(np.float32)
w = np.random.rand(87, 74).astype(np.float32)
def T(fn, n=2000):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return f"{np.percentile(ts, 25) * 1e6:.1f}us"
print("np max axis2", T(lambda: small.max(axis=2)))
print("cv2 split+max", T(lambda: (lambda b: cv2.max(cv2.max(b[0], b[1]), b[2]))(cv2.split(small))))
print("np 1ch mul", T(lambda: w * w))
print("cv2 1ch mul", T(lambda: cv2.multiply(w, w)))
print("np 3ch mul", T(lambda: small * small))
print("cv2 3ch mul", T(lambda: cv2.multiply(small, small)))
print("transform", T(lambda: cv2.transform(w, np.ones((3, 1), np.float32))))
print("np scalar sub", T(lambda: np.float32(1) - w))
print("np clip", T(lambda: np.clip(w, 0, 1)))
print("ones3", T(lambda: np.ones((87, 74, 3), np.float32)))
big = np.random.rand(317, 270, 3).astype(np.float32)
g = np.random.rand(317, 270, 3).astype(np.float32)
print("full mul", T(lambda: cv2.multiply(big, g), 300))
print("full add", T(lambda: cv2.add(big, g), 300))
print("full np.minimum", T(lambda: np.minimum(big, np.float32(1), out=g), 300))
print("full resize area", T(lambda: cv2.resize(big, (74, 87), interpolation=cv2.INTER_AREA), 300))
print("full resize lin", T(lambda: cv2.resize(big, (74, 87), interpolation=cv2.INTER_LINEAR), 300))
print("up resize", T(lambda: cv2.resize(small, (270, 317), interpolation=cv2.INTER_LINEAR), 300))
