import time, numpy as np, cv2
def t(label, fn, n=500):
    fn(); s = time.perf_counter()
    for _ in range(n): fn()
    print(f"{label}: {(time.perf_counter()-s)/n*1000:.4f} ms")
g = np.random.rand(136, 116).astype(np.float32); c = np.random.rand(136, 116, 3).astype(np.float32)
for sig in (1.7, 2.4, 3.8, 4.8):
    t(f"gauss1 s{sig}", lambda: cv2.GaussianBlur(g, (0, 0), sig, borderType=cv2.BORDER_REPLICATE))
    t(f"gauss3 s{sig}", lambda: cv2.GaussianBlur(c, (0, 0), sig, borderType=cv2.BORDER_REPLICATE))
    k = int(round(sig * 2.5)) * 2 + 1
    t(f"stack1 k{k}", lambda: cv2.stackBlur(g, (k, k)))
    t(f"stack3 k{k}", lambda: cv2.stackBlur(c, (k, k)))
t("box1 k9", lambda: cv2.boxFilter(g, -1, (9, 9)))
print(cv2.getNumThreads())
cv2.setNumThreads(1)
t("gauss1 s4.8 1thr", lambda: cv2.GaussianBlur(g, (0, 0), 4.8, borderType=cv2.BORDER_REPLICATE))
