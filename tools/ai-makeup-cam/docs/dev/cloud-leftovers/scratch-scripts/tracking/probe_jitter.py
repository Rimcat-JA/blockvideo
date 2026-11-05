import sys, time, numpy as np, cv2, mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
M = sys.argv[1]
buf = open(M + '/face_landmarker.task', 'rb').read()
def mk(conf=0.5):
    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_buffer=buf), running_mode=vision.RunningMode.VIDEO, num_faces=1, min_face_detection_confidence=conf))
rng = np.random.default_rng(0)
for name in ("frame_720p.png", "frame_dark.png", "frame_far.png"):
    bgr = cv2.imread(sys.argv[2] + "/" + name); H, W = bgr.shape[:2]
    noisy = [np.clip(bgr + rng.normal(0, 3, bgr.shape), 0, 255).astype(np.uint8) for _ in range(30)]
    for crop in ("full", "center"):
        x0 = (W - H)//2 if crop == "center" else 0
        cw = H if crop == "center" else W
        for long in (cw, 960, 640):
            if long > cw: continue
            fl = mk(0.4); pts = []; times = []
            for i, f in enumerate(noisy):
                src = f[:, x0:x0+cw]
                t = time.perf_counter()
                s = long / max(src.shape[:2])
                img = cv2.resize(src, (round(cw*s), round(H*s)), interpolation=cv2.INTER_AREA) if s < 1 else src
                r = fl.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), 33*(i+1))
                times.append((time.perf_counter()-t)*1000)
                if r.face_landmarks:
                    pts.append([(x0 + q.x*cw, q.y*H) for q in r.face_landmarks[0]])
            fl.close()
            if len(pts) < 5:
                print(name, crop, long, "found", len(pts)); continue
            p = np.array(pts)
            jit = p.std(axis=0).mean()
            iod = np.linalg.norm(p[:, 33] - p[:, 263], axis=1).mean()
            print(f"{name} {crop:6s} long={long:4d} found={len(pts)} jitter_px={jit:.2f} iod={iod:.1f} ms={np.median(times):.1f}")
