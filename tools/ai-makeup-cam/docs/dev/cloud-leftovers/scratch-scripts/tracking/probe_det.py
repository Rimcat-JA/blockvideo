import sys, glob, numpy as np, cv2, mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
M = sys.argv[1]
buf = open(M + '/face_landmarker.task', 'rb').read()
def mk(mode):
    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_buffer=buf), running_mode=mode, num_faces=1))
for path in sorted(glob.glob(sys.argv[2] + '/*.png')):
    bgr = cv2.imread(path); h, w = bgr.shape[:2]
    out = []
    for long in (max(h,w), 640, 480, 320):
        s = min(1, long / max(h, w))
        img = cv2.resize(bgr, (round(w*s), round(h*s)), interpolation=cv2.INTER_AREA) if s < 1 else bgr
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fl = mk(vision.RunningMode.VIDEO)
        found = []
        for i in range(3):
            r = fl.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), 33*(i+1))
            found.append(len(r.face_landmarks))
        if r.face_landmarks:
            p = np.array([(q.x*w, q.y*h) for q in r.face_landmarks[0]])
            iod = np.linalg.norm(p[33]-p[263])
            out.append(f"{long}:{found} iod={iod:.0f} nose=({p[1][0]:.0f},{p[1][1]:.0f})")
        else:
            out.append(f"{long}:{found}")
        fl.close()
    print(path.split('/')[-1], (w, h), ' | '.join(out))
