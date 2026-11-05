import numpy as np
from clips import *
from makeup_cam import regions
pairs = {93:323,132:361,58:288,172:397,136:365,150:379,149:378,176:400,148:377,234:454,127:356,162:389,21:251,54:284,103:332,67:297,109:338,
137:366,177:401,215:435,138:367,135:364,169:394,170:395,140:369,171:396,147:376,187:411,207:427,123:352,116:345,117:346,118:347,119:348,
100:329,47:277,50:280,205:425,36:266,101:330,227:447,143:372,111:340,35:265,124:353,46:276,156:383,139:368,71:301,68:298,104:333,69:299,108:337,34:264,
193:417,122:351,196:419,3:248,51:281,45:275,188:412,174:399,236:456,134:363,220:440,245:465,128:357,114:343,55:285,107:336,189:413,
37:267,39:269,40:270,185:409,61:291,167:393,165:391,92:322,186:410,216:436,206:426,203:423,192:416,213:433,214:434,212:432,202:422,
70:300,63:293,105:334,66:296,52:282,53:283,65:295,
130:359,25:255,110:339,24:254,23:253,22:252,26:256,112:341,243:463,226:446,31:261,228:448,229:449,230:450,231:451,232:452,233:453,
113:342,225:445,224:444,223:443,222:442,221:441,
}
mid = [10,151,9,8,168,6,197,195,5,4,1,2,164,0,17,18,200,199,175,152]
errs = {k: [] for k in pairs}
best = {k: [] for k in pairs}
for a in ACTORS:
    for c in ("DFA_NEU_XX",):
        L = np.load(TMP / f"lm_{a}_{c}.npy")
        for lm in L[::5]:
            s = regions.face_scale(lm)
            # midline fit: direction via PCA of midline points
            m = lm[mid]; mc = m.mean(0)
            u, sv, vt = np.linalg.svd(m - mc); d = vt[0]; n = np.array([-d[1], d[0]])
            refl = lambda p: p - 2 * np.outer((p - mc) @ n, n)
            R = refl(lm[:468])
            for r, l in pairs.items():
                errs[r].append(np.linalg.norm(R[r] - lm[l]) / s)
                best[r].append(int(np.argmin(np.linalg.norm(lm[:468] - R[r], axis=1))))
bad = []
for r, l in pairs.items():
    e = np.median(errs[r]); vals, cnt = np.unique(best[r], return_counts=True); b = vals[np.argmax(cnt)]
    flag = "" if b == l else f"  <-- nearest reflected {b}"
    if flag or e > 0.06:
        bad.append(r)
    print(f"{r:4d}->{l:4d} err={e:.3f}{flag}")
print("suspicious:", bad)
