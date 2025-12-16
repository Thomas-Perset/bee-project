import numpy as np
import cv2
from skimage.measure import moments_hu, regionprops
from skimage.feature import graycomatrix, graycoprops, canny

def masked_stats_rgb(img, mask):
    # img: HxWx3 RGB, mask: HxW {0,1}
    stats = {}
    area = mask.sum()
    H, W = mask.shape
    stats['area_ratio'] = area / float(H*W)

    for i, ch in enumerate('rgb'):
        vals = img[:, :, i][mask == 1].astype(np.float32)
        stats[f'{ch}_min']  = float(vals.min())
        stats[f'{ch}_max']  = float(vals.max())
        stats[f'{ch}_mean'] = float(vals.mean())
        stats[f'{ch}_median'] = float(np.median(vals))
        stats[f'{ch}_std']  = float(vals.std(ddof=0))
    return stats

def shape_features(mask):
    stats = {}
    props = regionprops(mask.astype(np.uint8))[0] if mask.any() else None
    if props:
        stats['eccentricity'] = float(props.eccentricity)
        stats['solidity'] = float(props.solidity)
        stats['extent'] = float(props.extent)

        # circularité 4πA/P²
        contours, _ = cv2.findContours(mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            A = cv2.contourArea(cnt)
            P = cv2.arcLength(cnt, True)
            stats['circularity'] = float(4*np.pi*A/(P*P + 1e-6))
        else:
            stats['circularity'] = 0.0
    else:
        stats.update({'eccentricity':0.0, 'solidity':0.0, 'extent':0.0, 'circularity':0.0})

    # Moments de Hu (via OpenCV)
    m = cv2.moments(mask.astype(np.uint8))
    hu = cv2.HuMoments(m).flatten()                # shape (7,)
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    for k, v in enumerate(hu_log, 1):
        stats[f'hu_{k}'] = float(v)
    return stats


def symmetry_features(mask):
    stats = {}
    flip_h = np.fliplr(mask)
    flip_v = np.flipud(mask)
    # corrélation normalisée
    def ncc(a, b):
        a = a.astype(np.float32); b = b.astype(np.float32)
        a = (a - a.mean()) / (a.std() + 1e-6)
        b = (b - b.mean()) / (b.std() + 1e-6)
        return float((a*b).mean())
    stats['sym_h'] = ncc(mask, flip_h)
    stats['sym_v'] = ncc(mask, flip_v)
    return stats

def texture_glcm(img, mask, distances=(2, 5), angles=(0, np.pi/4, np.pi/2)):
    # Niveaux réduits pour GLCM
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    vals = gray[mask == 1]
    if len(vals) < 50:
        return {'glcm_contrast':0.0, 'glcm_homogeneity':0.0}
    # discretisation 8 niveaux
    roi = np.zeros_like(gray)
    roi[mask==1] = gray[mask==1]
    roi = (roi / 32).astype(np.uint8)
    glcm = graycomatrix(roi, distances=distances, angles=angles, levels=8, symmetric=True, normed=True)
    con = graycoprops(glcm, 'contrast').mean()
    hom = graycoprops(glcm, 'homogeneity').mean()
    return {'glcm_contrast': float(con), 'glcm_homogeneity': float(hom)}

def edge_density(img, mask):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = canny(gray, sigma=1.0).astype(np.uint8)
    if mask.sum() == 0: return {'edge_density': 0.0}
    return {'edge_density': float(edges[mask==1].mean())}
