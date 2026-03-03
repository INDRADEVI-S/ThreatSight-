import numpy as np, cv2, os
def overlay_heatmap(rgb, heat, alpha=0.5):
    heat_uint = (np.clip(heat,0,1)*255).astype('uint8')
    heat_color = cv2.applyColorMap(heat_uint, cv2.COLORMAP_JET)[:,:,::-1]
    out = (alpha*rgb + (1-alpha)*heat_color).astype('uint8')
    return out
def save_vis(path, rgb):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
