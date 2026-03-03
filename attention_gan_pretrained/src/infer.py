import os, yaml, torch, cv2, numpy as np, glob, argparse
from src.models.generator_unet import SimpleUNet
from src.utils_viz import overlay_heatmap, save_vis

def parse():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--ckpt', default=None); ap.add_argument('--image', required=True); return ap.parse_args()

def load_latest_ckpt(dir_path):
    files = sorted(glob.glob(os.path.join(dir_path, 'epoch_*.pt')))
    return files[-1] if files else None

def preprocess_img(path, size):
    bgr = cv2.imread(path)
    if bgr is None: raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype('float32')/255.0
    h,w = rgb.shape[:2]; s = size; scale = s/max(h,w); nh,nw=int(h*scale), int(w*scale)
    img = cv2.resize(rgb, (nw,nh))
    pad_t = (s-nh)//2; pad_b = s-nh-pad_t; pad_l = (s-nw)//2; pad_r = s-nw-pad_l
    img = cv2.copyMakeBorder(img, pad_t,pad_b,pad_l,pad_r, cv2.BORDER_REFLECT_101)
    return (img*255).astype('uint8'), torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()

def main():
    args = parse(); cfg = yaml.safe_load(open(args.config,'r')); device = 'cuda' if torch.cuda.is_available() else 'cpu'; size = cfg['data']['image_size']
    ck = args.ckpt or load_latest_ckpt(cfg['train']['save_dir'])
    if not ck: raise FileNotFoundError('No checkpoint found. Train first.')
    G = SimpleUNet().to(device); ck_d = torch.load(ck, map_location=device); G.load_state_dict(ck_d['G'])
    rgb0, x = preprocess_img(args.image, size); x = x.to(device)
    with torch.no_grad():
        x_rec, att = G(x)
        a = att[-1][0,0].detach().cpu().numpy()
        a = (a - a.min())/(a.max()-a.min()+1e-6)
    vis = overlay_heatmap(rgb0, a, alpha=0.5)
    save_vis(os.path.join(cfg['infer']['save_vis_dir'],'vis.png'), vis)
    print('Saved to', os.path.join(cfg['infer']['save_vis_dir'],'vis.png'))
