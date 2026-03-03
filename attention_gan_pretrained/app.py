import streamlit as st
import os, glob, yaml, torch, cv2, numpy as np
from PIL import Image
from src.models.generator_unet import SimpleUNet
from src.utils_viz import overlay_heatmap

st.set_page_config(layout='wide', page_title='ThreatSight Demo')
st.title('ThreatSight — Attention-GAN Demo')

uploaded = st.file_uploader('Upload image (jpg/png)', type=['jpg','jpeg','png'])
if uploaded is None:
    sample = sorted(glob.glob('data/processed/val/anomaly/*'))
    if sample:
        uploaded = open(sample[0],'rb')

if uploaded:
    if hasattr(uploaded, 'read'):
        img = Image.open(uploaded).convert('RGB')
    else:
        img = Image.open(uploaded)
    st.image(img, caption='Input image')
    cfg = yaml.safe_load(open('configs/base.yaml','r'))
    size = cfg['data']['image_size']
    files = sorted(glob.glob('checkpoints/epoch_*.pt'))
    if files:
        ck = files[-1]
    else:
        st.error('No checkpoint found in checkpoints/. Place checkpoint file there.')
        st.stop()
    model = SimpleUNet(); ck_d = torch.load(ck, map_location='cpu'); model.load_state_dict(ck_d['G']); model.eval()
    rgb = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h,w = rgb.shape[:2]; s=size; scale = s/max(h,w); nh,nw=int(h*scale), int(w*scale)
    img_r = cv2.resize(rgb, (nw,nh))
    pad_t=(s-nh)//2; pad_b=s-nh-pad_t; pad_l=(s-nw)//2; pad_r=s-nw-pad_l
    img_p = cv2.copyMakeBorder(img_r, pad_t,pad_b,pad_l,pad_r, cv2.BORDER_REFLECT_101)
    inp = torch.from_numpy(cv2.cvtColor(img_p, cv2.COLOR_BGR2RGB).astype('float32')/255.0).permute(2,0,1).unsqueeze(0).float()
    with torch.no_grad(): out, att = model(inp)
    a = att[-1][0,0].cpu().numpy(); a = (a - a.min())/(a.max()-a.min()+1e-8)
    vis = overlay_heatmap(cv2.cvtColor(img_p, cv2.COLOR_BGR2RGB), a, alpha=0.55)
    st.image(vis, caption='Attention heatmap')
