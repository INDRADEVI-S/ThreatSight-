import os, yaml, random, numpy as np, torch, argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from src.models.generator_unet import SimpleUNet
from src.models.discriminator import PatchDiscriminator
from src.losses import hinge_d_loss, hinge_g_loss

class FolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder, size=128):
        import glob
        self.paths = []
        for ext in ('*.jpg','*.png','*.jpeg'): self.paths += glob.glob(os.path.join(folder, ext))
        self.paths.sort()
        self.size = size
        self.tf = transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor()])
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert('RGB')
        return self.tf(img)

def parse_args():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); return ap.parse_args()

def main():
    args = parse_args(); cfg = yaml.safe_load(open(args.config,'r'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(cfg.get('seed',42)); random.seed(cfg.get('seed',42)); np.random.seed(cfg.get('seed',42))
    ds = FolderDataset(cfg['data']['train_dir'], size=cfg['data']['image_size'])
    dl = DataLoader(ds, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=cfg['train'].get('num_workers',0), drop_last=True)
    G = SimpleUNet().to(device); D = PatchDiscriminator().to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg['train']['lr_g'], betas=(0.5,0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg['train']['lr_d'], betas=(0.5,0.999))
    save_dir = cfg['train']['save_dir']; os.makedirs(save_dir, exist_ok=True)
    epochs = cfg['train']['epochs_warmup'] + cfg['train']['epochs_adv']
    for epoch in range(epochs):
        G.train(); D.train()
        pbar = tqdm(dl, desc=f'Epoch {epoch+1}/{epochs}')
        for x in pbar:
            x = x.to(device)
            with torch.no_grad():
                x_rec, _ = G(x)
            real_logits = D(x); fake_logits = D(x_rec.detach())
            d_loss = hinge_d_loss(real_logits, fake_logits)
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()
            x_rec, att = G(x); fake_logits2 = D(x_rec)
            g_adv = hinge_g_loss(fake_logits2)
            g_rec = torch.nn.functional.l1_loss(x_rec, x) * cfg['train'].get('lambda_rec',50.0)
            g_att = sum([a.mean() for a in att]) * cfg['train'].get('lambda_att',0.1)
            g_loss = g_adv + g_rec + g_att
            opt_g.zero_grad(); g_loss.backward(); opt_g.step()
            pbar.set_postfix(d=float(d_loss.item()), g=float(g_loss.item()))
        torch.save({'G':G.state_dict(), 'D':D.state_dict(), 'epoch':epoch+1}, os.path.join(save_dir, f'epoch_{epoch+1}.pt'))
    print('Training finished. checkpoints in', save_dir)
