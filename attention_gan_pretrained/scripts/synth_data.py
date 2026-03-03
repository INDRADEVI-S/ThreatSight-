import os, cv2, numpy as np, argparse
def make_blob(h=128,w=128):
    img = np.zeros((h,w,3), np.uint8) + 20
    for _ in range(6):
        c = (int(np.random.randint(50,220)), int(np.random.randint(50,220)), int(np.random.randint(50,220)))
        r = np.random.randint(6,28)
        x = np.random.randint(r, w-r); y = np.random.randint(r, h-r)
        cv2.circle(img, (x,y), r, c, -1)
    return img

def main(root='data/processed', n_train=200, n_val=40):
    train = os.path.join(root, 'train', 'normal'); valn = os.path.join(root, 'val', 'normal'); vala = os.path.join(root, 'val', 'anomaly')
    os.makedirs(train, exist_ok=True); os.makedirs(valn, exist_ok=True); os.makedirs(vala, exist_ok=True)
    for i in range(n_train):
        cv2.imwrite(os.path.join(train, f'n_{i:03d}.jpg'), make_blob())
    for i in range(n_val):
        img = make_blob()
        cv2.imwrite(os.path.join(valn, f'n_{i:03d}.jpg'), img)
        a = img.copy(); h,w = a.shape[:2]; x1 = np.random.randint(0, w-40); y1 = np.random.randint(0, h-40)
        a[y1:y1+40, x1:x1+40] = 255
        cv2.imwrite(os.path.join(vala, f'a_{i:03d}.jpg'), a)
    print('Created synthetic dataset at', root)

if __name__=='__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--root', default='data/processed'); ap.add_argument('--n_train', type=int, default=200); ap.add_argument('--n_val', type=int, default=40)
    args = ap.parse_args(); main(args.root, args.n_train, args.n_val)
