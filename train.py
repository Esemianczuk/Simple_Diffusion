import os
import cv2
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import math

##############################################
# 1) Data Generation: Single-Channel B/W
##############################################

# We define shape types with consistent size, orientation => no random variation.

SHAPES = ("circle", "triangle", "square", "star")
SHAPE_SIZES = {
    "circle":   35,
    "triangle": 40,
    "square":   35,
    "star":     30
}

def draw_bw_shape_fixed(img, shape_name):
    """
    Draw shape in white (pixel=255) on black (pixel=0), 
    single channel (H,W).
    NO random scale/angle => each shape type is identical.
    """
    H, W = img.shape
    center = (W//2, H//2)
    size = SHAPE_SIZES[shape_name]

    if shape_name=="circle":
        # circle, radius = size
        cv2.circle(img, center, size, 255, thickness=-1)

    elif shape_name=="square":
        half = size//2
        # no rotation
        pts = np.array([
            [center[0]-half, center[1]-half],
            [center[0]+half, center[1]-half],
            [center[0]+half, center[1]+half],
            [center[0]-half, center[1]+half]
        ], dtype=np.int32)
        cv2.fillPoly(img, [pts], 255)

    elif shape_name=="triangle":
        half_base = size/2
        tri_height = int(0.866*size)
        pts = np.array([
            [center[0], center[1] - tri_height//2],  # top
            [center[0] - half_base, center[1] + tri_height//2], # bottom-left
            [center[0] + half_base, center[1] + tri_height//2]  # bottom-right
        ], dtype=np.int32)
        cv2.fillPoly(img, [pts], 255)

    elif shape_name=="star":
        # let's define a 5-point star with radius=size, no rotation
        num_points=5
        outer_r=size
        inner_r=size//2
        pts=[]
        for i in range(num_points*2):
            r=outer_r if i%2==0 else inner_r
            theta=(i*np.pi/num_points)
            x=int(center[0]+r*np.cos(theta))
            y=int(center[1]+r*np.sin(theta))
            pts.append([x,y])
        pts=np.array(pts,dtype=np.int32)
        cv2.fillPoly(img,[pts],255)


def generate_shapes_dataset(
    out_dir="synthetic_shapes_bw",
    num_samples=5000,
    img_size=(128,128),
    shapes=SHAPES
):
    """
    Generates single-channel BW images:
      - 0 for background,
      - 255 for shape,
      - each shape centered.
    NO random variation => each shape type is identical for all samples.
    Saves them to out_dir/images, plus labels.txt
    """
    os.makedirs(out_dir,exist_ok=True)
    images_dir=os.path.join(out_dir,"images")
    os.makedirs(images_dir,exist_ok=True)
    labels_path=os.path.join(out_dir,"labels.txt")

    with open(labels_path,"w") as f:
        for i in range(num_samples):
            img=np.zeros(img_size,dtype=np.uint8)  # (H,W)
            shape_name=random.choice(shapes)
            # draw shape with NO random modifications
            draw_bw_shape_fixed(img, shape_name)

            fn=f"sample_{i:04d}.png"
            fp=os.path.join(images_dir,fn)
            cv2.imwrite(fp,img)
            f.write(f"{fn}\t{shape_name}\n")

    print(f"Generated {num_samples} single-channel images in '{images_dir}'.")
    print(f"Labels saved to '{labels_path}'.")

##############################################
# 2) Dataset
##############################################

SHAPE2ID={"circle":0,"triangle":1,"square":2,"star":3}

class BWShapesDataset(Dataset):
    def __init__(self, root_dir="synthetic_shapes_bw"):
        super().__init__()
        self.root_dir=root_dir
        self.images_dir=os.path.join(root_dir,"images")
        self.samples=[]
        labels_path=os.path.join(root_dir,"labels.txt")
        with open(labels_path,"r") as f:
            for line in f:
                line=line.strip()
                if line:
                    fn,sh=line.split()
                    self.samples.append((fn,sh))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        fn,sh=self.samples[idx]
        p=os.path.join(self.images_dir,fn)
        img=cv2.imread(p,cv2.IMREAD_GRAYSCALE)
        bw=(img>127).astype(np.float32)
        bw_tensor=torch.from_numpy(bw).unsqueeze(0)
        shape_id=SHAPE2ID[sh]
        return bw_tensor, shape_id

##############################################
# 3) U-Net Architecture
##############################################

class ResBlock(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.bn1=nn.BatchNorm2d(in_ch)
        self.conv1=nn.Conv2d(in_ch,out_ch,3,padding=1)
        self.bn2=nn.BatchNorm2d(out_ch)
        self.conv2=nn.Conv2d(out_ch,out_ch,3,padding=1)
        if in_ch!=out_ch:
            self.skip_conv=nn.Conv2d(in_ch,out_ch,1)
        else:
            self.skip_conv=nn.Identity()
    def forward(self,x):
        h=self.bn1(x)
        h=F.relu(h)
        h=self.conv1(h)
        h=self.bn2(h)
        h=F.relu(h)
        h=self.conv2(h)
        return self.skip_conv(x)+h

class DownBlock(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.res1=ResBlock(in_ch,out_ch)
        self.pool=nn.MaxPool2d(2)
    def forward(self,x):
        x=self.res1(x)
        skip=x
        out=self.pool(x)
        return skip,out

class UpBlock(nn.Module):
    def __init__(self,in_ch,out_ch,skip_ch):
        super().__init__()
        self.up=nn.ConvTranspose2d(in_ch,out_ch,2,stride=2)
        self.res=ResBlock(out_ch+skip_ch,out_ch)
    def forward(self,x,skip):
        x=self.up(x)
        x=torch.cat([x,skip],dim=1)
        x=self.res(x)
        return x

class AdvancedBWUNet(nn.Module):
    def __init__(self,num_shapes=4,time_emb_dim=16,base_ch=32):
        super().__init__()
        self.shape_emb=nn.Embedding(num_shapes,8)
        self.time_emb=nn.Embedding(1000,time_emb_dim)
        self.cond_fc=nn.Linear(8+time_emb_dim,base_ch)

        self.down1=DownBlock(in_ch=1+base_ch, out_ch=base_ch)
        self.down2=DownBlock(base_ch, base_ch*2)
        self.down3=DownBlock(base_ch*2, base_ch*4)
        self.down4=DownBlock(base_ch*4, base_ch*8)

        self.bot1=ResBlock(base_ch*8, base_ch*8)
        self.bot2=ResBlock(base_ch*8, base_ch*8)

        self.up1=UpBlock(base_ch*8, base_ch*4, skip_ch=base_ch*8)
        self.up2=UpBlock(base_ch*4, base_ch*2, skip_ch=base_ch*4)
        self.up3=UpBlock(base_ch*2, base_ch,   skip_ch=base_ch*2)
        self.up4=UpBlock(base_ch,   base_ch,   skip_ch=base_ch)

        self.final=nn.Conv2d(base_ch,1,kernel_size=3,padding=1)

    def forward(self,x_noisy,t,shape_id):
        B,C,H,W=x_noisy.shape
        s_e=self.shape_emb(shape_id)
        t_e=self.time_emb(t)
        cond=torch.cat([s_e,t_e],dim=1)
        cond=self.cond_fc(cond)
        cond_map=cond.unsqueeze(-1).unsqueeze(-1).expand(B,cond.shape[1],H,W)

        x=torch.cat([x_noisy,cond_map],dim=1)

        s1,d1=self.down1(x)
        s2,d2=self.down2(d1)
        s3,d3=self.down3(d2)
        s4,d4=self.down4(d3)

        b=self.bot1(d4)
        b=self.bot2(b)

        u1=self.up1(b,s4)
        u2=self.up2(u1,s3)
        u3=self.up3(u2,s2)
        u4=self.up4(u3,s1)

        out=self.final(u4) # => (B,1,H,W)
        return out

##############################################
# 4) Diffusion
##############################################

def linear_alpha_schedule(T,start=1e-4,end=0.02):
    return torch.linspace(start,end,T)

def forward_diffusion_sample(x0,t,alpha_schedule):
    alphas_t=alpha_schedule[t].view(-1,1,1,1).to(x0.device)
    noise=torch.randn_like(x0)
    x_t=torch.sqrt(alphas_t)*x0 + torch.sqrt(1-alphas_t)*noise
    return x_t,noise

##############################################
# 5) Train
##############################################

from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

def train_diffusion(
    dataset_dir="synthetic_shapes_bw",
    num_samples=5000,
    epochs=500,
    batch_size=32,
    lr=3e-4,
    T=1000,
    device="cuda"
):
    # Step A: Generate dataset
    generate_shapes_dataset(
        out_dir=dataset_dir,
        num_samples=num_samples,
        img_size=(128,128),
        shapes=SHAPES
    )

    # Step B: load + split
    ds=BWShapesDataset(dataset_dir)
    n=len(ds)
    val_size=int(0.1*n)
    train_size=n-val_size
    ds_train,ds_val=random_split(ds,[train_size,val_size],generator=torch.Generator().manual_seed(42))
    dl_train=DataLoader(ds_train,batch_size=batch_size,shuffle=True,drop_last=True)
    dl_val=DataLoader(ds_val,batch_size=batch_size,shuffle=False,drop_last=False)

    # Step C: build model
    model=AdvancedBWUNet(num_shapes=4,time_emb_dim=16,base_ch=32).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    scheduler=CosineAnnealingLR(opt,T_max=epochs,eta_min=1e-6)

    alpha_schedule=linear_alpha_schedule(T,1e-4,0.02).to(device)

    best_val_loss=float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss=0
        for x0,shape_id in dl_train:
            x0=x0.to(device)
            shape_id=shape_id.to(device)

            t=torch.randint(0,T,(x0.shape[0],),device=device)
            x_t,noise=forward_diffusion_sample(x0,t,alpha_schedule)
            noise_pred=model(x_t,t,shape_id)
            loss=F.mse_loss(noise_pred,noise)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss+=loss.item()

        scheduler.step()
        train_avg=total_loss/len(dl_train)

        # validation
        model.eval()
        val_loss=0
        with torch.no_grad():
            for x0v, sidv in dl_val:
                x0v=x0v.to(device)
                sidv=sidv.to(device)
                tv=torch.randint(0,T,(x0v.shape[0],),device=device)
                xv,noisv=forward_diffusion_sample(x0v,tv,alpha_schedule)
                predv=model(xv,tv,sidv)
                val_loss+=F.mse_loss(predv,noisv).item()
        val_avg=val_loss/len(dl_val)
        print(f"Epoch [{epoch+1}/{epochs}] LR:{scheduler.get_last_lr()[0]:.6f} "
              f"Train:{train_avg:.4f}  Val:{val_avg:.4f}")

        if val_avg<best_val_loss:
            best_val_loss=val_avg
            torch.save(model.state_dict(),"advanced_bw_unet_best.pth")
            print(f"  [*] Saved best checkpoint (val_loss={val_avg:.4f})")

    torch.save(model.state_dict(),"advanced_bw_unet_final.pth")
    print("Done training. Final model in advanced_bw_unet_final.pth")
    print(f"Best val loss = {best_val_loss:.4f}")
    return model, alpha_schedule

##############################################
# 6) ONNX Export
##############################################

def export_to_onnx(model, fname="advanced_bw_unet.onnx", device="cuda"):
    model.eval()
    class Wrapper(nn.Module):
        def __init__(self,net):
            super().__init__()
            self.net=net
        def forward(self,x):
            # fix shape_id=0, t=0
            shape_id=torch.zeros((x.size(0),),dtype=torch.long,device=x.device)
            t=torch.zeros((x.size(0),),dtype=torch.long,device=x.device)
            return self.net(x,t,shape_id)

    wrap=Wrapper(model).to(device)
    dummy=torch.randn(1,1,128,128,device=device)
    torch.onnx.export(
        wrap,
        dummy,
        fname,
        opset_version=11,
        input_names=["input"],
        output_names=["output"]
    )
    print(f"Exported model to {fname}.")

##############################################
# 7) Sampling
##############################################

@torch.no_grad()
def sample_model(model,alpha_schedule,shape_id,steps=1000,device="cuda",img_size=(1,128,128)):
    model.eval()
    x=torch.randn((1,)+img_size,device=device)
    shape_id_batch=torch.tensor([shape_id],device=device)
    for t_cur in reversed(range(steps)):
        t_tensor=torch.tensor([t_cur],device=device)
        noise_pred=model(x,t_tensor,shape_id_batch)
        alpha_t=alpha_schedule[t_cur].view(1,1,1,1).to(device)
        x0_pred=(x - torch.sqrt(1-alpha_t)*noise_pred)/torch.sqrt(alpha_t)
        if t_cur>0:
            beta_t=1-alpha_t
            z=torch.randn_like(x)
            x=torch.sqrt(alpha_t)*x0_pred+torch.sqrt(beta_t)*z
        else:
            x=x0_pred

    x=x.clamp(0,1)
    return x.squeeze(0)

##############################################
# MAIN
##############################################

def main():
    device="cuda" if torch.cuda.is_available() else "cpu"

    # 1) Train
    model,alpha_sched=train_diffusion(
        dataset_dir="synthetic_shapes_bw",
        num_samples=5000,
        epochs=500,
        batch_size=32,
        lr=3e-4,
        T=1000,
        device=device
    )

    # 2) Export ONNX
    export_to_onnx(model,"advanced_bw_unet.onnx",device)

    # 3) Sample
    shape_dict={"circle":0,"triangle":1,"square":2,"star":3}
    for nm,idx in shape_dict.items():
        gen=sample_model(model,alpha_sched,idx,steps=1000,device=device)
        arr=(gen[0].cpu().numpy()*255).astype(np.uint8)
        outfn=f"bw_generated_{nm}.png"
        cv2.imwrite(outfn,arr)
        print(f"Saved {outfn}")

if __name__=="__main__":
    main()
