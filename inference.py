import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

##################################################
# 1) Model Definition (same as your training code)
##################################################

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
    """
    Same 4-level ResBlock-based U-Net from your training script.
    Single-channel input-> single-channel output
    with shape/time embeddings appended.
    """
    def __init__(self,num_shapes=4,time_emb_dim=16,base_ch=32):
        super().__init__()
        self.shape_emb=nn.Embedding(num_shapes,8)
        self.time_emb=nn.Embedding(1000,time_emb_dim)
        self.cond_fc=nn.Linear(8+time_emb_dim,base_ch)

        # down
        self.down1=DownBlock(in_ch=1+base_ch, out_ch=base_ch)
        self.down2=DownBlock(base_ch, base_ch*2)
        self.down3=DownBlock(base_ch*2, base_ch*4)
        self.down4=DownBlock(base_ch*4, base_ch*8)

        # bottleneck
        self.bot1=ResBlock(base_ch*8, base_ch*8)
        self.bot2=ResBlock(base_ch*8, base_ch*8)

        # up
        self.up1=UpBlock(in_ch=base_ch*8, out_ch=base_ch*4, skip_ch=base_ch*8)
        self.up2=UpBlock(in_ch=base_ch*4, out_ch=base_ch*2, skip_ch=base_ch*4)
        self.up3=UpBlock(in_ch=base_ch*2, out_ch=base_ch,   skip_ch=base_ch*2)
        self.up4=UpBlock(in_ch=base_ch,   out_ch=base_ch,   skip_ch=base_ch)

        # final
        self.final=nn.Conv2d(base_ch,1,kernel_size=3,padding=1)

    def forward(self,x_noisy,t,shape_id):
        B,C,H,W=x_noisy.shape
        s_e=self.shape_emb(shape_id)   # (B,8)
        t_e=self.time_emb(t)          # (B,time_emb_dim)
        cond=torch.cat([s_e,t_e],dim=1)  # => (B,8+time_emb_dim)
        cond=self.cond_fc(cond)          # => (B,base_ch)
        cond_map=cond.unsqueeze(-1).unsqueeze(-1).expand(B, cond.shape[1], H, W)

        x=torch.cat([x_noisy,cond_map],dim=1)  # => (B,1+base_ch,H,W)

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

##################################################
# 2) Diffusion Sampling Code
##################################################

def linear_alpha_schedule(T,start=1e-4,end=0.02):
    return torch.linspace(start,end,T)

@torch.no_grad()
def sample_model(model, alpha_schedule, shape_id, steps=1000, device="cuda", img_size=(1,128,128)):
    model.eval()
    x=torch.randn((1,)+img_size,device=device)  # (1,1,H,W)
    shape_id_batch=torch.tensor([shape_id],device=device)

    for t_cur in reversed(range(steps)):
        t_tensor=torch.tensor([t_cur],device=device)
        noise_pred=model(x,t_tensor,shape_id_batch)
        alpha_t=alpha_schedule[t_cur].view(1,1,1,1).to(device)
        x0_pred=(x - torch.sqrt(1-alpha_t)*noise_pred)/torch.sqrt(alpha_t)

        if t_cur>0:
            beta_t=1-alpha_t
            z=torch.randn_like(x)
            x=torch.sqrt(alpha_t)*x0_pred + torch.sqrt(beta_t)*z
        else:
            x=x0_pred

    x=x.clamp(0,1)
    return x.squeeze(0)  # => shape (1,H,W)

##################################################
# 3) Main Inference Script
##################################################

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="advanced_bw_unet_best.pth",
                        help="Path to the trained checkpoint (pth file)")
    parser.add_argument("--shape", type=str, default="square",
                        help="Which shape to generate: circle, triangle, square, star")
    parser.add_argument("--steps", type=int, default=1000,
                        help="How many reverse diffusion steps")
    parser.add_argument("--out", type=str, default="bw_inferred.png",
                        help="Output filename for the generated image")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (cpu or cuda)")
    parser.add_argument("--image_size", type=int, default=128,
                        help="Width/height of the generated image (model must match)")

    args=parser.parse_args()
    device=args.device

    # 1) Build model, load checkpoint
    model=AdvancedBWUNet(num_shapes=4,time_emb_dim=16,base_ch=32).to(device)
    state_dict=torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded model from {args.model}.")

    # 2) Build alpha schedule
    T=1000
    alpha_sched=linear_alpha_schedule(T,1e-4,0.02).to(device)

    # 3) Convert shape name to shape_id
    shape_dict={"circle":0,"triangle":1,"square":2,"star":3}
    if args.shape not in shape_dict:
        print(f"Unknown shape '{args.shape}'. Choose from {list(shape_dict.keys())}.")
        return
    shape_id=shape_dict[args.shape]

    # 4) Sample
    gen=sample_model(model, alpha_sched, shape_id, steps=args.steps, device=device,
                     img_size=(1,args.image_size,args.image_size))
    # shape => (1,H,W)
    arr=(gen[0].cpu().numpy()*255).astype(np.uint8)

    # 5) Save
    cv2.imwrite(args.out,arr)
    print(f"Saved final output to '{args.out}'.")

if __name__=="__main__":
    main()
