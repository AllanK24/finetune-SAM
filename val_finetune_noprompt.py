from models.sam import sam_model_registry
from models.sam_LoRa import LoRA_Sam
#Scientific computing 
import numpy as np
import os
#Pytorch packages
import torch
import torchvision
#Others
from torch.utils.data import DataLoader
from utils.dataset import Public_dataset
from pathlib import Path
from tqdm import tqdm
from utils.dsc import dice_coeff
import cfg
from monai.metrics.surface_dice import SurfaceDiceMetric
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode

def main(args,test_img_list):
    # change to 'combine_all' if you want to combine all targets into 1 cls
    test_dataset = Public_dataset(args,args.img_folder, args.mask_folder, test_img_list,phase='val',targets=[args.targets],if_prompt=False)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    if args.finetune_type == 'adapter' or args.finetune_type == 'vanilla':
        sam_fine_tune = sam_model_registry[args.arch](args,checkpoint=os.path.join(args.dir_checkpoint,'checkpoint_best.pth'),num_classes=args.num_cls)
    elif args.finetune_type == 'lora':
        sam = sam_model_registry[args.arch](args,checkpoint=os.path.join(args.sam_ckpt),num_classes=args.num_cls)
        sam_fine_tune = LoRA_Sam(args,sam,r=4).to('cuda').sam
        sam_fine_tune.load_state_dict(torch.load(args.dir_checkpoint + '/checkpoint_best.pth'), strict = False)
        
    sam_fine_tune = sam_fine_tune.to('cuda').eval()
    class_iou = torch.zeros(args.num_cls,dtype=torch.float)
    cls_dsc = torch.zeros(args.num_cls,dtype=torch.float)
    
    union_dsc_sum = 0.0
    nsd_union_sum = 0.0
    nsd_count     = 0
    eps = 1e-9

    # --- NSD tolerance in *pixels* on the 1024×1024 grid ---
    tau = args.tau  # try 2.0–4.0; 3.0 usually lands near the paper’s NSD
    sd_union = SurfaceDiceMetric(
        class_thresholds=[tau],     # single foreground channel
        include_background=False,   # ignore background
        reduction="none"            # we'll handle averaging + NaNs manually
    )
    
    img_name_list = []
    pred_msk = []
    test_img = []
    test_gt = []

    for i,data in enumerate(tqdm(testloader)):
        imgs = data['image'].to('cuda')
        
        msks = torchvision.transforms.Resize(
            (args.out_size, args.out_size),
            interpolation=InterpolationMode.NEAREST
        )(data['mask'].float())          # ensure numeric for Resize
        msks = (msks > 0.5).long().to('cuda')   # (B,1,H,W) in {0,1}
        img_name_list.append(data['img_name'][0])

        with torch.no_grad():
            img_emb= sam_fine_tune.image_encoder(imgs)

            sparse_emb, dense_emb = sam_fine_tune.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )
            pred_logits, _ = sam_fine_tune.mask_decoder(
                            image_embeddings=img_emb,
                            image_pe=sam_fine_tune.prompt_encoder.get_dense_pe(), 
                            sparse_prompt_embeddings=sparse_emb,
                            dense_prompt_embeddings=dense_emb, 
                            multimask_output=False,
                          )
            
        # --- Resize GT to pred size ---
        H, W = pred_logits.shape[-2:]
        msks = torchvision.transforms.Resize(
            (H, W), interpolation=InterpolationMode.NEAREST
        )(data['mask'].float())
        msks = (msks > 0.5).long().to('cuda')     # (B,1,H,W) ∈ {0,1}
        tgt_idx = msks.squeeze(1)                 # (B,H,W)

        # --- Turn logits into a class map (B,H,W) in {0,1} ---
        C = pred_logits.shape[1]
        if C == 2:
            probs     = torch.softmax(pred_logits, dim=1)  # (B,2,H,W)
            pred_cls  = probs.argmax(dim=1)                # (B,H,W)
        elif C == 1:
            prob_fg   = torch.sigmoid(pred_logits[:, 0])   # (B,H,W)
            pred_cls  = (prob_fg > 0.5).long()
        else:
            raise RuntimeError(f"Unexpected #channels: {C} (expected 1 or 2)")

        # --- Sanity checks ---
        assert tgt_idx.shape[-2:] == pred_cls.shape[-2:], "GT/pred size mismatch"
        uvals = tgt_idx.unique()
        assert set(uvals.tolist()).issubset({0,1}), f"Bad target values: {uvals}"

        # -------------------------
        # Per-class IoU (macro)
        # -------------------------
        yhat = pred_cls.flatten()
        y    = tgt_idx.flatten()
        for j in range(2):  # 0=bg, 1=fg
            y_bi, yhat_bi = (y == j), (yhat == j)
            I = (y_bi & yhat_bi).sum().item()
            U = torch.logical_or(y_bi, yhat_bi).sum().item()
            class_iou[j] += I / (U + 1e-9)

        # -------------------------
        # Per-class DSC (macro)
        # -------------------------
        for cls in range(2):
            mask_pred_cls = (pred_cls == cls).float()
            mask_gt_cls   = (tgt_idx  == cls).float()
            cls_dsc[cls] += dice_coeff(mask_pred_cls, mask_gt_cls).item()

        # --------------------------------------------
        # UNION foreground DSC + NSD
        # --------------------------------------------
        pred_union = (pred_cls > 0).float()
        gt_union   = (tgt_idx  > 0).float()

        union_dsc_sum += dice_coeff(pred_union, gt_union).item()

        pred_oh = F.one_hot(pred_union.long(), num_classes=2).movedim(-1, 1).float()
        gt_oh   = F.one_hot(gt_union.long(),   num_classes=2).movedim(-1, 1).float()

        sd_union(pred_oh, gt_oh)
        nsd_batch = sd_union.aggregate(); sd_union.reset()
        nsd_val = torch.nanmean(nsd_batch).item()
        if np.isfinite(nsd_val):
            nsd_union_sum += nsd_val
            nsd_count += 1

            
        if i == 0:
            print("logits shape:", pred_logits.shape)            # expect [B,2,H,W]
            print("mask uniques:", torch.unique(tgt_idx).tolist())  # expect [0, 1]
            # empty_rate = (pred_fine.sum(dim=(1,2)) == 0).float().mean().item()
            # print("empty FG pred rate:", empty_rate)

    # Averages
    num_batches = i + 1
    class_iou /= num_batches
    cls_dsc   /= num_batches

    union_dsc = union_dsc_sum / float(max(num_batches, 1))
    union_nsd = nsd_union_sum / float(max(nsd_count, 1))

    save_folder = os.path.join('test_results', args.dir_checkpoint)
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    print(dataset_name)
    print('class dsc:', cls_dsc)
    print('class iou:', class_iou)
    print(f'union dsc (foreground>0): {union_dsc:.4f}')
    print(f'union nsd @ tau={tau:.1f}px (foreground only): {union_nsd:.4f}')

    
if __name__ == "__main__":
    args = cfg.parse_args()

    #### COMPLETE BEFORE NEXT TRAINING RUN (LATER)
    # if 1: # if you want to load args from taining setting or you want to identify your own setting
    #     args_path = f"{args.dir_checkpoint}/args.json"

    #     # Reading the args from the json file
    #     with open(args_path, 'r') as f:
    #         args_dict = json.load(f)
        
    #     # Converting dictionary to Namespace
    #     args = Namespace(**args_dict)
        
    dataset_name = args.dataset_name
    
    # test_img_list =  args.img_folder + '/train_slices_info_sampled_1000.txt'
    main(args,args.test_img_list)