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
from utils.count_params import count_parameters
from quanta import QuanTAConfig, QuanTAModel

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
    elif args.finetune_type == 'quanta':
        sam = sam_model_registry[args.arch](args,checkpoint=os.path.join(args.sam_ckpt),num_classes=args.num_cls)
        quanta_config = QuanTAConfig(
            d=3,
            per_dim_features=[12,8,8],
            target_modules=[
                "image_encoder.blocks.0.attn.qkv",
                "image_encoder.blocks.1.attn.qkv",
                "image_encoder.blocks.2.attn.qkv",
                "image_encoder.blocks.3.attn.qkv",
                "image_encoder.blocks.4.attn.qkv",
                "image_encoder.blocks.5.attn.qkv",
                "image_encoder.blocks.6.attn.qkv",
                "image_encoder.blocks.7.attn.qkv",
                "image_encoder.blocks.8.attn.qkv",
                "image_encoder.blocks.9.attn.qkv",
                "image_encoder.blocks.10.attn.qkv",
                "image_encoder.blocks.11.attn.qkv",
                # Mask Decoder
                "mask_decoder.transformer.layers.0.self_attn.q_proj",
                # "mask_decoder.transformer.layers.0.self_attn.k_proj",
                "mask_decoder.transformer.layers.0.self_attn.v_proj",
                'mask_decoder.transformer.layers.0.cross_attn_token_to_image.q_proj',
                # 'mask_decoder.transformer.layers.0.cross_attn_token_to_image.k_proj',
                'mask_decoder.transformer.layers.0.cross_attn_token_to_image.v_proj',
                'mask_decoder.transformer.layers.0.cross_attn_image_to_token.q_proj',
                # 'mask_decoder.transformer.layers.0.cross_attn_image_to_token.k_proj',
                'mask_decoder.transformer.layers.0.cross_attn_image_to_token.v_proj',
                'mask_decoder.transformer.layers.1.self_attn.q_proj',
                # 'mask_decoder.transformer.layers.1.self_attn.k_proj',
                'mask_decoder.transformer.layers.1.self_attn.v_proj',
                'mask_decoder.transformer.layers.1.cross_attn_token_to_image.q_proj',
                # 'mask_decoder.transformer.layers.1.cross_attn_token_to_image.k_proj',
                'mask_decoder.transformer.layers.1.cross_attn_token_to_image.v_proj',
                'mask_decoder.transformer.layers.1.cross_attn_image_to_token.q_proj',
                # 'mask_decoder.transformer.layers.1.cross_attn_image_to_token.k_proj',
                'mask_decoder.transformer.layers.1.cross_attn_image_to_token.v_proj',
                'mask_decoder.transformer.final_attn_token_to_image.q_proj',
                # 'mask_decoder.transformer.final_attn_token_to_image.k_proj',
                'mask_decoder.transformer.final_attn_token_to_image.v_proj'
            ],
            quanta_dropout=0.0,
            bias="none",
            merge_weights=True
        )
        sam_fine_tune = QuanTAModel(quanta_config,sam)
        sam_fine_tune.load_state_dict(torch.load(args.dir_checkpoint + '/checkpoint_best.pth'), strict = False)
        # Turn off the gradients
        for param in sam_fine_tune.parameters():
            param.requires_grad = False
        
        total, trainable = count_parameters(sam_fine_tune)
        
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Frozen parameters: {total - trainable:,}")
        
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
        
        msks = torchvision.transforms.Resize((args.out_size,args.out_size), interpolation=InterpolationMode.NEAREST)(data['mask'])
        msks = msks.to('cuda')
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
                            multimask_output=True,
                          )
        
        # Predicted class map [B,H,W]
        pred_fine = pred_logits.argmax(dim=1)

        pred_msk.append(pred_fine.cpu())
        test_img.append(imgs.cpu())
        test_gt.append(msks.cpu())

        # -------------------------
        # Per-class IoU (as before)
        # -------------------------
        yhat = pred_fine.cpu().long().flatten()
        # if msks has shape [B,1,H,W], squeeze channel before flatten
        y_src = msks.cpu()
        if y_src.ndim == 4 and y_src.size(1) == 1:
            y_src = y_src.squeeze(1)
        y = y_src.flatten()

        for j in range(args.num_cls):
            y_bi    = (y == j)
            yhat_bi = (yhat == j)
            I = ((y_bi & yhat_bi).sum()).item()
            U = (torch.logical_or(y_bi, yhat_bi).sum()).item()
            class_iou[j] += I / (U + eps)

        # -------------------------
        # Per-class DSC (as before)
        # -------------------------
        msrc = msks.cpu()
        if msrc.ndim == 4 and msrc.size(1) == 1:
            msrc = msrc.squeeze(1)  # [B,H,W]

        for cls in range(args.num_cls):
            mask_pred_cls_torch = (pred_fine.cpu() == cls)        # [B,H,W]
            mask_gt_cls_torch   = (msrc == cls)                   # [B,H,W]
            cls_dsc[cls] += dice_coeff(
                mask_pred_cls_torch.float(),
                mask_gt_cls_torch.float()
            ).item()

        # --------------------------------------------
        # UNION foreground DSC + NSD (to match paper)
        # --------------------------------------------
        pred_union = (pred_fine > 0).cpu()       # [B,H,W]
        gt_union   = (msrc > 0).cpu()            # [B,H,W] (already squeezed)

        # Union DSC
        union_dsc_sum += dice_coeff(pred_union.float(), gt_union.float()).item()

        # One-hot -> [B,H,W,2] then move class axis to channel dim: [B,2,H,W]
        pred_oh = F.one_hot(pred_union.long(), num_classes=2)
        gt_oh   = F.one_hot(gt_union.long(),   num_classes=2)
        pred_oh = pred_oh.movedim(-1, 1).float()
        gt_oh   = gt_oh.movedim(-1, 1).float()

        # NSD on foreground only
        sd_union(pred_oh, gt_oh)
        nsd_batch = sd_union.aggregate()   # tensor([value]) or tensor([nan])
        sd_union.reset()

        nsd_val = torch.nanmean(nsd_batch).item()
        if np.isfinite(nsd_val):
            nsd_union_sum += nsd_val
            nsd_count += 1

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