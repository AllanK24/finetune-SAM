#from segment_anything import SamPredictor, sam_model_registry
from models.sam import sam_model_registry
from models.sam_LoRa import LoRA_Sam
#Scientific computing 
import os
#Pytorch packages
import torch
from torch import nn
import torch.optim as optim
import torchvision
from tensorboardX import SummaryWriter
#Others
from torch.utils.data import DataLoader
from utils.dataset import Public_dataset
from pathlib import Path
from tqdm import tqdm
from utils.dsc import dice_coeff
from quanta import QuanTAConfig, QuanTAModel
import monai
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torchvision
import cfg
from utils.count_params import count_parameters

args = cfg.parse_args()

def cleanup():
    dist.destroy_process_group()

def setup(rank, world_size, model_basic_fn, train_dataset, eval_dataset, dir_checkpoint, backend='nccl'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12333'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    # Pass the datasets down to the training function
    model_basic_fn(args, rank, world_size, train_dataset, eval_dataset, dir_checkpoint)
                

def model_basic(args,rank, world_size,train_dataset,val_dataset,dir_checkpoint):
    device = rank
    
    # 1. Now that we are in a distributed process, create the samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # 2. Create the dataloaders using the samplers
    trainloader = DataLoader(train_dataset, batch_size=args.b, shuffle=False, num_workers=0, sampler=train_sampler, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=args.b, shuffle=False, num_workers=0, sampler=val_sampler, pin_memory=True)
    
    if args.if_warmup:
        b_lr = args.lr / args.warmup_period
    else:
        b_lr = args.lr
    
    epochs = args.epochs
    iter_num = 0
    max_iterations = epochs * len(trainloader) 
    writer = None
    if rank==0:
        writer = SummaryWriter(dir_checkpoint + '/log')
    
    print(f"Running basic DDP example on rank {rank}.")
    # create model and move it to GPU with id rank
    model = sam_model_registry["vit_b"](args,checkpoint=args.sam_ckpt,num_classes=2)

    match args.finetune_type:
        case "lora":
            print("[INFO] Fine Tuning with LoRA")
            model = LoRA_Sam(args,model,r=4).sam
        case "quanta":
            print("[INFO] Fine Tuning with Quanta")
            quanta_config = QuanTAConfig(
                d=args.d,
                per_dim_features=args.per_dim_features,
                target_modules=args.target_modules,
                merge_weights=args.merge_weights,
                bias=args.bias,
                quanta_dropout=args.quanta_dropout
            )
            model = QuanTAModel(quanta_config, model)            
            total, trainable = count_parameters(model)
            print(f"Total parameters: {total:,}")
            print(f"Trainable parameters: {trainable:,}")
            print(f"Frozen parameters: {total - trainable:,}")
        case _:
            print("[INFO] Other fine tune type detected, raising -1")
            raise -1
    
    model.to(device)
    
    # Wrap the model with DDP and tell it which device to use
    ddp_model = DDP(model, device_ids=[device], output_device=device)
    
    optimizer = optim.AdamW(ddp_model.parameters(), lr=b_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1, amsgrad=False)
    optimizer.zero_grad()
    criterion1 = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, to_onehot_y=True,reduction='mean')
    criterion2 = nn.CrossEntropyLoss()
    
    # Only rank 0 shows progress bar
    pbar = tqdm(range(epochs)) if rank == 0 else range(epochs)
    
    should_stop = torch.tensor(0, device=device)  # For early stopping coordination
    
    val_largest_dsc = 0
    last_update_epoch = 0
    try:
        for epoch in pbar if rank == 0 else range(epochs):
            trainloader.sampler.set_epoch(epoch)
            ddp_model.train()
            train_loss = 0
            for i,data in enumerate(trainloader):
                imgs = data['image'].to(device)
                msks = torchvision.transforms.Resize((args.out_size,args.out_size), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(data['mask'])
                msks = msks.to(device)
                
                img_emb = ddp_model.module.image_encoder(imgs)
                sparse_emb, dense_emb = ddp_model.module.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                )
                pred, _ = ddp_model.module.mask_decoder(
                                image_embeddings=img_emb,
                                image_pe=ddp_model.module.prompt_encoder.get_dense_pe(), 
                                sparse_prompt_embeddings=sparse_emb,
                                dense_prompt_embeddings=dense_emb, 
                                multimask_output=True,
                            )
                
                loss_dice = criterion1(pred,msks.float()) 
                loss_ce = criterion2(pred,torch.squeeze(msks.long(),1))
                loss =  loss_dice + loss_ce
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
                if args.if_warmup and iter_num < args.warmup_period:
                    lr_ = args.lr * ((iter_num + 1) / args.warmup_period)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_

                else:
                    if args.if_warmup:
                        shift_iter = iter_num - args.warmup_period
                        assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                        lr_ = args.lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_

                train_loss += loss.item()
                iter_num+=1
                if rank==0:
                    writer.add_scalar('info/lr', lr_, iter_num)
                    writer.add_scalar('info/total_loss', loss, iter_num)
                    writer.add_scalar('info/loss_ce', loss_ce, iter_num)
                    writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            train_loss /= (i+1)
            if rank==0:
                pbar.set_description('Epoch num {}| train loss {} \n'.format(epoch,train_loss))

            if epoch%2==0:
                dist.barrier()
                
                eval_loss=0
                dsc = 0
                ddp_model.eval()
                with torch.no_grad():
                    for i,data in enumerate(valloader):
                        imgs = data['image'].to(device)
                        msks = torchvision.transforms.Resize((args.out_size,args.out_size), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(data['mask'])
                        msks = msks.to(device)
                        
                        
                        img_emb= ddp_model.module.image_encoder(imgs)
                        sparse_emb, dense_emb = ddp_model.module.prompt_encoder(
                            points=None,
                            boxes=None,
                            masks=None,
                        )
                        pred, _ = ddp_model.module.mask_decoder(
                                        image_embeddings=img_emb,
                                        image_pe=ddp_model.module.prompt_encoder.get_dense_pe(), 
                                        sparse_prompt_embeddings=sparse_emb,
                                        dense_prompt_embeddings=dense_emb, 
                                        multimask_output=True,
                                    )
                
                        loss = criterion1(pred,msks.float()) + criterion2(pred,torch.squeeze(msks.long(),1))
                        
                        eval_loss +=loss.item()
                        dsc_batch = dice_coeff((pred[:,1,:,:].cpu()>0).long(),msks.cpu().long()).item()
                        dsc+=dsc_batch

                    eval_loss /= (i+1)
                    dsc /= (i+1)
                    
                    eval_loss_tensor = torch.tensor(eval_loss, device=device)
                    dsc_tensor = torch.tensor(dsc, device=device)
                    
                    dist.all_reduce(eval_loss_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(dsc_tensor, op=dist.ReduceOp.SUM)
                    
                    eval_loss = eval_loss_tensor.item() / world_size
                    dsc = dsc_tensor.item() / world_size
                    
                    if rank==0:
                        writer.add_scalar('eval/loss', eval_loss, epoch)
                        writer.add_scalar('eval/dice', dsc, epoch)
                        
                        print('***Eval Epoch num {} | val loss {} | dsc {} \n'.format(epoch,eval_loss,dsc))
                        if dsc>val_largest_dsc:
                            val_largest_dsc = dsc
                            last_update_epoch = epoch
                            print('largest DSC now: {}'.format(dsc))
                            Path(dir_checkpoint).mkdir(parents=True,exist_ok = True)
                            torch.save(ddp_model.module.state_dict(),dir_checkpoint + '/checkpoint_best.pth')
                        elif (epoch-last_update_epoch)>20:
                            print('Training finished####################')
                            # the network haven't been updated for 20 epochs
                            should_stop.fill_(1)
                    
                    # Broadcast early stopping decision
                    dist.broadcast(should_stop, src=0)
                    dist.barrier()
                    
                    if should_stop.item() == 1:
                        break
                
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        raise
    finally:
        if rank == 0 and writer is not None:
            try:
                # Flush any remaining data
                writer.flush()
                # Give background thread time to finish
                import time
                time.sleep(0.5)
                # Close the writer
                writer.close()
            except Exception as e:
                print(f"Warning: Error closing writer: {e}")
        
        cleanup()

def run_demo(demo_fn, size, model_basic_fn, train_dataset, eval_dataset, dir_checkpoint):
    mp.spawn(demo_fn,
             # Pass the datasets in the args tuple
             args=(size, model_basic_fn, train_dataset, eval_dataset, dir_checkpoint),
             nprocs=size,
             join=True)
    
if __name__ == "__main__":
    dataset_name = args.dataset_name
    print('train dataset: {}'.format(dataset_name))

    num_workers = 0
    if_vis = True

    n_gpus = torch.cuda.device_count()
    # For a 2xT4 machine, this will be 2.
    size = n_gpus
    
    train_dataset = Public_dataset(args,args.img_folder, args.mask_folder, args.train_img_list,phase='train',targets=[f'{args.targets}'],normalize_type='sam',if_prompt=False)
    val_dataset = Public_dataset(args,args.img_folder, args.mask_folder, args.val_img_list,phase='val',targets=[f'{args.targets}'],normalize_type='sam',if_prompt=False)

    run_demo(setup, size, model_basic, train_dataset,val_dataset,args.dir_checkpoint)