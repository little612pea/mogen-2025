# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import shutil
import json
from utils.fixseed import fixseed
import subprocess
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation

def main():
    args = train_args()
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')
    if os.path.exists(args.save_dir):
        items = os.listdir(args.save_dir)
        files = [item for item in items if os.path.isfile(os.path.join(args.save_dir, item))]
        if len(files) == 1 and files[0] == 'args.json':
            shutil.rmtree(args.save_dir)
            print(f"Deleted folder {args.save_dir} because it only contains args.json")
        else:
            print(f"Folder {args.save_dir} contains other files or has more than one file.")

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        print(args.overwrite)
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)
    print("training in stage ",args.stage)
    if args.stage == "full-text" or args.stage == "warm-up":
        # all-text
        command = f"cd ../HumanML3D && cp ./texts_humanml/* ./texts && cp ./splits/original_splits/* ./"
        subprocess.run(command, shell=True, check=True)
    elif args.stage == "only-actions":
        # only-actions
        command = f"cd ../HumanML3D && cp ./texts_only_action_processed/* ./texts && cp ./splits/only_action_splits/* ./"
        subprocess.run(command, shell=True, check=True)
    elif args.stage == "wo-physics":
        # remove physics
        command = f"cd ../HumanML3D && cp ./texts_remove_physics_processed/* ./texts && cp ./splits/wo_phy_splits/* ./"
        subprocess.run(command, shell=True, check=True)
    print("creating data loader...")
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    model.to(dist_util.dev())

    if args.stage == "warm-up":
        layers_to_train = ["prior_network","mlp_fc","seqTransEncoder","embed_text"]
        for name, param in model.named_parameters():
            # 检查参数是否在要训练的层中
            if any(name.startswith(layer) for layer in layers_to_train):
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif args.stage=="only-actions":
        for name, param in model.named_parameters():
            # 检查参数是否在要训练的层中
            if 'mlp_fc.atomic_action_branch' in name or 'prior_network' in name:
                param.requires_grad = True  # 确保 mlp_fc.detail_description_mlp 的参数可训练
            else:
                param.requires_grad = False
    elif args.stage=="wo-physics":
        for name, param in model.named_parameters():
            # 检查参数是否在要训练的层中
            if 'mlp_fc.latent_feature_branch' in name or 'prior_network'or "embed_text" in name:
                param.requires_grad = True  # 确保 mlp_fc.detail_description_mlp 的参数可训练
            else:
                param.requires_grad = False
    elif args.stage=="full-text":
        for name, param in model.named_parameters():
            # 检查参数是否在要训练的层中
            if 'prior_network' or "seqTransEncoder" or "embed_text" in name:
                param.requires_grad = True  # 确保 mlp_fc.detail_description_mlp 的参数可训练
            else:
                param.requires_grad = False
    else:
        print("no stage specified, freezing no parameters")
    print("Model parameters:")
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, requires_grad: {param.requires_grad}, shape: {param.shape}")
    model.rot2xyz.smpl_model.eval()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()
