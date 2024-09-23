from argparse import ArgumentParser
from pathlib import Path
import torch
from safetensors import safe_open
from natsort import natsorted
from glob import glob
from tqdm import tqdm
import numpy as np
import os
import shutil


def read_data(data_folder):
    params = {}
    with safe_open(data_folder, framework="pt", device="cpu") as f:
        for key in ['exp', 'pose', 'shape']:
            params[key] = f.get_tensor(key)
    return params


def generate_style(flame_motion_seq, style_save_path, exp_name, iter):
    cmd = f'python extract_style.py --coef {flame_motion_seq} -o {style_save_path} --exp_name {exp_name} --iter {iter}'
    os.system(cmd)
    
    
def get_motion_seq(params_dir):
    data = natsorted(glob(f"{params_dir}/*.safetensors"))
    
    expressions = []
    poses = []
    motion_seq = {}
    
    for i, d in enumerate(tqdm(data)):
        params = read_data(d)
        expressions.append(params["exp"])
        poses.append(params["pose"])

    motion_seq["exp"] = torch.cat(expressions, dim=0).numpy()
    motion_seq["pose"] = torch.cat(poses, dim=0).numpy()
    
    return motion_seq


def get_shape(params_dir):
    safetensor_path = os.path.join(params_dir, '0.safetensors')
    
    params = read_data(safetensor_path)
    shape = params['shape'].view(-1).cpu().numpy()
    
    return shape
    
    
def generate_diffposetalk_assets(args, params_dir):
    # if assets are given don't make them again
    if args.diffpose_assets_folder is not None:
        print(f'>>>>>>> Loading style and shape from {args.diffpose_assets_folder}')
        shape_save_path = os.path.join(args.diffpose_assets_folder, 'shape.npy')
        motion_seq_save_path = os.path.join(args.diffpose_assets_folder, 'flame_motion_seq.npz')
        style_save_path = os.path.join(args.diffpose_assets_folder, 'style.npy')
        
        shape = get_shape(params_dir)
        shape_save_path = os.path.join(args.out_dir, 'shape.npy')
        np.save(shape_save_path, shape)
        
        return shape_save_path, style_save_path, motion_seq_save_path
    
    # if assets folder is not provided generate and save them
    shape = get_shape(params_dir)
    shape_save_path = os.path.join(args.out_dir, 'shape.npy')
    np.save(shape_save_path, shape)
    
    motion_seq = get_motion_seq(params_dir)
    motion_seq_save_path = os.path.join(args.out_dir, 'flame_motion_seq.npz')
    np.savez(motion_seq_save_path, **motion_seq)
    
    style_save_path = os.path.join(args.out_dir, 'style.npy')
    generate_style(motion_seq_save_path, style_save_path, args.se_exp_name, args.se_iter)
    
    return shape_save_path, style_save_path, motion_seq_save_path
    

if __name__ == '__main__':
        
    parser = ArgumentParser()
    
    parser.add_argument('--audio', '-a', type=Path, required=True, help='path of the input audio signal')
    parser.add_argument('--out_dir', '-o', type=Path, default=None, help='path to save generated params')
    parser.add_argument('--data_folder', type=Path, required=True, help='path to training assets folder')
    parser.add_argument('--se_exp_name', type=str, default='se_deca_run-240909_104413', help='path to training assets folder')
    parser.add_argument('--se_iter', type=str, default='22000', help='path to training assets folder')
    parser.add_argument('--dpt_exp_name', type=str, default='deca_run-240910_113050', help='path to training assets folder')
    parser.add_argument('--dpt_iter', type=str, default='20000', help='path to training assets folder')
    parser.add_argument('--delete_assets', action='store_true', help='deletes the assets - style.npy, flame_motion_seq.npz after generating diffposetalk results')
    parser.add_argument('--diffpose_assets_folder', type=str, default=None, help='deletes the assets - style.npy, flame_motion_seq.npz after generating diffposetalk results')
    parser.add_argument('--scale_audio', '-sa', type=float, default=1.15, help='guiding scale')
    parser.add_argument('--scale_style', '-ss', type=float, default=3, help='guiding scale')
    parser.add_argument('--inf', action='store_true', help='will directly save audio_params.npz in out_dir instead of creating another directory inside it')
    args = parser.parse_args()
            
    # handles both deca and emoca saving conventions
    if os.path.exists(f"{args.data_folder}/uv_data"):
        params_path = f"{args.data_folder}/uv_data/params"
    else:
        params_path = f"{args.data_folder}/params"
    
    # create assets for diffposetalk using tracking data
    shape_save_path, style_save_path, motion_seq_save_path = generate_diffposetalk_assets(args, params_path)
    
    out_dir = None
    if args.out_dir is None:   
        out_dir = os.path.join(args.data_folder)
    else:
        out_dir = os.path.join(args.out_dir)
        os.makedirs(out_dir, exist_ok=True)
    

    vis_out_path = os.path.join(out_dir, f'{args.dpt_exp_name}_{args.dpt_iter}.mp4')
    params_save_path = os.path.join(out_dir, f'audio_params.npz')
        
    cmd = f"python demo.py --exp_name {args.dpt_exp_name} --iter {args.dpt_iter} --params_save_path {params_save_path} -a {args.audio} -c {shape_save_path} -s {style_save_path} -o {vis_out_path} -n 1 -ss {args.scale_style} -sa {args.scale_audio}" 
        
    # cmd = f"python demo.py --exp_name SA-hubert-WM --iter 100000 --params_save_path {params_save_path} -a {args.audio} -c {shape_save_path} -s {style_save_path} -o {vis_out_path} -n 1 -ss 3 -sa 1.15" # -ss 3 -sa 1.15
    
    os.system(cmd)
    
    if args.delete_assets:
        os.remove(shape_save_path)
        os.remove(style_save_path)
        os.remove(motion_seq_save_path)
    
    
    
    
    
    # AUDIO_PARAMS_PATH = str(args.params_save_path)
    # TRACKED_PARAMS_PATH = params_path
    
    # audio_params = dict(np.load(AUDIO_PARAMS_PATH))
    # n_audio_frames = audio_params['exp'].shape[0]
    
    # params_path_list = natsorted(glob(f'{TRACKED_PARAMS_PATH}/*.safetensors'))
    # tracked_params = {'exp': [], 'jaw' : []}
    # # print(params_path_list[:10])
    # for i, path in enumerate(params_path_list):
    #     idx = int(path.split('/')[-1].split('.')[0])
    #     assert i == idx
    #     with safe_open(path, framework="pt", device='cuda') as f:
    #         params = {}
    #         for key in ['exp', 'pose']:
    #             params[key] = f.get_tensor(key)
    #         tracked_params['exp'].append(params['exp'])
    #         tracked_params['jaw'].append(params['pose'][:, 3:])
    
    # tracked_params['exp'] = np.concatenate([t.cpu().numpy() for t in tracked_params['exp']], axis=0)
    # tracked_params['jaw'] = np.concatenate([t.cpu().numpy() for t in tracked_params['jaw']], axis=0)
    
    # max_jaw = np.max(tracked_params['jaw'], axis=0)
    # min_jaw = np.min(tracked_params['jaw'], axis=0)

    # max_exp = np.max(tracked_params['exp'], axis=0)
    # min_exp = np.min(tracked_params['exp'], axis=0)
    
    # eps = 1e-8
    # def min_max_scale(array, new_min, new_max):
    #     min_vals = np.min(array, axis=0)
    #     max_vals = np.max(array, axis=0)
        
    #     return ((array - min_vals) / (max_vals - min_vals + eps)) * (new_max - new_min) + new_min

    # audio_params_min_max = {}
    
    # audio_params_min_max['jaw'] = np.clip(np.copy(audio_params['jaw']), min_jaw, max_jaw)
    # audio_params_min_max['exp'] = np.clip(np.copy(audio_params['exp']), min_exp, max_exp)
    
    # # audio_params_min_max['jaw'] = min_max_scale(np.copy(audio_params['jaw']), min_jaw, max_jaw)
    # # audio_params_min_max['exp'] = min_max_scale(np.copy(audio_params['exp']), min_exp, max_exp)
    
    # shutil.move(AUDIO_PARAMS_PATH, AUDIO_PARAMS_PATH.replace('audio_params.npz', 'audio_params_orig.npz'))
    # np.savez(AUDIO_PARAMS_PATH, **audio_params_min_max)