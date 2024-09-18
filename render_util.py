from psbody.mesh import Mesh
from utils.renderer import MeshRenderer
import numpy as np
import tempfile
import cv2
from pathlib import Path
from utils.media import reencode_audio, combine_video_and_audio, convert_video
from tqdm import tqdm
import os
import lmdb
import pickle
import soundfile as sf
import io
import torch
from scipy.io.wavfile import write
from glob import glob
from natsort import natsorted
from safetensors.torch import safe_open


def load_flame(device):
    from models.flame import FLAME, FLAMEConfig
    flame = FLAME(FLAMEConfig)
    flame.to(device)
    flame.eval()
    return flame


def read_safetensor(safetensor_path, device='cpu'):
    params = {}
    with safe_open(safetensor_path, framework='pt', device=device) as f:
        for key in f.keys():
            params[key] = f.get_tensor(key)
    return params


class GenericRenderer:
    def __init__(self, zero_pose=True):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.Mesh = Mesh
        self.uv_coords = np.load('models/data/uv_coords.npz')
        self.size = (640, 640)
        self.fps=25
        self.renderer = MeshRenderer(self.size, black_bg=True)

        self.flame = load_flame(self.device)
        self.faces = self.flame.faces_tensor.detach().cpu().numpy()
        self.texture=None

        self.zero_pose = zero_pose
        # if isinstance(texture, (str, Path)):
        #     texture = cv2.cvtColor(cv2.imread(str(texture)), cv2.COLOR_BGR2RGB)
        

    def render_verts(self, verts_list, audio_path=None, out_path=None):
        if out_path is None:
            out_path = os.path.join('/'.join(audio_path.split('.')[:-1]), 'vis.mp4')
        out_path = Path(out_path)
        tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=out_path.parent)
        writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.size)

        center = np.mean(verts_list, axis=(0, 1))
        for verts in tqdm(verts_list, desc='Rendering'):
            mesh = Mesh(verts, self.faces)
            rendered, _ = self.renderer.render_mesh(mesh, center, tex_img=self.texture, tex_uv=self.uv_coords)
            writer.write(cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
        writer.release()

        if audio_path is not None:
            # needs to re-encode audio to AAC format first, or the audio will be ahead of the video!
            tmp_audio_file = tempfile.NamedTemporaryFile('w', suffix='.aac', dir=out_path.parent)
            reencode_audio(audio_path, tmp_audio_file.name)
            combine_video_and_audio(tmp_video_file.name, tmp_audio_file.name, out_path, copy_audio=False)
            combine_video_and_audio(tmp_video_file.name, audio_path, out_path, copy_audio=False)
            tmp_audio_file.close()
        else:
            print('>>>> Audio path not given, video will be saved without audio')
            convert_video(tmp_video_file.name, out_path)
        tmp_video_file.close()
        
        print(f'>>>> Vis saved to : {out_path}')

    def read_params_from_data_dir(self, data_dir):
        if os.path.exists(os.path.join(data_dir, 'uv_data')):
            params_dir = os.path.join(data_dir, 'uv_data', 'params')
        elif os.path.exists(os.path.join(data_dir, 'params')):
            params_dir = os.path.join(data_dir, 'uv_data', 'params')
        else:
            raise RuntimeError(f'params dir not found inside data_dir: {data_dir}')

        audio_path = os.path.join(data_dir, 'audio', 'audio.wav')
        if not os.path.isfile(audio_path):
            audio_path = None
            print(f">>>> File {os.path.join(data_dir, 'audio', 'audio.wav')} does not exist, rendering without audio")

        param_paths = natsorted(glob(f'{params_dir}/*.safetensors'))

        expressions = []
        poses = []
        shapes = []

        for path in param_paths:
            params = read_safetensor(path, device=self.device)
            expressions.append(params['exp'])
            poses.append(params['pose'])
            shapes.append(params['shape'])
        
        expressions = torch.stack(expressions, dim=0)
        poses = torch.stack(poses, dim=0)  

        if self.zero_pose:
            poses[:, :3] = 0.

        shapes = torch.stack(shapes, dim=0)  

        verts, *_ = self.flame(
            shape_params = shapes,
            expression_params = expressions,
            pose_params = poses
        )
        
        verts_list = [vert.cpu().numpy() for vert in verts]
        self.render_verts(verts_list, audio_path, out_path)
        

    def render_from_lmdb(self, lmdb_dir, key, out_path):
        env = lmdb.open(str(lmdb_dir), readonly=True, lock=False, readahead=False, meminit=False)
        expressions = []
        poses = []
        shapes = []
        audios = []
        with env.begin(write=False) as txn:
            metadata = pickle.loads(txn.get('metadata'.encode()))
            seg_len = metadata['seg_len']
            
            sample_metadata = pickle.loads(txn.get(f'{key}/metadata'.encode()))
            n_frames = sample_metadata['n_frames']
        
            for i in range(n_frames // seg_len + 1):
                entry = pickle.loads(txn.get(f'{key}/{i:03d}'.encode()))
                audio_data, sr = sf.read(io.BytesIO(entry['audio']))
                assert sr == 16000
                audios.append(audio_data)
                shapes.append(entry['coef']['shape'])
                expressions.append(entry['coef']['exp'])
                poses.append(entry['coef']['pose'])  
                # print(len(audios[i]) / 16000, poses[i].shape[0] / 25)
                assert len(audios[i]) / 16000 == poses[i].shape[0] / 25, f'Frames and audio are of different length for {key}, clip {i}'
        expressions = np.concatenate(expressions, axis=0)
        poses = np.concatenate(poses, axis=0)  

        if self.zero_pose:
            poses[:, :3] = 0.

        shapes = np.concatenate(shapes, axis=0)  
        audios = np.concatenate(audios, axis=0) 
        
        verts, *_ = self.flame(
            shape_params = torch.from_numpy(shapes).to(self.device),
            expression_params = torch.from_numpy(expressions).to(self.device),
            pose_params = torch.from_numpy(poses).to(self.device)
        )
        
        verts_list = [vert.cpu().numpy() for vert in verts]
        temp_wav_file = out_path.replace('.mp4', '_temp.wav')
        write(temp_wav_file, sr, audios)
        self.render_verts(verts_list, temp_wav_file, out_path)
        os.remove(temp_wav_file)
        

def main(args):
    zero_pose = args.zero_pose
    renderer = GenericRenderer(zero_pose)

    if args.lmdb_dir:
        out_path = args.out_path
        if not out_path:
            out_path = os.path.join(os.path.dirname(args.lmdb_dir), f'{args.key}_tracking_vis.mp4')
        renderer.render_from_lmdb(args.lmdb_dir, args.key, out_path)
    else:
        out_path = args.out_path
        if not out_path:
            out_path = os.path.join(args.data_dir, 'tracking_vis.mp4')
        renderer.render_from_data_dir(args.data_dir, out_path)
    
        
if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, default=None, help='Path to training assets directory')
    parser.add_argument('--lmdb_dir', type=str, default=None, help='Path to root dir of LMDB dataset')
    parser.add_argument('--key', '-k', type=str, required=True, help='key for the video to be visualized like WDA_BradSchneider/000')
    parser.add_argument('--out_path', '-o', type=str, required=True, help='Path to save the output to')
    parser.add_argument('--zero_pose', action='store_true', help='Whether to render the vis in zero pose')

    args = parser.parse_args()

    if args.data_dir or args.lmdb_dir:
        main(args)
    else:
        raise RuntimeError('One of lmdb_dir and data_dir has to be provided, either both or none were provided')
    
    
    