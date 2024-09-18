# lmdb_dir='/mnt/data2/anirudh/3DLipSync/diffposetalk/tfhp_hdtf_dataset'
lmdb_dir='/mnt/data2/anirudh/3DLipSync/diffposetalk/tfhp_hdtf_dataset_mouth_interior_error_less_than_5'

key=TH_00299/000
# dpt_exp_name=tfhp_hdtf_dataset_mouth_interior_error_less_than_5_savgol_order_3-240917_134344
# dpt_exp_name=deca_wo_cfg-240917_054404
dpt_exp_name=deca_run-240910_113050
dpt_iter=80000
out_dir=TH_00299_w_cfg_ss_2

data_dir=/mnt/data2/anirudh/processed_videos/$key

# audio=/mnt/data1/anirudh/deca_training_assets/vivo_audio_files/vivo_campaign_2_v2.mp3 
audio=$data_dir/audio/audio.wav

params_save_path=$data_dir/audio_params.npz

mkdir $out_dir

# python /mnt/data2/ketan/video_from_photos.py --image_dir $data_dir/images --audio_path $audio --output_video_path $out_dir/cropped_video.mp4

# python render_util.py --lmdb_dir $lmdb_dir -k $key -o $out_dir/tracking_vis.mp4 --zero_pose

# soxi $audio

python audio2exp.py --out_dir $out_dir -a $audio --dpt_exp_name $dpt_exp_name --dpt_iter $dpt_iter --data_folder $data_dir --delete_assets -ss 2 -sa 3