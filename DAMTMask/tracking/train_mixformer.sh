# There are the detailed training settings for MixFormer, MixFormer-L and MixFormer-1k.
# 1. download pretrained CvT models (CvT-21-384x384-IN-22k.pth/CvT-w24-384x384-IN-22k.pth/CvT-21-384x384-IN-1k.pth) at https://onedrive.live.com/?authkey=%21AMXesxbtKwsdryE&id=56B9F9C97F261712%2115004&cid=56B9F9C97F261712
# 2. set the proper pretrained CvT models path 'MODEL:BACKBONE:PRETRAINED_PATH' at experiment/mixformer/CONFIG_NAME.yaml.
# 3. uncomment the following code to train corresponding trackers.

### Training MixFormer-22k
# Stage1: train mixformer without SPM
# python tracking/train.py --script mixformer --config baseline --save_dir /YOUR/PATH/TO/SAVE/MIXFORMER --mode multiple --nproc_per_node 8
# ## Stage2: train mixformer_online, i.e., SPM (score prediction module)
# python tracking/train.py --script mixformer_online --config baseline --save_dir /YOUR/PATH/TO/SAVE/MIXFORMER_ONLINE --mode multiple --nproc_per_node 8 --stage1_model /STAGE1/MODEL


### Training MixFormer-L-22k
#python tracking/train.py --script mixformer --config baseline_large --save_dir /YOUR/PATH/TO/SAVE/MIXFORMERL --mode multiple --nproc_per_node 8
#python tracking/train.py --script mixformer_online --config baseline_large --save_dir /YOUR/PATH/TO/SAVE/MIXFORMERL_ONLINE --mode multiple --nproc_per_node 8 --stage1_model /STAGE1/MODEL





# Training MixFormer-L-22k-DIMP-Dfcos-384
# python tracking/train.py --script mixformer_online384_dimp_dfcos --config baseline_large --save_dir /home/tiger/tracking_code/MixFormer/workdirs --mode multiple --nproc_per_node 8 --mixformer_model /home/tiger/tracking_code/MixFormer/models/MixFormerOnlineScore_input_size_384_ep0040.pth.tar | tee mixformer_online384_dimp_fcos.txt

# Training MixFormer-L-22k-DIMP-Dfcos-upsample-384 //merge stage2 and stage3
# python tracking/train.py --script mixformer_online384_dimp_dfcos_upsampe --config baseline_large --save_dir /home/tiger/tracking_code/MixFormer/workdirs --mode multiple --nproc_per_node 8 --mixformer_model /home/tiger/tracking_code/MixFormer/models/MixFormerOnlineScore_input_size_384_ep0040.pth.tar | tee mixformer_online384_dimp_dfcos_upsampe.txt

# Training MixFormer-L-22k-DIMP-Dfcos
# python tracking/train.py --script mixformer_online_dimp_dfcos --config baseline_large --save_dir /home/tiger/tracking_code/MixFormer/workdirs --mode multiple --nproc_per_node 8 --mixformer_model /home/tiger/tracking_code/MixFormer/models/RetrainedMixFormerOnlineScore_ep0040.pth.tar | tee mixformer_online_dimp_dfcos.txt

# Training MixFormer-L-22k-DIMP-Dfcos-upsample
# python tracking/train.py --script mixformer_online_dimp_dfcos_upsample --config baseline_large --save_dir /home/tiger/tracking_code/MixFormer/workdirs --mode multiple --nproc_per_node 8 --mixformer_model /home/tiger/tracking_code/MixFormer/models/RetrainedMixFormerOnlineScore_ep0040.pth.tar | tee mixformer_online_dimp_dfcos_upsample.txt







# Training MixFormer-L-22k-DIMP-Robust-Dfcos-384
# python tracking/train.py --script mixformer_online384_dimprobust_dfcos --config baseline_large --save_dir /home/tiger/tracking_code/MixFormer/workdirs --mode multiple --nproc_per_node 8 --mixformer_model /home/tiger/tracking_code/MixFormer/models/MixFormerOnlineScore_input_size_384_ep0040.pth.tar | tee mixformer_online384_dimprobust_dfcos.txt

# Training MixFormer-L-22k-DIMP-Robust-Dfcos
# python tracking/train.py --script mixformer_online_dimprobust_dfcos --config baseline_large --save_dir /home/tiger/tracking_code/MixFormer/workdirs --mode multiple --nproc_per_node 8 --mixformer_model /home/tiger/tracking_code/MixFormer/models/RetrainedMixFormerOnlineScore_ep0040.pth.tar | tee mixformer_online_dimprobust_dfcos.txt


# Training MixFormer-L-22k-DIMP-Robust-Dfcos-upsample-384
# python tracking/train.py --script mixformer_online384_dimprobust_dfcos_upsampe --config baseline_large --save_dir /home/tiger/tracking_code/MixFormer/workdirs --mode multiple --nproc_per_node 8 --mixformer_model /home/tiger/tracking_code/MixFormer/models/MixFormerOnlineScore_input_size_384_ep0040.pth.tar | tee mixformer_online384_dimprobust_dfcos_upsampe.txt


# Training MixFormer-L-22k-DIMP-Robust-Dfcos-upsample
# python tracking/train.py --script mixformer_online_dimprobust_dfcos_upsample --config baseline_large --save_dir /home/tiger/tracking_code/MixFormer/workdirs --mode multiple --nproc_per_node 8 --mixformer_model /home/tiger/tracking_code/MixFormer/models/RetrainedMixFormerOnlineScore_ep0040.pth.tar | tee mixformer_online_dimprobust_dfcos_upsample.txt

# Training MixFormer-L-22k-DIMP-Robust-Dfcos-upsample-256
# python tracking/train.py --script mixformer_online256_dimprobust_dfcos --config baseline_large --save_dir /home/tiger/tracking_code/MixFormer/workdirs --mode multiple --nproc_per_node 8 --mixformer_model /home/tiger/tracking_code/MixFormer/models/RetrainedMixFormerOnlineScore_ep0040.pth.tar | tee mixformer_online256_dimprobust_dfcos.txt


# Training MixFormer-L-22k-DIMP-Robust-Dfcos-Neck
# python tracking/train.py --script mixformer_online_dimprobust_dfcos_neck --config baseline_large --save_dir /home/tiger/tracking_code/MixFormer/workdirs --mode multiple --nproc_per_node 8 --mixformer_model /home/tiger/tracking_code/MixFormer/models/MixFormerOnlineScoreTripleHead_ep0090.pth.tar | tee mixformer_online_dimprobust_dfcos_neck.txt

# Training MixFormer-L-22k-DIMP-Robust-Dfcos-Neck-transt
# python tracking/train.py --script mixformer_online_dimprobust_dfcos_transt --config baseline_large --save_dir /home/tiger/tracking_code/MixFormer/workdirs --mode multiple --nproc_per_node 8 --mixformer_model /home/tiger/tracking_code/MixFormer/models/MixFormerOnlineScoreTripleHead_DimpFcos.pth.tar | tee mixformer_online_dimprobust_dfcos_transt.txt



# Training MixFormer-L-22k-DIMP-Robust-Dfcos-Neck2
# python tracking/train.py --script mixformer_online_dimprobust_dfcos_neck2 --config baseline_large --save_dir /home/tiger/tracking_code/MixFormer/workdirs --mode multiple --nproc_per_node 8 --mixformer_model /home/tiger/tracking_code/MixFormer/models/MixFormerOnlineScoreTripleHead_ep0090.pth.tar | tee mixformer_online_dimprobust_dfcos_neck2.txt


# Training MixFormer-L-22k-DIMP-Robust-reppoint
# python tracking/train.py --script mixformer_online_dimprobust_reppoint --config baseline_large --save_dir /home/tiger/tracking_code/MixFormer/workdirs --mode multiple --nproc_per_node 8 --mixformer_model /home/tiger/tracking_code/MixFormer/models/mixformer_online_dimprobust_dfcos.pth.tar | tee mixformer_online_dimprobust_reppoint.txt


# Training MixFormer-L-22k-DIMP-Robust-reppoint-upsample
# python tracking/train.py --script mixformer_online_dimprobust_reppoint_upsample --config baseline_large --save_dir /home/tiger/tracking_code/MixFormer/workdirs --mode multiple --nproc_per_node 8 --mixformer_model /home/tiger/tracking_code/MixFormer/models/mixformer_online_dimprobust_dfcos.pth.tar | tee mixformer_online_dimprobust_reppoint_upsample.txt



# Training MixFormer-L-22k-DIMP-Robust-reppoint-deep
# python tracking/train.py --script mixformer_online_dimprobust_reppoint_deep --config baseline_large --save_dir /home/tiger/tracking_code/MixFormer/workdirs --mode multiple --nproc_per_node 8 --mixformer_model /home/tiger/tracking_code/MixFormer/models/mixformer_online_dimprobust_dfcos.pth.tar | tee mixformer_online_dimprobust_reppoint_deep.txt




# Training MixFormer-L-22k-DIMP-Robust-reppoint-mask
# python tracking/train.py --script mixformer_dimprobust_reppoint_mask --config baseline_large --save_dir /home/tiger/tracking_code/MixFormer/workdirs --mode multiple --nproc_per_node 8 --mixformer_model /home/tiger/tracking_code/MixFormer/models/dimp_reppoint_pretrained_model.pth.tar | tee mixformer_dimprobust_reppoint_mask.txt

# Training MixFormer-L-22k-DIMP-Robust-reppoint-mask-singletemp
# python tracking/train.py --script mixformer_dimprobust_reppoint_mask_singletemp --config baseline_large --save_dir /home/tiger/tracking_code/MixFormer/workdirs --mode multiple --nproc_per_node 8 --mixformer_model /home/tiger/tracking_code/MixFormer/models/dimp_reppoint_pretrained_model.pth.tar | tee mixformer_dimprobust_reppoint_mask_singletemp_base.txt


# Training MixFormer-Base-22k-DIMP-Robust-reppoint-mask
# python tracking/train.py --script mixformer_dimprobust_reppoint_mask --config baseline --save_dir /home/tiger/tracking_code/MixFormer/workdirs --mode multiple --nproc_per_node 8 --mixformer_model /home/tiger/tracking_code/MixFormer/models/mixformer_online_22k.pth.tar | tee mixformer_dimprobust_reppoint_mask_base.txt

# Training MixFormer-Base-22k-DIMP-Robust-reppoint-mask-singletemp
# python tracking/train.py --script mixformer_dimprobust_reppoint_mask_singletemp --config baseline --save_dir /home/tiger/tracking_code/MixFormer/workdirs --mode multiple --nproc_per_node 8 --mixformer_model /home/tiger/tracking_code/MixFormer/models/mixformer_online_22k.pth.tar | tee mixformer_dimprobust_reppoint_mask_singletemp_base.txt


# Training MixFormer-L-22k-DIMP-Robust-reppoint-segm
# python tracking/train.py --script mixformer_online_dimprobust_reppoint_segm --config baseline_large --save_dir /home/tiger/tracking_code/MixFormer/workdirs --mode multiple --nproc_per_node 8 --mixformer_model /home/tiger/tracking_code/MixFormer/models/mixformer_online_dimprobust_reppoint.pth.tar | tee mixformer_online_dimprobust_reppoint_segm.txt





# Training MixFormer-L-22k-DIMP-Robust-reppoint-mask
# python tracking/train.py --script mixformer_dimprobust_reppoint_mask --config baseline_large_alldata_finetune --save_dir /home/tiger/tracking_code/MixFormer/workdirs --mode multiple --nproc_per_node 8 --mixformer_model /home/tiger/tracking_code/MixFormer/models/pretrained_mixformer_dimprobust_reppoint_mask.pth.tar | tee mixformer_dimprobust_reppoint_mask.txt


# Training MixFormer-Base-22k-DIMP-Robust-reppoint-mask-segm-singletemp
python tracking/train.py --script mixformer_dimprobust_reppoint_mask_segm_singletemp --config baseline --save_dir /home/tiger/tracking_code/MixFormer/workdirs --mode multiple --nproc_per_node 8 --mixformer_model /home/tiger/tracking_code/MixFormer/models/mixformer_dimprobust_reppoint_mask_singletemp.pth.tar | tee mixformer_dimprobust_reppoint_mask_segm_singletemp.txt













