class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/tiger/tracking_code/MixFormer/workdirs'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/tiger/tracking_code/MixFormer/workdirs/tensorboard'    # Directory for tensorboard files.

        self.pretrained_networks = '/home/tiger/tracking_code/MixFormer/pretrained_networks'

        self.lasot_dir = '/home/tiger/data/tracking_data/LaSOT/LaSOTBenchmark'
        self.got10k_dir = '/home/tiger/data/tracking_data/GOT-10k/train/'
        self.lasot_lmdb_dir = ''
        self.got10k_lmdb_dir = ''
        self.trackingnet_dir = '/home/tiger/data/tracking_data/TrackingNet'
        self.trackingnet_lmdb_dir = ''
        self.coco_dir = '/home/tiger/data/tracking_data/COCO2017'
        self.coco_lmdb_dir = ''
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenet_lmdb_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''

        self.ovis_dir = '/home/tiger/data/tracking_data/OVIS/'
        self.youtubevis2021_dir = '/home/tiger/data/tracking_data/Youtube-VIS-2021/'
        self.youtube_vos_dir = '/home/tiger/data/tracking_data/Youtube-VOS-2019/'
        self.saliency_dir = '/home/tiger/data/tracking_data/saliency/MERGED/'
