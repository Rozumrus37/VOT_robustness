from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/tiger/tracking_code/MixFormer/data/got10k_lmdb'
    settings.got10k_path = '/home/tiger/tracking_code/MixFormer/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = '/home/tiger/tracking_code/MixFormer/data/lasot_lmdb'
    settings.lasot_path = '/home/tiger/tracking_code/MixFormer/data/lasot'
    settings.network_path = '/home/tiger/tracking_code/MixFormer/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/tiger/tracking_code/MixFormer/data/nfs'
    settings.otb_path = '/home/tiger/tracking_code/MixFormer/data/OTB2015'
    settings.prj_dir = '/home/tiger/tracking_code/MixFormer'
    settings.result_plot_path = '/home/tiger/tracking_code/MixFormer/test/result_plots'
    settings.results_path = '/home/tiger/tracking_code/MixFormer/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/tiger/tracking_code/MixFormer'
    settings.segmentation_path = '/home/tiger/tracking_code/MixFormer/test/segmentation_results'
    settings.tc128_path = '/home/tiger/tracking_code/MixFormer/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/tiger/tracking_code/MixFormer/data/trackingNet'
    settings.uav_path = '/home/tiger/tracking_code/MixFormer/data/UAV123'
    settings.vot_path = '/home/tiger/tracking_code/MixFormer/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings