ann_file_test = './data/animal-kingdom/test_annotations.txt'
ann_file_train = './data/animal-kingdom/train_annotations.txt'
ann_file_val = './data/animal-kingdom/test_annotations.txt'
root = './data/animal-kingdom/rawframes'

dataset_type = 'RawframeDataset'
work_dir = './work_dirs/ircsn_ig65m-pretrained-r152-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb'

auto_scale_lr = dict(base_batch_size=12, enable=False)
default_hooks = dict(
    checkpoint=dict(
        interval=2, max_keep_ckpts=5, save_best='auto', type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=20, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
file_client_args = dict(io_backend='disk')
find_unused_parameters = True
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
model = dict(
    backbone=dict(
        bn_frozen=True,
        bottleneck_mode='ir',
        depth=152,
        norm_eval=True,
        pretrained=
        'https://download.openmmlab.com/mmaction/recognition/csn/ircsn_from_scratch_r152_ig65m_20200807-771c4135.pth',
        pretrained2d=False,
        type='ResNet3dCSN',
        with_pool2=False,
        zero_init_residual=False),
    cls_head=dict(
        average_clips='prob',
        dropout_ratio=0.5,
        in_channels=2048,
        init_std=0.01,
        num_classes=400,
        spatial_type='avg',
        type='I3DHead'),
    data_preprocessor=dict(
        format_shape='NCTHW',
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='ActionDataPreprocessor'),
    test_cfg=dict(max_testing_views=10),
    train_cfg=None,
    type='Recognizer3D')
optim_wrapper = dict(
    clip_grad=dict(max_norm=40, norm_type=2),
    optimizer=dict(lr=0.0005, momentum=0.9, type='SGD', weight_decay=0.0001))
param_scheduler = [
    dict(begin=0, by_epoch=True, end=16, start_factor=0.1, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=58,
        gamma=0.1,
        milestones=[
            32,
            48,
        ],
        type='MultiStepLR'),
]
randomness = dict(deterministic=False, diff_rank_seed=False, seed=None)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=ann_file_test,
        data_prefix=dict(img=root),
        multi_class=True,
        num_classes=400,
        pipeline=[
            dict(
                clip_len=32,
                frame_interval=2,
                num_clips=10,
                test_mode=True,
                type='SampleFrames'),
            dict(type='RawFrameDecode', **file_client_args),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(crop_size=256, type='ThreeCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type=dataset_type),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(metric_list='mean_average_precision', type='AccMetric')
test_pipeline = [
    dict(
        clip_len=32,
        frame_interval=2,
        num_clips=10,
        test_mode=True,
        type='SampleFrames'),
    dict(type='RawFrameDecode', **file_client_args),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(crop_size=256, type='ThreeCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
train_cfg = dict(
    max_epochs=58, type='EpochBasedTrainLoop', val_begin=1, val_interval=3)
train_dataloader = dict(
    batch_size=12,
    dataset=dict(
        ann_file=ann_file_train,
        data_prefix=dict(img=root),
        multi_class=True,
        num_classes=400,
        pipeline=[
            dict(
                clip_len=32,
                frame_interval=2,
                num_clips=1,
                type='SampleFrames'),
            dict(type='RawFrameDecode', **file_client_args),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(type='RandomResizedCrop'),
            dict(keep_ratio=False, scale=(
                224,
                224,
            ), type='Resize'),
            dict(flip_ratio=0.5, type='Flip'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        type='RawframeDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(clip_len=32, frame_interval=2, num_clips=1, type='SampleFrames'),
    dict(type='RawFrameDecode', **file_client_args),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(type='RandomResizedCrop'),
    dict(keep_ratio=False, scale=(
        224,
        224,
    ), type='Resize'),
    dict(flip_ratio=0.5, type='Flip'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=ann_file_val,
        data_prefix=dict(img=root),
        multi_class=True,
        num_classes=400,
        pipeline=[
            dict(
                clip_len=32,
                frame_interval=2,
                num_clips=1,
                test_mode=True,
                type='SampleFrames'),
            dict(type='RawFrameDecode', **file_client_args),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type=dataset_type),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(metric_list='mean_average_precision', type='AccMetric')
val_pipeline = [
    dict(
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True,
        type='SampleFrames'),
    dict(type='RawFrameDecode', **file_client_args),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(crop_size=224, type='CenterCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])

