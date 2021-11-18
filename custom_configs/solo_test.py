_base_ = '../configs/solo/solo_r50_fpn_3x_coco.py'

# ==============================================================================
#                                dataset
# ==============================================================================

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CustomDataset',
        ann_file='/dgx/workspace/kaggle_notebooks/cell_instance/DeepPipeline/exp/sartorius/imgs/train.pkl',
        img_prefix='/dgx/workspace/kaggle_notebooks/cell_instance/DeepPipeline/exp/sartorius/imgs/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True,
                poly2mask=True),
            dict(
                type='Resize',
                img_scale=[(512, 512), (448, 448), (256, 256)],
                multiscale_mode='value',
                keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.68, 116.779, 103.939],
                std=[58.393, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
        ]),
    val=dict(
        type='CustomDataset',
        ann_file='/dgx/workspace/kaggle_notebooks/cell_instance/DeepPipeline/exp/sartorius/imgs/val.pkl',
        img_prefix='/dgx/workspace/kaggle_notebooks/cell_instance/DeepPipeline/exp/sartorius/imgs/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.68, 116.779, 103.939],
                        std=[58.393, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CustomDataset',
        ann_file='/dgx/workspace/kaggle_notebooks/cell_instance/DeepPipeline/exp/sartorius/imgs/val.pkl',
        img_prefix='/dgx/workspace/kaggle_notebooks/cell_instance/DeepPipeline/exp/sartorius/imgs/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.68, 116.779, 103.939],
                        std=[58.393, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))

# ==============================================================================
#                                model
# ==============================================================================
model = dict(
    type='SOLO',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=5),
    mask_head=dict(
        type='SOLOHead',
        num_classes=3,
        in_channels=256,
        stacked_convs=7,
        feat_channels=256,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
        pos_scale=0.2,
        num_grids=[40, 36, 24, 16, 12],
        cls_down_index=0,
        loss_mask=dict(type='DiceLoss', use_sigmoid=True, loss_weight=3.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
    test_cfg=dict(
        nms_pre=500,
        score_thr=0.1,
        mask_thr=0.5,
        filter_thr=0.05,
        kernel='gaussian',
        sigma=2.0,
        max_per_img=100))


# ==============================================================================
#                                      Others
# ==============================================================================


evaluation = dict(interval=1000)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type="TensorboardLoggerHook")
    ],
)
custom_hooks = [
    dict(type="NumClassCheckHook"),
]
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]
