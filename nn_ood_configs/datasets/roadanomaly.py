import os
ds_root = os.environ["DATASETS_ROOT"]
data_root = os.path.join(ds_root, 'RoadAnomaly/RoadAnomaly_jpg')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
max_ratio = 2

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1280, 720),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            # dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    test=dict(
        type='CustomDataset',
        data_root=data_root,
        img_dir="frames",
        ann_dir="frames",
        img_suffix='.jpg',
        seg_map_suffix='.labels/labels_semantic.png',
        pipeline=test_pipeline,
    ),
)

ood_index = 2
