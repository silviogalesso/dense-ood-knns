_base_ = [
    '../datasets/cityscapes.py',
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='VisionTransformerSETR',
        model_name='vit_large_patch16_384',
        img_size=768,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        num_classes=19,
        drop_rate=0.1,
        norm_cfg=norm_cfg,
        pos_embed_interp=True,
        align_corners=False,
    ),
    decode_head=dict(
        type='VisionTransformerUpHead',
        in_channels=1024,
        channels=512,
        in_index=23,
        img_size=768,
        embed_dim=1024,
        num_classes=19,
        conv3x3_conv1x1=False,
        norm_cfg=norm_cfg,
        num_conv=2,
        upsampling_method='bilinear',
        align_corners=False,
        ),
    auxiliary_head=[
        dict(
            type='VisionTransformerUpHead',
            in_channels=1024,
            channels=512,
            in_index=2,
            img_size=768,
            embed_dim=1024,
            num_classes=19,
            norm_cfg=norm_cfg,
            num_conv=2,
            upsampling_method='bilinear',
            align_corners=False,
            conv3x3_conv1x1=False,
        ),
        dict(
            type='VisionTransformerUpHead',
            in_channels=1024,
            channels=512,
            in_index=5,
            img_size=768,
            embed_dim=1024,
            num_classes=19,
            norm_cfg=norm_cfg,
            num_conv=2,
            upsampling_method='bilinear',
            align_corners=False,
            conv3x3_conv1x1=False,
        ),
        dict(
            type='VisionTransformerUpHead',
            in_channels=1024,
            channels=512,
            in_index=8,
            img_size=768,
            embed_dim=1024,
            num_classes=19,
            norm_cfg=norm_cfg,
            num_conv=2,
            upsampling_method='bilinear',
            align_corners=False,
            conv3x3_conv1x1=False,
        ),
    ]
)
