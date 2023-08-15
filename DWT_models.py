import torch
import torch.nn as nn
from functools import partial
from Dwt_models.EfficientNet_B0_DWT import EfficientNet_DwtSa
from Dwt_models.Resnet18_DWT import BasicBlock, ResNet_DwtSa, Bottleneck, BasicBlock_ori, Bottleneck_ori, ResNet
from Dwt_models.Resnet50_DWT import ResNet as ResNet50
from Dwt_models.Resnet50_DWT import ResNet_DwtSa as ResNet_DwtSa50
from timm.models._builder import build_model_with_cfg
from timm.models._efficientnet_builder import decode_arch_def, round_channels, resolve_act_layer, resolve_bn_args
from Dwt_models.vanillanet_DWT import VanillaNet, VanillaNet_DwtSa, VanillaNet_DwtSa112


def load_DwtSa_weights(pretrained_dict, model):
    model_dict = model.state_dict()
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k.startswith("dwt_sa"):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)

    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
    return model

def vanillaNet_8_Ori(pretrained,num_classes):
    import tools.utils as utils
    model = VanillaNet(dims=[128 * 4, 128 * 4, 256 * 4, 512 * 4, 512 * 4, 1024 * 4, 1024 * 4],
                       strides=[1, 2, 2, 1, 2, 1], num_classes=num_classes)

    if pretrained:
        model_key = 'model|module'
        checkpoint = torch.load("model_save/vanillanet_8.pth", map_location='cpu')
        checkpoint_model = None
        for model_key in model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        utils.load_state_dict(model, checkpoint_model, prefix='')

    return model

def vanillaNet_8_DwtSa(pretrained,num_classes):
    import tools.utils as utils
    model = VanillaNet_DwtSa(dims=[128 * 4, 128 * 4, 256 * 4, 512 * 4, 512 * 4, 1024 * 4, 1024 * 4],
                       strides=[1, 2, 2, 1, 2, 1], num_classes=num_classes)

    if pretrained:
        model_key = 'model|module'
        checkpoint = torch.load("model_save/vanillanet_8.pth", map_location='cpu')
        checkpoint_model = None
        for model_key in model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        utils.load_state_dict(model, checkpoint_model, prefix='')

    return model

def vanillaNet_5_Ori(pretrained,num_classes):
    import tools.utils as utils
    model = VanillaNet(dims=[128*4, 256*4, 512*4, 1024*4], strides=[2,2,2], num_classes=num_classes)

    if pretrained:
        model_key = 'model|module'
        checkpoint = torch.load("model_save/vanillanet_5.pth", map_location='cpu')
        checkpoint_model = None
        for model_key in model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        utils.load_state_dict(model, checkpoint_model, prefix='')

    return model

def vanillaNet_5_DwtSa(pretrained,num_classes):
    import tools.utils as utils
    model = VanillaNet_DwtSa(dims=[128*4, 256*4, 512*4, 1024*4], strides=[2,2,2], num_classes=num_classes)

    if pretrained:
        model_key = 'model|module'
        checkpoint = torch.load("model_save/vanillanet_5.pth", map_location='cpu')
        checkpoint_model = None
        for model_key in model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        utils.load_state_dict(model, checkpoint_model, prefix='')

    return model

def resnet18_DwtSa(pretrained,num_classes):
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2])
    kwargs = {'pretrained_cfg': '', 'pretrained_cfg_overlay': None, 'num_classes': num_classes}
    model = build_model_with_cfg(ResNet_DwtSa, variant="resnet18", pretrained=pretrained, **dict(model_args, **kwargs))
    return model

def resnet18_Ori(pretrained,num_classes):
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2])
    kwargs = {'pretrained_cfg': '', 'pretrained_cfg_overlay': None, 'num_classes': num_classes}
    model = build_model_with_cfg(ResNet, variant="resnet18", pretrained=pretrained, **dict(model_args, **kwargs))
    return model

def resnet50_Ori(pretrained,num_classes):
    model_args = dict(block=Bottleneck_ori, layers=[3, 4, 6, 3])
    kwargs = {'pretrained_cfg': '', 'pretrained_cfg_overlay': None, 'num_classes': num_classes}
    model = build_model_with_cfg(ResNet50, variant="resnet50", pretrained=pretrained, **dict(model_args, **kwargs))
    return model

def resnet50_DwtSa(pretrained,num_classes):
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3])
    kwargs = {'pretrained_cfg': '', 'pretrained_cfg_overlay': None, 'num_classes': num_classes}
    model = build_model_with_cfg(ResNet_DwtSa50, variant="resnet50", pretrained=pretrained, **dict(model_args, **kwargs))
    return model

def efficientNet_b1_DwtSa(pretrained,num_classes):
    kwargs = {'pretrained_cfg': 'ft_in1k', 'pretrained_cfg_overlay': None, 'num_classes': num_classes}
    channel_multiplier=1.0
    depth_multiplier=1.1
    channel_divisor=8

    arch_def = [
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'],
        ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25'],
        ['ir_r4_k5_s2_e6_c192_se0.25'],
        ['ir_r1_k3_s1_e6_c320_se0.25'],
    ]
    round_chs_fn = partial(round_channels, multiplier=channel_multiplier, divisor=channel_divisor)
    model_kwargs = dict(
    block_args=decode_arch_def(arch_def, depth_multiplier, group_size=None),
    num_features=round_chs_fn(1280),
    stem_size=32,
    round_chs_fn=round_chs_fn,
    act_layer=resolve_act_layer(kwargs, 'swish'),
    norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
    **kwargs,
    )

    model_cls = EfficientNet_DwtSa
    model = build_model_with_cfg(
     model_cls, variant="efficientnet_b1", pretrained=pretrained,
     pretrained_strict=True,
     kwargs_filter=None,
     **model_kwargs)
    return model

def efficientNet_b4_DwtSa(pretrained,num_classes):
    kwargs = {'pretrained_cfg': 'ra2_in1k', 'pretrained_cfg_overlay': None, 'num_classes': num_classes}
    channel_multiplier=1.4
    depth_multiplier=1.8
    channel_divisor=8

    arch_def = [
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'],
        ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25'],
        ['ir_r4_k5_s2_e6_c192_se0.25'],
        ['ir_r1_k3_s1_e6_c320_se0.25'],
    ]
    round_chs_fn = partial(round_channels, multiplier=channel_multiplier, divisor=channel_divisor)
    model_kwargs = dict(
    block_args=decode_arch_def(arch_def, depth_multiplier, group_size=None),
    num_features=round_chs_fn(1280),
    stem_size=32,
    round_chs_fn=round_chs_fn,
    act_layer=resolve_act_layer(kwargs, 'swish'),
    norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
    **kwargs,
    )

    model_cls = EfficientNet_DwtSa
    model = build_model_with_cfg(
     model_cls, variant="efficientnet_b4", pretrained=pretrained,
     pretrained_strict=True,
     kwargs_filter=None,
     **model_kwargs)
    return model

