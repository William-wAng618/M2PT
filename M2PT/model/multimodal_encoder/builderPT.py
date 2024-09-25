from .clip_encoderPT import CLIPVisionTower


def build_vision_tower(vision_tower_cfg,VIT_PT_len, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    # if vision_tower.startswith("openai") or vision_tower.startswith("laion") or vision_tower.startswith("clip"):
    return CLIPVisionTower(vision_tower, args=vision_tower_cfg, VIT_PT_len=VIT_PT_len, **kwargs)

    # raise ValueError(f'Unknown vision tower: {vision_tower}')
