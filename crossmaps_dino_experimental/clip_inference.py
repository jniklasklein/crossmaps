"""Inference-only OpenCLIP loader shared by the fleet, standalone node and A/B test."""


def load_clip_for_inference(model_name, pretrained, device):
    import open_clip

    # Use the published *pair*: the tag selects both weights and eval transform.
    # MobileCLIP2-S2/dfndr2b uses 256px with [0,1] RGB, not ViT-L/14 normalization.
    mobile = model_name == "MobileCLIP2-S2"
    if mobile and pretrained != "dfndr2b":
        raise ValueError("MobileCLIP2-S2 requires the dfndr2b tag in this integration.")
    if mobile and pretrained not in open_clip.list_pretrained_tags_by_model(model_name):
        raise RuntimeError("Install open_clip_torch >= 3.2 with timm MobileCLIP2 support, "
                           "or select the vit-l-14 preset for this run.")

    model, _, preprocess = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained)
    model.eval()
    if mobile:
        from timm.utils import reparameterize_model
        # Fuse the trained FastViT convolution/BN branches once, in eval mode on
        # CPU. In-place avoids duplicating the text tower and temporary GPU weights.
        model = reparameterize_model(model, inplace=True).eval()
    return model.to(device), preprocess, open_clip.get_tokenizer(model_name)
