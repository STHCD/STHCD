import torch


def residual_unshuffle(x: torch.Tensor, h_out: int, w_out: int):
    """
    Generic residual transform for arbitrary target spatial sizes.

    Args:
        x: Input tensor with shape `[B, C, H, W]`.
        h_out: Target output height.
        w_out: Target output width.

    Returns:
        Tensor with shape `[B, C * r_h * r_w, h_out, w_out]`, where
        `r_h = H / h_out` and `r_w = W / w_out`.
    """
    batch_size, channels, height, width = x.shape
    assert height % h_out == 0 and width % w_out == 0, (
        f"h_out={h_out} must divide H={height}, and w_out={w_out} must divide W={width}."
    )
    r_h = height // h_out
    r_w = width // w_out

    # Split the input into `h_out x w_out` blocks and fold the block coordinates into channels.
    x = x.view(batch_size, channels, h_out, r_h, w_out, r_w)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(batch_size, channels * r_h * r_w, h_out, w_out)


def residual_restore(x: torch.Tensor, h_out: int, w_out: int):
    """
    Generic residual restoration, i.e. the inverse of `residual_unshuffle`.

    Args:
        x: Input tensor with shape `[B, C_in, h_in, w_in]`, where
            `C_in = C_out * r_h * r_w`.
        h_out: Output height after spatial restoration.
        w_out: Output width after spatial restoration.

    Returns:
        Tensor with shape `[B, C_out, h_out, w_out]`.
    """
    batch_size, channels_in, h_in, w_in = x.shape
    assert h_out % h_in == 0 and w_out % w_in == 0, (
        f"H_out={h_out} must be a multiple of h_in={h_in}, and W_out={w_out} must be a multiple of w_in={w_in}."
    )
    r_h = h_out // h_in
    r_w = w_out // w_in
    assert channels_in % (r_h * r_w) == 0, (
        f"C_in={channels_in} must be divisible by r_h * r_w = {r_h * r_w}."
    )

    channels_out = channels_in // (r_h * r_w)
    x = x.view(batch_size, channels_out, r_h, r_w, h_in, w_in)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    return x.view(batch_size, channels_out, h_out, w_out)

