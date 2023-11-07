import torch


def get_ray_directions_pts(img_h=960,
                           img_w=1280,
                           focal=[1100, 1100],
                           pts=None,
                           normalized_input=True,
                           center=None):
    """
    :param img_h:
    :param img_w:
    :param focal:
    :param pts: requested points, [B, N, 2]
    :param normalized_input
    :param center:
    :return: [B, N, 3]
    """
    cent = center if center is not None else (img_w / 2, img_h / 2)
    if normalized_input:
        directions = torch.stack([
            pts[:, :, 0] / (focal[0] / cent[0]), pts[:, :, 1] /
            (focal[1] / cent[1]),
            torch.ones_like(pts[:, :, 0])
        ], -1)    # (B, N, 3)
    else:
        directions = torch.stack([(pts[:, :, 0] - cent[0]) / focal[0],
                                  (pts[:, :, 1] - cent[1]) / focal[1],
                                  torch.ones_like(pts[:, :, 0])],
                                 -1)    # (B, N, 3)
    return directions
