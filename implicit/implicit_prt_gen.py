import numpy as np
import math
import vis_fuse_utils
import trimesh


def factratio(N, D):
    if N >= D:
        prod = 1.0
        for i in range(D + 1, N + 1):
            prod *= i
        return prod
    else:
        prod = 1.0
        for i in range(N + 1, D + 1):
            prod *= i
        return 1.0 / prod


def KVal(M, L):
    return math.sqrt(((2 * L + 1) / (4 * math.pi)) * (factratio(L - M, L + M)))


def AssociatedLegendre(M, L, x):
    if M < 0 or M > L or np.max(np.abs(x)) > 1.0:
        return np.zeros_like(x)

    pmm = np.ones_like(x)
    if M > 0:
        somx2 = np.sqrt((1.0 + x) * (1.0 - x))
        fact = 1.0
        for i in range(1, M + 1):
            pmm = -pmm * fact * somx2
            fact = fact + 2

    if L == M:
        return pmm
    else:
        pmmp1 = x * (2 * M + 1) * pmm
        if L == M + 1:
            return pmmp1
        else:
            pll = np.zeros_like(x)
            for i in range(M + 2, L + 1):
                pll = (x * (2 * i - 1) * pmmp1 - (i + M - 1) * pmm) / (i - M)
                pmm = pmmp1
                pmmp1 = pll
            return pll


def SphericalHarmonic(M, L, theta, phi):
    if M > 0:
        return math.sqrt(2.0) * KVal(M, L) * np.cos(
            M * phi) * AssociatedLegendre(M, L, np.cos(theta))
    elif M < 0:
        return math.sqrt(2.0) * KVal(-M, L) * np.sin(
            -M * phi) * AssociatedLegendre(-M, L, np.cos(theta))
    else:
        return KVal(0, L) * AssociatedLegendre(0, L, np.cos(theta))


def getSHCoeffs(order, phi, theta):
    shs = []
    for n in range(0, order + 1):
        for m in range(-n, n + 1):
            s = SphericalHarmonic(m, n, theta, phi)
            shs.append(s)

    return np.stack(shs, 1)


# Modified from: https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
def cartesian_to_sphere(xyz):
    xy = xyz[:, 0]**2 + xyz[:, 1]**2
    theta = np.arctan2(np.sqrt(xy),
                       xyz[:,
                           2])    # for elevation angle defined from Z-axis down
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    return phi, theta


# Modified from: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
def fibonacci_sphere(samples):
    points = []
    phi = math.pi * (3. - math.sqrt(5.))    # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2    # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)    # radius at y

        theta = phi * i    # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    xyz = np.array(points)
    phi, theta = cartesian_to_sphere(xyz)

    return xyz, phi, theta


def sun_flower(n, offset=0.5):
    indices = np.arange(0, n, dtype=float) + offset

    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5**offset) * indices

    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(
        phi)

    return np.concatenate(
        (x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)),
        axis=-1), phi, theta


def sample_visibility(vertices, faces, origins, dirs):
    mesh = trimesh.Trimesh(vertices, faces)
    sample_count = len(origins)
    dir_count = len(dirs)
    batch_origins = np.repeat(origins, dir_count, axis=0)
    batch_dirs = np.tile(dirs, (sample_count, 1))
    return np.logical_not(mesh.ray.intersects_any(batch_origins,
                                                  batch_dirs)).reshape(
                                                      sample_count, dir_count)


def compute_sample_occlusion(vertices, faces, sample_pts, n=8):
    dirs, _, _ = fibonacci_sphere(n * n)
    return np.logical_not(
        vis_fuse_utils.sample_occlusion_embree(vertices, faces, sample_pts,
                                               dirs))
