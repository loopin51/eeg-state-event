"""2D sensor positions of the listening datasets in ONE frame: the frame of the ds004408 stage A cache (D-099).

The ds004408 cache stores positions.npy = MNE find_layout(biosemi128) min-max normalised per axis. That equals the MNE
topomap projection (_find_topomap_coords, default sphere (0, 0, 0, 0.095)) of the biosemi128 template, normalised with
the biosemi128 per-axis min / max (checked to 3e-8 on the server, D-099). Every other montage is put in that frame by
  1. a 3D similarity fit (Umeyama: scale, rotation, translation) of its own 3D positions to the biosemi64 template on
     named anchor sites (10-20 names, or template sites matched by distance for numbered montages),
  2. the same topomap projection and the same biosemi128 min / max normalisation.
Biosemi64 / biosemi128 templates share the idealised sphere of the Biosemi caps, so for them step 1 is the identity
(biosemi64 Cz = biosemi128 A1 = (0.5, 0.525)). The ds004408 positions are therefore unchanged, and P1 (ds004408 only)
and P2 (with external pretraining) see the same ds004408 positions.

Positions may fall slightly outside [0, 1] (EGI face and neck sensors); the Fourier embedding of the spatial attention
covers [-0.2, 1.2] with its 0.2 margin. A channel is never placed at the INVALID sentinel (-0.1, -0.1) (asserted).
"""
import numpy as np

INVALID = -0.1
B128_MIN = np.array([-0.0778439, -0.08609924])   # topomap projection of the biosemi128 template (ds004408 cache frame)
B128_MAX = np.array([0.0778439, 0.0778439])

# EGI HydroCel GSN 128/129 sensors at (or next to) 10-20 sites (EGI technical note "HydroCel 10-10 equivalents")
EGI_1020 = {'E11': 'Fz', 'E62': 'Pz', 'E75': 'Oz', 'E36': 'C3', 'E104': 'C4', 'E24': 'F3', 'E124': 'F4', 'E52': 'P3',
            'E92': 'P4', 'E70': 'O1', 'E83': 'O2', 'E33': 'F7', 'E122': 'F8', 'E58': 'P7', 'E96': 'P8', 'E45': 'T7',
            'E108': 'T8', 'E22': 'Fp1', 'E9': 'Fp2', 'Cz': 'Cz'}


def umeyama(A, B):
    """similarity s, R, t minimising |s R a + t - b|^2 over rows of A [n, 3] -> B [n, 3]."""
    ma, mb = A.mean(0), B.mean(0); A0, B0 = A - ma, B - mb; U, S, Vt = np.linalg.svd(B0.T @ A0 / len(A))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt)); R = U @ D @ Vt; s = np.trace(np.diag(S) @ D) / (A0 ** 2).sum(1).mean()
    return s, R, mb - s * R @ ma


def template_3d(name):
    """template positions in the MNE head frame (after the fiducial transform that set_montage applies)."""
    import mne
    names = mne.channels.make_standard_montage(name).ch_names; info = mne.create_info(names, 120.0, 'eeg'); info.set_montage(name)
    return {ch['ch_name']: np.asarray(ch['loc'][:3]) for ch in info['chs']}


def project(names, pos3d):
    """topomap projection (MNE default sphere) of 3D head-frame positions, normalised with the biosemi128 min / max."""
    import mne
    from mne.channels.layout import _find_topomap_coords
    info = mne.create_info(list(names), 120.0, 'eeg')
    info.set_montage(mne.channels.make_dig_montage(dict(zip(names, pos3d)), coord_frame='head'))
    raw = _find_topomap_coords(info, picks=list(range(len(names))), ignore_overlap=True)
    xy = ((raw - B128_MIN) / (B128_MAX - B128_MIN)).astype(np.float32)
    assert not np.any(np.all(np.abs(xy - INVALID) < 1e-4, 1)), 'a sensor landed on the INVALID sentinel'
    return xy


def common_positions(names, pos3d, anchors):
    """names [C], pos3d [C, 3] head-frame positions, anchors {channel name: biosemi64 10-20 name} -> (xy [C, 2], fit dict)."""
    ref = template_3d('biosemi64'); names = list(names); P = np.asarray(pos3d, float)
    src = np.array([P[names.index(a)] for a in anchors]); dst = np.array([ref[b] for b in anchors.values()])
    s, R, t = umeyama(src, dst); err = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
    Q = (s * (R @ P.T)).T + t
    return project(names, Q), dict(anchors=dict(anchors), scale=float(s), mean_error_mm=float(err.mean() * 1000),
                                   max_error_mm=float(err.max() * 1000))


def biosemi_positions(names, montage):
    """Biosemi caps (biosemi64 / biosemi128 templates, channel names as in the template): identity in step 1."""
    ref = template_3d(montage); return project(names, np.array([ref[n] for n in names])), dict(montage=montage, identity=True)


def egi_positions(names, montage='GSN-HydroCel-129'):
    """EGI HydroCel caps: template 3D positions, similarity-fitted to biosemi64 on the 10-20 equivalent sensors."""
    ref = template_3d(montage); P = np.array([ref[n] for n in names])
    anchors = {k: v for k, v in EGI_1020.items() if k in names}
    xy, fit = common_positions(names, P, anchors); fit['montage'] = montage; return xy, fit
