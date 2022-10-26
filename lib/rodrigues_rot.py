import numpy as np
from math import cos, sin


def rodrigues(pos,axis,angle):
    """Rotates a point POS around AXIS by ANGLE and returns.

    POS and AXIS must be numpy arrays of length 3.
    AXIS must be a nonzero vector.
    ANGLE is in radians."""

    if (np.linalg.norm(axis)<0.001):
        raise Exception("Axis not normalizable!")

    axis=axis/np.linalg.norm(axis)

    pos_proj=np.dot(axis,pos)*axis
    pos_orth=np.cross(axis,pos)

    return pos*cos(angle) + pos_orth*sin(angle) + pos_proj*(1-cos(angle))
