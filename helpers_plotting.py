import matplotlib.pyplot as plt
import numpy as np
from cv2 import cv2 # for pylint to work
from mpl_toolkits.mplot3d import Axes3D

def plot_reference_frame(xyz_origin,R_base_to_local,axes_obj,name=''):
    ''' NOT WORKING Plot a reference frame rotated by R with an origin at {x,y,z}'''
    axes_directions = np.array([[1,0,0],[0,1,0],[0,0,1]])
    axes_rotated = R_base_to_local.apply(axes_directions)
    xyz_arrow_tip = axes_rotated + xyz_origin
    axes_obj.quiver([xyz_origin[0]]*3,[xyz_origin[1]]*3,[xyz_origin[2]]*3,xyz_arrow_tip[:,0],xyz_arrow_tip[:,1],xyz_arrow_tip[:,2],length=100,normalize=True)

    # add axes tags
    i = 0
    axes_tags = ['x','y','z']
    for xi,yi,zi in xyz_arrow_tip:
        axes_obj.text(xi, yi, zi, axes_tags[i]+name)
        i+=1

    return axes_obj

def plot_3dscatter(x,y,z):
    ''' Plot 3D scatter with equal axis scales'''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='r', marker='o')

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
    xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(x.max()+x.min())
    yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(y.max()+y.min())
    zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(z.max()+z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(xb, yb, zb):
        ax.plot([xb], [yb], [zb], 'w')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    #plt.axis('equal')
    plt.grid()
    plt.show()

    return None

def plot_3d_wand_leds(x,y,z):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='r', marker='o')

    # add wand tags
    i = 1
    for x, y, z in zip(x,y,z):
        led_tag = 'L' + str(i)
        ax.text(x, y, z, led_tag)
        i = i + 1

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
    xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(x.max()+x.min())
    yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(y.max()+y.min())
    zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(z.max()+z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(xb, yb, zb):
        ax.plot([xb], [yb], [zb], 'w')

    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    #plt.axis('equal')
    plt.grid()
    plt.show()

    return None

def plot_2d_wand_leds(u, v):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(u, v, c='r', marker='o')

    # add wand tags
    i = 1
    for u, v in zip(u,v):
        led_tag = 'L' + str(i)
        ax.text(u, v, led_tag)
        i = i + 1

    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel('U [px]')
    ax.set_ylabel('V [px]')
    #plt.axis('equal')
    plt.grid()
    plt.show()

    return None

def plot_correlation(x,y,z,u,v,title,show=True):
    fig = plt.figure(figsize=(10,6))
    fig.suptitle(title)
    ax3d = fig.add_subplot(121, projection='3d')
    ax2d = fig.add_subplot(122)
    ax3d.scatter(x, y, z, c='r', marker='o')

    # add wand tags
    i = 1
    for xi, yi, zi in zip(x,y,z):
        led_tag = 'L' + str(i)
        ax3d.text(xi, yi, zi, led_tag)
        i = i + 1

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
    xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(x.max()+x.min())
    yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(y.max()+y.min())
    zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(z.max()+z.min())

    for xb, yb, zb in zip(xb, yb, zb):
        ax3d.plot([xb], [yb], [zb], 'w')

    ax3d.set_xlabel('X [mm]')
    ax3d.set_ylabel('Y [mm]')
    ax3d.set_zlabel('Z [mm]')

    ax2d.scatter(u, v, c='r', marker='o')

    # add wand tags
    i = 1
    for u, v in zip(u,v):
        led_tag = 'L' + str(i)
        ax2d.text(u, v, led_tag)
        i = i + 1

    ax2d.set_aspect('equal', adjustable='box')

    ax2d.set_xlabel('U [px]')
    ax2d.set_ylabel('V [px]')
    ax2d.invert_yaxis()
    plt.grid()

    if show:
        plt.show()

    return fig,ax3d,ax2d

# Create CV2 window
def create_window(image, window_name, width=None, height=None, inter=cv2.INTER_AREA):
    ## Helper fcn for plotting
    cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO) # Create window with freedom of dimensions
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    cv2.resizeWindow(window_name, dim)
    cv2.imshow(window_name, image)
    return 0