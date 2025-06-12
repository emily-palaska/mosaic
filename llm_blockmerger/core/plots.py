import numpy as np
import matplotlib.pyplot as plt

lang = 'eng'

def sphere(ax):
    u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:100j]
    x_s = np.cos(u) * np.sin(v)
    y_s = np.sin(u) * np.sin(v)
    z_s = np.cos(v)
    ax.plot_surface(x_s, y_s, z_s, color='lavender', alpha=0.6, edgecolor='none')

def vectors(v1, v2, ax):
    label1 = 'Search Vector s' if lang=='eng' else 'Διάνυσμα Αναζήτησης s'
    label2 = 'Neighbor Vector n' if lang=='eng' else 'Διάνυσμα Γείτονα n'

    ax.plot([0, v1[0]], [0, v1[1]], [0, v1[2]], marker='o', markevery=[-1],
            color='red', linewidth=2, label=label1, zorder=10)
    ax.plot([0, v2[0]], [0, v2[1]], [0, v2[2]], marker='o', markevery=[-1],
            color='blue', linewidth=2, label=label1, zorder=10)

def plane(v1: np.ndarray, v2: np.ndarray, ax):
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    a, b, c = normal
    d = np.dot(normal, v1)

    xx, yy = np.meshgrid(np.linspace(-0.9, 0.9, 50), np.linspace(-0.9, 0.9, 50))
    zz = (d - a * xx - b * yy) / c

    xx_flat = xx.flatten()
    yy_flat = yy.flatten()
    zz_flat = zz.flatten()


    mask = xx_flat ** 2 + yy_flat ** 2 + zz_flat ** 2 >= 1

    xx_out = xx_flat[mask]
    yy_out = yy_flat[mask]
    zz_out = zz_flat[mask]

    ax.scatter(xx_out, yy_out, zz_out, color='violet', zorder=2, s=5, alpha=0.8)

def disc(v1: np.ndarray, v2: np.ndarray, ax):
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    a, b, c = normal
    d = np.dot(normal, v1)

    xx, yy = np.meshgrid(np.linspace(-0.9, 0.9, 100), np.linspace(-0.9, 0.9, 100))
    zz = (d - a * xx - b * yy) / c

    xx_flat = xx.flatten()
    yy_flat = yy.flatten()
    zz_flat = zz.flatten()

    mask = xx_flat**2 + yy_flat**2 + zz_flat**2 < 1

    xx_in = xx_flat[mask]
    yy_in = yy_flat[mask]
    zz_in = zz_flat[mask]

    ax.scatter(xx_in, yy_in, zz_in, color='darkviolet', zorder=10, s=10, alpha=1)

def circle(v1: np.ndarray, v2: np.ndarray, ax):
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)

    u_dir = v1 / np.linalg.norm(v1)
    v_dir = np.cross(normal, u_dir)

    theta = np.linspace(0, 2 * np.pi, 200)
    c = np.array([np.cos(t) * u_dir + np.sin(t) * v_dir for t in theta])
    label = 'Intersection Circle' if lang=='eng' else 'Τομή Επιπέδου Διανυσμάτων'
    ax.plot(c[:, 0], c[:, 1], c[:, 2], color='darkviolet', linewidth=2, label=label, zorder=9)

def plot_sphere():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    v1 = np.array([0, 1.5, 1])
    v2 = np.array([1, 0, 1])
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)


    vectors(v1, v2, ax)
    plane(v1, v2, ax)
    disc(v1, v2, ax)
    circle(v1, v2, ax)
    sphere(ax)

    ax.set_xlim((-1.2, 1.2))
    ax.set_ylim((-1.2, 1.2))
    ax.set_zlim((-1.2, 1.2))
    ax.set_box_aspect([1, 1, 1])
    ax.legend()
    plt.tight_layout()
    plt.savefig('../../plots/sphere.png')

if __name__ == '__main__':
    plot_sphere()