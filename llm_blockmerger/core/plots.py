import numpy as np
import matplotlib.pyplot as plt

lang = 'eng'

def sphere(ax):
    u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:100j]
    x_s = np.cos(u) * np.sin(v)
    y_s = np.sin(u) * np.sin(v)
    z_s = np.cos(v)
    ax.plot_surface(x_s, y_s, z_s, color='lavender', alpha=0.4, edgecolor='none')

def vectors(v1, v2, ax):
    label1 = 'Search Vector s' if lang=='eng' else 'Διάνυσμα Αναζήτησης s'
    label2 = 'Neighbor Vector n' if lang=='eng' else 'Διάνυσμα Γείτονα n'

    ax.quiver(0, 0, 0, *v1, color='red', linewidth=2, arrow_length_ratio=0.08, length=1.0, label=label1)
    ax.quiver(0, 0, 0, *v2, color='blue', linewidth=2, arrow_length_ratio=0.08, length=1.0, label=label2)

def plane(v1: np.ndarray, v2: np.ndarray, ax):
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    a, b, c = normal
    d = np.dot(normal, v1)

    xx, yy = np.meshgrid(np.linspace(-0.9, 0.9, 30), np.linspace(-0.9, 0.9, 30))
    zz = (d - a * xx - b * yy) / c

    ax.plot_surface(xx, yy, zz, alpha=0.2, color='violet')

def circle(v1: np.ndarray, v2: np.ndarray, ax):
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)

    u_dir = v1 / np.linalg.norm(v1)
    v_dir = np.cross(normal, u_dir)

    theta = np.linspace(0, 2 * np.pi, 200)
    c = np.array([np.cos(t) * u_dir + np.sin(t) * v_dir for t in theta])
    label = 'Intersection Circle' if lang=='eng' else 'Τομή Επιπέδου Διανυσμάτων'
    ax.plot(c[:, 0], c[:, 1], c[:, 2], color='darkviolet', linewidth=2, label=label)

def plot_sphere():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    v1 = np.array([0, 1.5, 1])
    v2 = np.array([1, 0, 1])
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    sphere(ax)
    vectors(v1, v2, ax)
    plane(v1, v2, ax)
    circle(v1, v2, ax)

    ax.set_xlim((-1.2, 1.2))
    ax.set_ylim((-1.2, 1.2))
    ax.set_zlim((-1.2, 1.2))
    ax.set_box_aspect([1, 1, 1])
    ax.legend()
    plt.tight_layout()
    plt.savefig('../../plots/sphere.png')

if __name__ == '__main__':
    plot_sphere()