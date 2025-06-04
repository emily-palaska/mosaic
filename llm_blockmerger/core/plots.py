import numpy as np
import matplotlib.pyplot as plt

def plot_sphere():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
    x_s = np.cos(u) * np.sin(v)
    y_s = np.sin(u) * np.sin(v)
    z_s = np.cos(v)
    ax.plot_surface(x_s, y_s, z_s, color='lavender', alpha=0.3, edgecolor='none', label='Διανυσματικός Χώρος')

    v1 = np.array([1, 0.2, 0.1])
    v1 = v1 / np.linalg.norm(v1)

    v2 = np.array([0.8, 0.6, 0.05])
    v2 = v2 / np.linalg.norm(v2)

    ax.quiver(0, 0, 0, *v1, color='red', linewidth=2, arrow_length_ratio=0.08, length=1.0, label='Διάνυσμα Αναζήτησης s')
    ax.quiver(0, 0, 0, *v2, color='blue', linewidth=2, arrow_length_ratio=0.08, length=1.0, label='Διάνυσμα Γείτονα n')

    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)

    u_dir = v1 / np.linalg.norm(v1)
    v_dir = np.cross(normal, u_dir)

    theta = np.linspace(0, 2 * np.pi, 200)
    circle = np.array([np.cos(t)*u_dir + np.sin(t)*v_dir for t in theta])
    ax.plot(circle[:, 0], circle[:, 1], circle[:, 2], color='darkviolet', linewidth=2, label='Τομή Επιπέδου Διανυσμάτων')

    axis_length = 1.2
    ax.quiver(0, 0, 0, axis_length, 0, 0, color='black', linewidth=1.5, arrow_length_ratio=0.05)
    ax.quiver(0, 0, 0, 0, axis_length, 0, color='black', linewidth=1.5, arrow_length_ratio=0.05)
    ax.quiver(0, 0, 0, 0, 0, axis_length, color='black', linewidth=1.5, arrow_length_ratio=0.05)

    ax.text(axis_length, 0, 0, 'X', color='black', fontsize=12)
    ax.text(0, axis_length, 0, 'Y', color='black', fontsize=12)
    ax.text(0, 0, axis_length, 'Z', color='black', fontsize=12)

    ax.set_xlim((-1.2, 1.2))
    ax.set_ylim((-1.2, 1.2))
    ax.set_zlim((-1.2, 1.2))
    ax.set_box_aspect([1,1,1])
    ax.legend()
    plt.tight_layout()
    plt.savefig('../../plots/sphere.png')

if __name__ == '__main__':
    plot_sphere()