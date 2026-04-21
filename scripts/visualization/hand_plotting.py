import numpy as np

finger_paths = [
    [0, 1, 2, 3, 4],          # thumb
    [0, 5, 6, 7, 8, 9],       # index
    [0, 10, 11, 12, 13, 14],  # middle
    [0, 15, 16, 17, 18, 19],  # ring
    [0, 20, 21, 22, 23, 24],  # pinky
]

def create_hand_artists(ax, color="blue", label=None, linewidth=1, alpha=0.5, fontsize=7):
    scatter = ax.scatter([], [], [], color=color, label=label)
    lines = [ax.plot([], [], [], color=color, linewidth=linewidth, alpha=alpha)[0] for _ in range(5)]
    texts = [ax.text(0, 0, 0, "", color=color, fontsize=fontsize) for _ in range(25)]
    return scatter, lines, texts


def update_skeleton(pts, scatter, lines, texts):
    pts = np.asarray(pts, dtype=np.float32)

    scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])

    for i, path in enumerate(finger_paths):
        finger = pts[path]
        lines[i].set_data(finger[:, 0], finger[:, 1])
        lines[i].set_3d_properties(finger[:, 2])

    for i, txt in enumerate(texts):
        txt.set_position((pts[i, 0], pts[i, 1]))
        txt.set_3d_properties(pts[i, 2])
        txt.set_text(str(i))


def clear_skeleton(scatter, lines, texts):
    scatter._offsets3d = ([], [], [])
    for line in lines:
        line.set_data([], [])
        line.set_3d_properties([])
    for txt in texts:
        txt.set_text("")