def create_frame_artists(ax, axis_len=0.04, lw=2):
    x_line = ax.plot([], [], [], "r-", linewidth=lw)[0]
    y_line = ax.plot([], [], [], "g-", linewidth=lw)[0]
    z_line = ax.plot([], [], [], "b-", linewidth=lw)[0]
    origin_pt = ax.scatter([], [], [], color="k", s=20)
    label = ax.text(0, 0, 0, "", color="black", fontsize=9)
    return {
        "x": x_line,
        "y": y_line,
        "z": z_line,
        "origin": origin_pt,
        "label": label,
        "axis_len": axis_len,
    }


def update_frame_artists(frame_artists, T, label_text):
    origin = T[:3, 3]
    Rm = T[:3, :3]
    L = frame_artists["axis_len"]

    x_end = origin + Rm[:, 0] * L
    y_end = origin + Rm[:, 1] * L
    z_end = origin + Rm[:, 2] * L

    frame_artists["x"].set_data([origin[0], x_end[0]], [origin[1], x_end[1]])
    frame_artists["x"].set_3d_properties([origin[2], x_end[2]])

    frame_artists["y"].set_data([origin[0], y_end[0]], [origin[1], y_end[1]])
    frame_artists["y"].set_3d_properties([origin[2], y_end[2]])

    frame_artists["z"].set_data([origin[0], z_end[0]], [origin[1], z_end[1]])
    frame_artists["z"].set_3d_properties([origin[2], z_end[2]])

    frame_artists["origin"]._offsets3d = ([origin[0]], [origin[1]], [origin[2]])
    frame_artists["label"].set_position((origin[0], origin[1]))
    frame_artists["label"].set_3d_properties(origin[2])
    frame_artists["label"].set_text(label_text)