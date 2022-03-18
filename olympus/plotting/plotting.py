"""Plotting functions."""
import awkward as ak
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def plot_event(det, hit_times, record=None, plot_tfirst=False, plot_hull=False):

    if plot_tfirst:
        plot_target = ak.fill_none(ak.firsts(hit_times, axis=1), np.nan)
    else:
        plot_target = np.log10(ak.count(hit_times, axis=1))

    mask = (plot_target > 0) & (plot_target != np.nan)

    traces = [
        go.Scatter3d(
            x=det.module_coords[mask, 0],
            y=det.module_coords[mask, 1],
            z=det.module_coords[mask, 2],
            mode="markers",
            marker=dict(
                size=5,
                color=plot_target[mask],  # set color to an array/list of desired values
                colorscale="Viridis",  # choose a colorscale
                opacity=0.8,
                showscale=True,
            ),
        ),
        go.Scatter3d(
            x=det.module_coords[~mask, 0],
            y=det.module_coords[~mask, 1],
            z=det.module_coords[~mask, 2],
            mode="markers",
            marker=dict(
                size=1,
                color="black",  # set color to an array/list of desired values
                colorscale="Viridis",  # choose a colorscale
                opacity=0.1,
            ),
        ),
    ]

    if record is not None:
        positions = []
        sizes = []
        for source in record.sources:
            sizes.append(np.asscalar((np.log10(source.n_photons) / 2) ** 2))
            positions.append(
                [source.position[0], source.position[1], source.position[2]]
            )
        positions = np.asarray(positions)
        traces.append(
            go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode="markers",
                marker=dict(size=sizes, color="black", opacity=0.5, line=dict(width=0)),
            )
        )
    if plot_hull:
        # Cylinder
        radius = det.outer_cylinder[0]
        height = det.outer_cylinder[1]
        z = np.linspace(-height / 2, height / 2, 100)
        theta = np.linspace(0, 2 * np.pi, 50)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = radius * np.cos(theta_grid)
        y_grid = radius * np.sin(theta_grid)

        traces.append(
            go.Surface(
                x=x_grid,
                y=y_grid,
                z=z_grid,
                colorscale=[[0, "blue"], [1, "blue"]],
                opacity=0.2,
            )
        )
    fig = go.Figure(
        data=traces,
    )
    fig.update_layout(
        showlegend=False,
        height=700,
        width=1400,
        coloraxis_showscale=True,
        scene=dict(
            xaxis=dict(range=[-1500, 1500]),
            yaxis=dict(range=[-1500, 1500]),
            zaxis=dict(range=[-1500, 1500]),
        ),
    )
    fig.update_coloraxes(colorbar_title=dict(text="log10(det. photons)"))
    fig.show()
    return fig


def plot_events(
    det, events, labels=None, records=None, plot_tfirst=False, plot_hull=False
):
    nplt = int(np.ceil(np.sqrt(len(events))))
    fig = plt.figure(figsize=(nplt * 4, nplt * 4))

    if plot_hull:
        # Cylinder
        radius = det.outer_cylinder[0]
        height = det.outer_cylinder[1]
        z = np.linspace(-height / 2, height / 2, 100)
        theta = np.linspace(0, 2 * np.pi, 100)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = radius * np.cos(theta_grid)
        y_grid = radius * np.sin(theta_grid)

    for i, hit_times in enumerate(events):
        if plot_tfirst:
            plot_target = ak.fill_none(ak.firsts(hit_times, axis=1), np.nan)
        else:
            plot_target = ak.count(hit_times, axis=1)

        mask = (plot_target > 0) & (plot_target != np.nan)

        ax = fig.add_subplot(nplt, nplt, i + 1, projection="3d")

        ax.scatter(
            det.module_coords[mask, 0],
            det.module_coords[mask, 1],
            det.module_coords[mask, 2],
            c=plot_target[mask],
            cmap=plt.cm.viridis,
            norm=matplotlib.colors.LogNorm(),
        )

        if labels is not None:
            ax.set_title(labels[i], fontsize="small")
        if records is not None:
            record = records[i]
            for source in record.sources:
                ms = (np.log10(source.amp) / 2) ** 2
                ax.plot(
                    [source.pos[0]],
                    [source.pos[1]],
                    [source.pos[2]],
                    "ok",
                    markersize=ms,
                )

        if plot_hull:
            ax.plot_surface(
                x_grid, y_grid, z_grid, linewidth=0, antialiased=True, alpha=0.1
            )

        ax.set_ylim3d(-700, 700)
        ax.set_zlim3d(-500, 500)

    return fig
