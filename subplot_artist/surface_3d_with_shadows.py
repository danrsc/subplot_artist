import numpy
from matplotlib import colors
from matplotlib.cm import get_cmap
from .subplot_artist import get_margined_limits


__all__ = ['plot_3d_surface_with_shadows']


def plot_3d_surface_with_shadows(
        axes,
        x_grid,
        y_grid,
        z_grid,
        z_std_grid=None,
        cmap=None,
        vmin=None,
        vmax=None,
        x_min_margin=None,
        x_max_margin=None,
        y_min_margin=None,
        y_max_margin=None,
        z_min_margin=None,
        z_max_margin=None,
        x_shadow_location=None,
        y_shadow_location=None,
        shadow_alpha=0.3,
        is_offset_shadows=True):

    if cmap is None:
        cmap = 'plasma'

    color_map = get_cmap(cmap)
    color_map.set_bad(alpha=0)

    if vmin is None:
        vmin = numpy.nanmin(z_grid)
    if vmax is None:
        vmax = numpy.nanmax(z_grid)
    xmin, xmax = get_margined_limits(x_grid, x_min_margin, x_max_margin)
    ymin, ymax = get_margined_limits(y_grid, y_min_margin, y_max_margin)
    zmin, zmax = get_margined_limits(z_grid, z_min_margin, z_max_margin)
    x_shadow = xmin if x_shadow_location == 'min' else xmax
    y_shadow = ymin if y_shadow_location == 'min' else ymax
    z_shadow = zmin

    mat_show_norm = colors.Normalize(vmin=vmin, vmax=vmax)
    face_colors = color_map(mat_show_norm(numpy.ma.masked_where(numpy.isnan(z_grid), z_grid)))
    # noinspection PyTypeChecker
    axes.plot_surface(
        x_grid, y_grid, numpy.full_like(z_grid, z_shadow),
        rstride=1, cstride=1, facecolors=face_colors, shade=False, antialiased=False)

    # draw error bar shadows on the walls and legend lines on the floor
    indicator_diagonal_active = numpy.full(x_grid.shape[0] + x_grid.shape[1], False, dtype=bool)
    for k in range(indicator_diagonal_active.shape[0]):
        x_k = min(k, x_grid.shape[0] - 1)
        y_k = k - x_k
        while y_k < x_grid.shape[1] and x_k >= 0:
            if not numpy.isnan(z_grid[x_k, y_k]):
                indicator_diagonal_active[k] = True
                break
            y_k += 1
            x_k -= 1

    for k in range(indicator_diagonal_active.shape[0]):
        if not indicator_diagonal_active[k]:
            continue

        x_k = min(k, x_grid.shape[0] - 1)
        y_k = k - x_k

        offset_scalar = (k - float(sum(indicator_diagonal_active)) / 2)
        offset_multiplier = .005 if is_offset_shadows else 0
        while y_k < x_grid.shape[1] and x_k >= 0:
            if not numpy.isnan(z_grid[x_k, y_k]):
                offset_y = y_grid[x_k, y_k] + offset_scalar * offset_multiplier * numpy.max(y_grid)
                offset_x = x_grid[x_k, y_k] + offset_scalar * offset_multiplier * numpy.max(x_grid)

                axes.plot(
                    (x_shadow,),
                    (offset_y,),
                    (z_grid[x_k, y_k],),
                    linestyle='None',
                    color=color_map(mat_show_norm(z_grid[x_k, y_k])),
                    alpha=shadow_alpha,
                    marker='o')

                if z_std_grid is not None:
                    axes.plot(
                        (x_shadow, x_shadow),
                        (offset_y, offset_y),
                        (z_grid[x_k, y_k] - z_std_grid[x_k, y_k],
                         z_grid[x_k, y_k] + z_std_grid[x_k, y_k]),
                        color=color_map(mat_show_norm(z_grid[x_k, y_k])),
                        alpha=self.shadow_alpha,
                        marker='_')

                axes.plot(
                    (offset_x,),
                    (y_shadow,),
                    (z_grid[x_k, y_k],),
                    linestyle='None',
                    color=color_map(mat_show_norm(z_grid[x_k, y_k])),
                    alpha=shadow_alpha,
                    marker='o')

                if z_std_grid is not None:
                    axes.plot(
                        (offset_x, offset_x),
                        (y_shadow, y_shadow),
                        (z_grid[x_k, y_k] - z_std_grid[x_k, y_k],
                         z_grid[x_k, y_k] + z_std_grid[x_k, y_k]),
                        color=color_map(mat_show_norm(z_grid[x_k, y_k])),
                        alpha=shadow_alpha,
                        marker='_')

            y_k += 1
            x_k -= 1

    axes.plot_surface(
        x_grid, y_grid, z_grid,
        rstride=1,
        cstride=1,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shade=False,
        antialiased=False,
        alpha=shadow_alpha)

    axes.set_xlim(xmin, xmax)
    axes.set_ylim(ymin, ymax)
    axes.set_zlim(zmin, zmax)
