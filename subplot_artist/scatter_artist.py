import numpy
from .subplot_artist import SubplotArtist, get_margined_limits


__all__ = ['ScatterArtist']


class ScatterArtist(SubplotArtist):

    def __init__(
            self,
            bind_data=None,
            x_label=None,
            y_label=None,
            title=None,
            is_show_x_ticks=False,
            is_show_y_ticks=False,
            is_show_fit=False,
            width=4,
            height=4,
            x=None,
            y=None,
            annotations=None,
            colors=None):
        SubplotArtist.__init__(
            self,
            width=width,
            height=height,
            title=title,
            x_label=x_label,
            y_label=y_label)
        self.is_show_x_ticks = is_show_x_ticks
        self.is_show_y_ticks = is_show_y_ticks
        self.is_show_fit = is_show_fit
        self.x = x
        self.y = y
        self.annotations = annotations
        self.colors = colors
        self.bind_data = bind_data

    def _get_data_dict(self):
        data = {'x': self.x, 'y': self.y, 'annotations': self.annotations, 'colors': self.colors}
        if self.bind_data is not None:
            bound = self.bind_data()
            if bound is not None:
                data.update(bound)
        return data

    def render(self, axes, bound_data):
        x, y, annotations, colors_ = SubplotArtist._extract_bound_data(bound_data, 'x', 'y', 'annotations', 'colors')

        axes.scatter(x, y, color=colors_)

        if annotations is not None:
            for x_a, y_a, a in zip(x, y, annotations):
                axes.annotate(
                    a, xy=(x_a, y_a), verticalalignment='top', textcoords='data', size='small')

        if self.is_show_fit:
            best_fit = numpy.polyfit(x, y, 1)
            axes.plot(x, best_fit[0] * x + best_fit[1], '-')

        # seems like for scatter data, it is necessary to set the limits manually
        axes.set_xlim(get_margined_limits(x))
        axes.set_ylim(get_margined_limits(y))
        if not self.is_show_x_ticks:
            axes.set_xticks([])
        if not self.is_show_y_ticks:
            axes.set_yticks([])
        else:
            axes.tick_params(labelsize=6)
