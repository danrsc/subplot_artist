import itertools
from .subplot_artist import SubplotArtist


__all__ = ['MultiLineArtist']


class MultiLineArtist(SubplotArtist):

    def __init__(
            self,
            bind_data=None,
            x_label=None,
            y_label=None,
            title=None,
            legend_parameters=None,
            tick_parameters=None,
            x_tick_label_parameters=None,
            width=4,
            height=4,
            x_axis_ticks_and_labels=None,
            series=None,
            labels=None,
            annotations=None,
            colors=None,
            plot_fraction=None):
        SubplotArtist.__init__(
            self,
            width=width,
            height=height,
            x_label=x_label,
            y_label=y_label,
            title=title)
        self.bind_data = bind_data
        self.legend_parameters = legend_parameters
        self.tick_params = tick_parameters
        self.x_tick_label_params = x_tick_label_parameters
        self.plot_fraction = plot_fraction
        self.x_axis_ticks_and_labels = x_axis_ticks_and_labels
        self.series = series
        self.labels = labels
        self.annotations = annotations
        self.colors = colors

    def _get_data_dict(self):
        data = {
            'x_axis_ticks_and_labels': self.x_axis_ticks_and_labels,
            'series': self.series,
            'labels': self.labels,
            'annotations': self.annotations,
            'colors': self.colors}
        if self.bind_data is not None:
            bound = self.bind_data()
            if bound is not None:
                data.update(bound)
        return data

    def render(self, axes, bound_data):
        x_axis_ticks_and_labels, all_series, labels, annotations, colors_ = SubplotArtist._extract_bound_data(
            bound_data, 'x_axis_ticks_and_labels', 'series', 'labels', 'annotations', 'colors')
        if labels is None:
            labels = itertools.cycle([None])
        max_x = None
        min_x = None
        for index_series, (series, label) in enumerate(zip(all_series, labels)):
            separated = list(zip(*series))
            if len(separated) == 2:
                x = separated[0]
                y = separated[1]
                err = None
                if len(x) > 0 and isinstance(x[0], tuple):
                    err = y
                    x, y = zip(*x)
            elif len(separated) == 3:
                x = separated[0]
                y = separated[1]
                err = separated[2]
            else:
                raise ValueError('series must be a list of (x, y) tuples or (x, y, err) tuples or ((x, y), err) tuples')

            if colors_ is not None:
                if len(colors_) == 1:
                    color = colors_[0]
                else:
                    color = colors_[index_series]
            else:
                color = None

            max_current_x = max(x)
            min_current_x = min(x)
            if max_x is None or max_current_x > max_x:
                max_x = max_current_x
            if min_x is None or min_current_x < min_x:
                min_x = min_current_x

            if err is not None:
                axes.errorbar(x, y, yerr=err, label=label, color=color)
            else:
                axes.plot(x, y, label=label, color=color)

        # this is inconsistent with how annotations work in ScatterArtist :(
        if annotations is not None:
            for x_a, y_a, a, vertical in annotations:
                axes.annotate(
                    a, xy=(x_a, y_a), verticalalignment=vertical, textcoords='data', size='small')

        x_tick_label_params = self.x_tick_label_params
        if x_tick_label_params is None:
            x_tick_label_params = dict()

        if x_axis_ticks_and_labels is not None and len(x_axis_ticks_and_labels) > 0 and \
                isinstance(x_axis_ticks_and_labels[0], tuple):
            x_ticks, x_tick_labels = zip(*x_axis_ticks_and_labels)
        else:
            x_ticks = x_axis_ticks_and_labels
            x_tick_labels = None

        if x_ticks is None:
            axes.set_xlim(min_x, max_x)
            x_ticks = axes.get_xticks()
        if x_tick_labels is None:
            # not sure why this is necessary, but calling get_xticklabels sometimes returns
            # all 0's
            x_tick_labels = ['{0}'.format(x) for x in x_ticks]

        axes.set_xticks(x_ticks)
        axes.set_xticklabels(x_tick_labels, **x_tick_label_params)

        if self.plot_fraction is not None:
            box = axes.get_position()
            axes.set_position([box.x0, box.y0, box.width * self.plot_fraction[0], box.height * self.plot_fraction[1]])
        if self.tick_params is not None:
            axes.tick_params(**self.tick_params)
        if self.legend_parameters is not None:
            axes.legend(**self.legend_parameters)
