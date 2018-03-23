from .subplot_artist import SubplotArtist


__all__ = ['MatrixArtist']


class MatrixArtist(SubplotArtist):

    def __init__(
            self,
            width,
            height,
            cmap=None,
            vmin=None,
            vmax=None,
            title=None,
            x_label=None,
            y_label=None,
            aspect='auto',
            origin='upper',
            matrix=None,
            bind_data=None,
            show_x_ticks=True,
            show_y_ticks=True,
            map_x_ticks=None,
            map_y_ticks=None,
            tick_params=None):
        SubplotArtist.__init__(
            self,
            width=width,
            height=height,
            title=title,
            x_label=x_label,
            y_label=y_label,
            bind_data=bind_data)
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.aspect = aspect
        self.origin = origin
        self.matrix = matrix
        self.map_x_ticks = map_x_ticks
        self.map_y_ticks = map_y_ticks
        self.show_x_ticks = show_x_ticks
        self.show_y_ticks = show_y_ticks
        self.tick_params = tick_params
        self.__inner_bind = bind_data

    def _bind(self):
        bound_data = SubplotArtist._bind(self)
        if self.matrix is not None:
            if bound_data is None:
                bound_data = dict()
            if 'matrix' not in bound_data:
                bound_data['matrix'] = self.matrix
        return bound_data

    def render(self, axes, bound_data):
        matrix = SubplotArtist._extract_bound_data(bound_data, 'matrix')
        axes.matshow(matrix, cmap=self.cmap, aspect=self.aspect, origin=self.origin, vmin=self.vmin, vmax=self.vmax)
        if not self.show_x_ticks:
            axes.set_xticks([])
        else:
            if self.map_x_ticks is not None:
                x_tick_labels = map(lambda i: self.map_x_ticks(i) if 0 <= i < matrix.shape[1] else i, axes.get_xticks())
                axes.set_xticklabels(x_tick_labels)
        if not self.show_y_ticks:
            axes.set_yticks([])
        else:
            if self.map_y_ticks is not None:
                y_tick_labels = map(lambda i: self.map_y_ticks(i) if 0 <= i < matrix.shape[0] else i, axes.get_yticks())
                axes.set_yticklabels(y_tick_labels)
        if self.tick_params is not None:
            axes.tick_params(**self.tick_params)
