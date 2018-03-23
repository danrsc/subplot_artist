from .subplot_artist import SubplotArtist


__all__ = ['GenericSubplotArtist']


class GenericSubplotArtist(SubplotArtist):

    def __init__(
            self,
            render_fn,
            width=4,
            height=4,
            x_label=None,
            y_label=None,
            title=None):
        super(GenericSubplotArtist, self).__init__(
            width=width, height=height, title=title, x_label=x_label, y_label=y_label)
        self._render_fn = render_fn

    def render(self, axes, bound_data):
        self._render_fn(axes)
