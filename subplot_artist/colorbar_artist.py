from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .subplot_artist import SubplotArtist


__all__ = ['ColorbarArtist']


class ColorbarArtist(SubplotArtist):

    def __init__(self, width=None, height=None):
        super(ColorbarArtist, self).__init__(width=width, height=height)
        self.mappable = None
        self.mappable_axes = None
        self.axes = None
        self.fig = None

    def set_mappable(self, mappable, axes):
        self.mappable_axes = axes
        self.mappable = mappable
        self.render_if_ready()

    def render(self, axes):
        self.axes = axes
        self.render_if_ready()

    def _add_subplot(self, fig, subplot_spec):
        self.fig = fig
        return super(ColorbarArtist, self)._add_subplot(fig, subplot_spec)

    def render_if_ready(self):
        if self.axes is not None and self.mappable is not None:
            self.fig.colorbar(self.mappable, ax=self.mappable_axes, cax=self.axes)
