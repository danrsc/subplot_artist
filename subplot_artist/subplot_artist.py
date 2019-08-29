from numbers import Integral
import numpy
from matplotlib import rcParams
from matplotlib import gridspec
# noinspection PyUnresolvedReferences
import mpl_toolkits.mplot3d
from six import itervalues


__all__ = [
    'SubplotArtistGrid',
    'SubplotArtistBase',
    'SubplotArtist',
    'SubplotArtist3D',
    'get_margined_limits',
    'render_subplot_artists',
    'make_figure',
    'matshow',
    'bar3d_from_heatmap']


class _SubplotArtistSpec(object):

    def __init__(self, subplot_artist, index_grid_left, index_grid_top, num_rows=1, num_columns=1):
        self._subplot_artist = subplot_artist
        self._index_grid_left = index_grid_left
        self._index_grid_top = index_grid_top
        self._num_rows = num_rows
        self._num_columns = num_columns

    @property
    def subplot_artist(self):
        return self._subplot_artist

    @property
    def width(self):
        return self._subplot_artist.width

    @property
    def height(self):
        return self._subplot_artist.height

    @property
    def index_grid_left(self):
        return self._index_grid_left

    @property
    def index_grid_top(self):
        return self._index_grid_top

    @property
    def num_rows(self):
        return self._num_rows

    @property
    def num_columns(self):
        return self._num_columns

    def update(self, index_grid_left=None, index_grid_top=None, num_rows=None, num_columns=None):
        if index_grid_left is not None:
            self._index_grid_left = index_grid_left
        if index_grid_top is not None:
            self._index_grid_top = index_grid_top
        if num_rows is not None:
            self._num_rows = num_rows
        if num_columns is not None:
            self._num_columns = num_columns


class _RemoveChoice(object):

    def __init__(self):
        pass

    top = 'top'
    bottom = 'bottom'
    left = 'left'
    right = 'right'
    all = 'all'


class SubplotFigure:

    def __init__(self, figure, grid):
        self.figure = figure
        self.grid = grid

    def show(
            self,
            output_figure_path=None,
            is_tight_layout=False,
            tight_layout_pad=1.08,
            tight_layout_h_pad=None,
            tight_layout_w_pad=None,
            should_close_figure=True):

        from matplotlib import pyplot as plt

        if is_tight_layout:
            self.grid.tight_layout(
                self.figure, pad=tight_layout_pad, h_pad=tight_layout_h_pad, w_pad=tight_layout_w_pad)
        if output_figure_path is None:
            # this will block until the figures are closed
            plt.show()
        else:
            self.figure.savefig(output_figure_path, bbox_inches='tight')
            if should_close_figure:
                plt.close(self.figure)
        if should_close_figure:
            del self.figure


class SubplotArtistGrid(object):

    @staticmethod
    def render_artists(
            artists,
            num_rows=None,
            num_columns=None,
            wspace=None,
            hspace=None,
            output_figure_path=None,
            is_tight_layout=False,
            tight_layout_pad=1.08,
            tight_layout_h_pad=None,
            tight_layout_w_pad=None,
            dpi=None):
        grid = SubplotArtistGrid(
            iterable=artists, wspace=wspace, hspace=hspace, num_rows=num_rows, num_columns=num_columns)
        grid.render(
            output_figure_path=output_figure_path,
            is_tight_layout=is_tight_layout,
            tight_layout_pad=tight_layout_pad,
            tight_layout_h_pad=tight_layout_h_pad,
            tight_layout_w_pad=tight_layout_w_pad,
            dpi=dpi)

    @staticmethod
    def artists_figure(
            artists,
            num_rows=None,
            num_columns=None,
            wspace=None,
            hspace=None,
            dpi=None,
            keyed_axes=False):
        grid = SubplotArtistGrid(
            iterable=artists, wspace=wspace, hspace=hspace, num_rows=num_rows, num_columns=num_columns)
        fig, axes = grid.make_figure(dpi)
        if keyed_axes:
            return fig, axes
        flat_axes = list()
        if isinstance(artists, (list, tuple, dict)):
            for key in artists:
                flat_axes.append(axes[key])
            return fig, flat_axes
        # artists is one item
        return fig, axes[artists]

    def __init__(
            self,
            num_rows=None,
            num_columns=None,
            iterable=None,
            wspace=None,
            hspace=None):
        self._current_row = 0
        self._current_column = 0
        self._num_fixed_columns = -1
        self._num_fixed_rows = -1
        self._subplot_artist_specs = list()
        self._grid = []
        self.wspace = wspace if wspace is not None else rcParams['figure.subplot.wspace']
        self.hspace = hspace if hspace is not None else rcParams['figure.subplot.hspace']

        if num_rows is not None:
            if num_rows >= 1:
                self._num_fixed_rows = num_rows
        if num_columns is not None:
            if num_columns >= 1:
                self._num_fixed_columns = num_columns

        if iterable is not None:
            if SubplotArtistGrid._is_spec(iterable) or SubplotArtistGrid._is_artist(iterable):
                if SubplotArtistGrid._is_spec(iterable):
                    artist, num_rows, num_columns = iterable
                    self[0:num_rows, 0:num_columns] = artist
                else:
                    self[0, 0] = iterable
            else:
                iterable = list(iterable)
                if len(iterable) > 0:
                    has_iterable = False
                    has_artist = False
                    for item in iterable:
                        if SubplotArtistGrid._is_spec(item) or SubplotArtistGrid._is_artist(item):
                            has_artist = True
                        else:
                            has_iterable = True
                    if has_artist and has_iterable:
                        raise ValueError('Cannot mix non-artists and artists in iterable')
                    if has_artist:
                        for artist in iterable:
                            self.append_subplot_artist(artist)
                    else:
                        for index_row, row in enumerate(iterable):
                            for artist in row:
                                self.append_subplot_artist(artist)
                            if index_row < len(iterable) - 1:
                                self.next_row()

    @property
    def row_count(self):
        if self._grid is None:
            return 0
        return len(self._grid)

    @property
    def column_count(self):
        if self._grid is None or len(self._grid) == 0:
            return 0
        return len(self._grid[0])

    @property
    def current_row(self):
        return self._current_row

    @property
    def current_column(self):
        return self._current_column

    @property
    def width(self):
        _, column_widths = self.calculate_row_heights_and_column_widths()
        if len(column_widths) == 0:
            return 0
        total_width = sum(column_widths)
        if total_width == 0:
            return 0
        return total_width / (1 - (len(column_widths) - 1) * self.wspace / len(column_widths))

    @property
    def height(self):
        row_heights, _ = self.calculate_row_heights_and_column_widths()
        if len(row_heights) == 0:
            return 0
        total_height = sum(row_heights)
        if total_height == 0:
            return 0
        return total_height / (1 - (len(row_heights) - 1) * self.hspace / len(row_heights))

    def iterate_subplot_artists(self):
        for artist_spec in self._subplot_artist_specs:
            yield artist_spec.subplot_artist

    def get_is_empty_grid(self):
        is_empty = numpy.full((self.row_count, self.column_count), True, dtype=bool)
        for artist_spec in self._subplot_artist_specs:
            is_empty[
                artist_spec.index_grid_top:(artist_spec.index_grid_top + artist_spec.num_rows),
                artist_spec.index_grid_left:(artist_spec.index_grid_left + artist_spec.num_columns)] = False
        return is_empty

    def check_grid(self):
        if len(self._grid) > 0 and len(self._grid[-1]) != len(self._grid[0]):
            raise RuntimeError('wrong number of columns in current row: {0}'.format(len(self._grid) - 1))

    @staticmethod
    def _is_spec(x):
        return (
            (isinstance(x, tuple) or isinstance(x, list))
            and len(x) == 3
            and SubplotArtistGrid._is_artist(x[0])
            and isinstance(x[1], Integral)
            and isinstance(x[2], Integral))

    @staticmethod
    def _is_artist(x):
        return (
            hasattr(x, 'width')
            and hasattr(x, 'height')
            and hasattr(x, 'render_subplots'))

    def append_subplot_artist(self, artist):

        if SubplotArtistGrid._is_spec(artist):
            artist, num_rows, num_columns = artist
        else:
            num_rows = 1
            num_columns = 1

        # noinspection PyTypeChecker
        if 0 <= self._num_fixed_columns < self.current_column + num_columns:
            # noinspection PyTypeChecker
            if num_columns > self._num_fixed_columns:
                raise ValueError('artist out of bounds for fixed number of columns')
            # noinspection PyTypeChecker
            if 0 <= self._num_fixed_rows <= self.current_row + 1:
                raise ValueError('new row is required, but would be out of bounds')
            self.next_row()

        # noinspection PyTypeChecker
        if 0 <= self._num_fixed_rows < self.current_row + num_rows:
            raise ValueError('artist out of bounds for fixed number of rows')

        artist_spec = None
        for i in range(self.current_row, self.current_row + num_rows):
            if i == len(self._grid):
                self._grid.append(list())
            if i == self.current_row:
                while self.current_column < len(self._grid[i]) and \
                        self._grid[i][self.current_column] is not None:
                    self._current_column += 1
                artist_spec = _SubplotArtistSpec(
                    artist,
                    index_grid_left=self.current_column,
                    index_grid_top=self.current_row,
                    num_rows=num_rows,
                    num_columns=num_columns)
            else:
                while len(self._grid[i]) < self.current_column:
                    self._grid[i].append(None)
            for j in range(self.current_column, self.current_column + num_columns):
                if j == len(self._grid[i]):
                    self._grid[i].append(artist_spec)
                elif self._grid[i][j] is None:
                    self._grid[i][j] = artist_spec
                else:
                    raise Exception('bad grid')

        self._subplot_artist_specs.append(artist_spec)
        self._current_column += artist_spec.num_columns

    def next_row(self):

        # noinspection PyTypeChecker
        if 0 <= self._num_fixed_rows <= self.current_row + 1:
            raise RuntimeError('Adding a row would exceed the number of fixed rows')

        if self.current_row < len(self._grid):
            while self.current_column < len(self._grid[self.current_row]) \
                    and self._grid[self.current_row][self.current_column] is not None:
                self._current_column += 1
        if len(self._grid) > 0 and self.current_column != len(self._grid[0]):
            raise RuntimeError('wrong number of columns in current row: {0}, expected {1}, got {2}'.format(
                self.current_row, len(self._grid[0]), self.current_column))
        self._current_column = 0
        self._current_row += 1

    def fix_rows(self):
        self._num_fixed_rows = self.row_count

    def fix_columns(self):
        self._num_fixed_columns = self.column_count

    def __setitem__(self, key, value):

        if not isinstance(key, tuple):
            raise IndexError('key must be an (index_row, index_column) pair')
        try:
            row_key, column_key = key
        except ValueError:
            raise ValueError('key must be an (index_row, index_column) pair')

        if not SubplotArtistGrid._is_artist(value):
            raise ValueError('value must be an artist')

        if isinstance(row_key, slice):
            row1, row2, row_step = row_key.indices(max(row_key.start, row_key.stop, self.row_count))
            if row_step != 1:
                raise IndexError('row index cannot use step other than 1')
        else:
            if row_key < 0:
                row_key += self.row_count
            if row_key < 0:
                raise IndexError('row index out of range: {0}'.format(row_key))
            row1, row2 = row_key, row_key + 1

        if isinstance(column_key, slice):
            col1, col2, column_step = column_key.indices(max(column_key.start, column_key.stop, self.column_count))
            if column_step != 1:
                raise IndexError('column index cannot use step other than 1')
        else:
            if column_key < 0:
                column_key += self.column_count
            if column_key < 0:
                raise IndexError('column index out of range: {0}'.format(column_key))
            col1, col2 = column_key, column_key + 1

        # noinspection PyTypeChecker
        if 0 <= self._num_fixed_columns < col2:
            raise IndexError('column index out of range: {0}'.format(column_key))
        # noinspection PyTypeChecker
        if 0 <= self._num_fixed_rows < row2:
            raise IndexError('row index out of range: {0}'.format(row_key))

        if self.column_count < col2:
            for i in range(self.row_count):
                while len(self._grid[i]) < col2:
                    self._grid[i].append(None)

        while self.row_count < row2:
            self._grid.append([None] * max(self.column_count, col2))

        new_spec = _SubplotArtistSpec(value, col1, row1, row2 - row1, col2 - col1)
        self._subplot_artist_specs.append(new_spec)

        to_modify = dict()

        for index_row in range(row1, row2):
            for index_column in range(col1, col2):
                artist_spec = self._grid[index_row][index_column]
                if artist_spec is not None:
                    to_modify[(artist_spec.index_grid_top, artist_spec.index_grid_left)] = artist_spec
                self._grid[index_row][index_column] = new_spec

        for artist_spec in itervalues(to_modify):
            if artist_spec.index_grid_top < new_spec.index_grid_top:
                if artist_spec.index_grid_left < new_spec.index_grid_left:
                    # we've lost the lower right corner
                    if (new_spec.index_grid_left - artist_spec.index_grid_left >
                            new_spec.index_grid_top - artist_spec.index_grid_top):
                        remove_choice = _RemoveChoice.right
                    else:
                        # if equal, prefer to keep horizontal space
                        remove_choice = _RemoveChoice.bottom
                elif (new_spec.index_grid_left + new_spec.num_columns <
                      artist_spec.index_grid_left + artist_spec.num_columns):
                    # we've lost the lower left corner
                    if (artist_spec.index_grid_left + artist_spec.num_columns -
                            (new_spec.index_grid_left + new_spec.num_columns) >
                            new_spec.index_grid_top - artist_spec.index_grid_top):
                        remove_choice = _RemoveChoice.left
                    else:
                        # if equal, prefer to keep horizontal space
                        remove_choice = _RemoveChoice.bottom
                else:
                    # we've lost the bottom
                    remove_choice = _RemoveChoice.bottom
            elif new_spec.index_grid_top + new_spec.num_rows < artist_spec.index_grid_top + artist_spec.num_rows:
                if artist_spec.index_grid_left < new_spec.index_grid_left:
                    # we've lost the top right corner
                    if (new_spec.index_grid_left - artist_spec.index_grid_left >
                            artist_spec.index_grid_top + artist_spec.num_rows -
                            (new_spec.index_grid_top + new_spec.num_rows)):
                        remove_choice = _RemoveChoice.right
                    else:
                        remove_choice = _RemoveChoice.top
                elif (new_spec.index_grid_left + new_spec.num_columns <
                      artist_spec.index_grid_left + artist_spec.num_columns):
                    # we've lost the top left corner
                    if (artist_spec.index_grid_left + artist_spec.num_columns -
                            (new_spec.index_grid_left + new_spec.num_columns) >
                            new_spec.index_grid_top - artist_spec.index_grid_top):
                        remove_choice = _RemoveChoice.left
                    else:
                        # if equal, prefer to keep horizontal space
                        remove_choice = _RemoveChoice.top
                else:
                    # we've lost the top
                    remove_choice = _RemoveChoice.top
            elif artist_spec.index_grid_left < new_spec.index_grid_left:
                # we've lost the right
                remove_choice = _RemoveChoice.right
            elif (new_spec.index_grid_left + new_spec.num_columns <
                  artist_spec.index_grid_left + artist_spec.num_columns):
                # we've lost the left
                remove_choice = _RemoveChoice.left
            else:
                remove_choice = _RemoveChoice.all

            if remove_choice == _RemoveChoice.right:
                new_num_columns = new_spec.index_grid_left - artist_spec.index_grid_left
                for index_row in range(artist_spec.index_grid_top, artist_spec.index_grid_top + artist_spec.num_rows):
                    for index_column in range(artist_spec.index_grid_left + new_num_columns,
                                              artist_spec.index_grid_left + artist_spec.num_columns):
                        self._grid[index_row][index_column] = None
                artist_spec.update(num_columns=new_num_columns)
            elif remove_choice == _RemoveChoice.bottom:
                new_num_rows = new_spec.index_grid_top - artist_spec.index_grid_top
                for index_row in range(artist_spec.index_grid_top + new_num_rows,
                                       artist_spec.index_grid_top + artist_spec.num_rows):
                    for index_column in range(artist_spec.index_grid_left,
                                              artist_spec.index_grid_left + artist_spec.num_columns):
                        self._grid[index_row][index_column] = None
                artist_spec.update(num_rows=new_num_rows)
            elif remove_choice == _RemoveChoice.left:
                new_left = new_spec.index_grid_left + new_spec.num_columns
                for index_row in range(artist_spec.index_grid_top, artist_spec.index_grid_top + artist_spec.num_rows):
                    for index_column in range(artist_spec.index_grid_left, new_left):
                        self._grid[index_row][index_column] = None
                artist_spec.update(index_grid_left=new_left,
                                   num_columns=artist_spec.index_grid_left + artist_spec.num_columns - new_left)
            elif remove_choice == _RemoveChoice.top:
                new_top = new_spec.index_grid_top + new_spec.num_rows
                for index_row in range(artist_spec.index_grid_top, new_top):
                    for index_column in range(artist_spec.index_grid_left,
                                              artist_spec.index_grid_left + artist_spec.num_columns):
                        self._grid[index_row][index_column] = None
                artist_spec.update(index_grid_top=new_top,
                                   num_rows=artist_spec.index_grid_top + artist_spec.num_rows - new_top)
            elif remove_choice == _RemoveChoice.all:
                self._subplot_artist_specs.remove(artist_spec)
            else:
                raise RuntimeError('Bad code - unknown remove_choice: {0}'.format(remove_choice))

    def calculate_row_heights_and_column_widths(self):
        column_widths = [0] * self.column_count
        row_heights = [0] * self.row_count
        for artist_spec in self._subplot_artist_specs:
            if artist_spec.num_rows == 1:
                row_heights[artist_spec.index_grid_top] = max(
                    row_heights[artist_spec.index_grid_top], artist_spec.height)
            if artist_spec.num_columns == 1:
                column_widths[artist_spec.index_grid_left] = max(
                    column_widths[artist_spec.index_grid_left], artist_spec.width)

        for artist_spec in self._subplot_artist_specs:
            if artist_spec.num_rows > 1:
                current = sum(
                    row_heights[artist_spec.index_grid_top:(artist_spec.index_grid_top + artist_spec.num_rows)])
                if current < artist_spec.height:
                    for i in range(artist_spec.index_grid_top, artist_spec.index_grid_top + artist_spec.num_rows):
                        proportion = 0 if current == 0 else float(row_heights[i]) / current
                        row_heights[i] += int(numpy.floor(proportion * (artist_spec.height - current)))
                    new_total = sum(
                        row_heights[artist_spec.index_grid_top:(artist_spec.index_grid_top + artist_spec.num_rows)])
                    if new_total < artist_spec.height:
                        row_heights[artist_spec.index_grid_top + artist_spec.num_rows - 1] += (
                            artist_spec.height - new_total)
            if artist_spec.num_columns > 1:
                current = sum(
                    column_widths[artist_spec.index_grid_left:(artist_spec.index_grid_left + artist_spec.num_columns)])
                if current < artist_spec.width:
                    for i in range(artist_spec.index_grid_left, artist_spec.index_grid_left + artist_spec.num_columns):
                        proportion = 0 if current == 0 else float(column_widths[i]) / current
                        column_widths[i] += int(numpy.floor(proportion * (artist_spec.width - current)))
                    new_total = sum(
                        column_widths[
                            artist_spec.index_grid_left:(artist_spec.index_grid_left + artist_spec.num_columns)])
                    if new_total < artist_spec.width:
                        column_widths[artist_spec.index_grid_left + artist_spec.num_columns - 1] += (
                            artist_spec.width - new_total)

        return row_heights, column_widths

    def prepare_subplots(self, subplot_spec):

        row_heights, column_widths = self.calculate_row_heights_and_column_widths()

        grid = gridspec.GridSpecFromSubplotSpec(
            self.row_count,
            self.column_count,
            subplot_spec,
            wspace=self.wspace,
            hspace=self.hspace,
            height_ratios=row_heights,
            width_ratios=column_widths)

        return [
            (artist_spec.subplot_artist,
             grid[
                artist_spec.index_grid_top:(artist_spec.index_grid_top + artist_spec.num_rows),
                artist_spec.index_grid_left:(artist_spec.index_grid_left + artist_spec.num_columns)]
             ) for artist_spec in self._subplot_artist_specs]

    def render(
            self,
            output_figure_path=None,
            is_tight_layout=False,
            tight_layout_pad=1.08,
            tight_layout_h_pad=None,
            tight_layout_w_pad=None,
            dpi=None):

        fig, axes = self.make_figure(dpi)
        self.render_subplots(axes)
        fig.show(output_figure_path, is_tight_layout, tight_layout_pad, tight_layout_h_pad, tight_layout_w_pad)

    def make_figure(self, dpi=None):
        from matplotlib import pyplot as plt

        row_heights, column_widths = self.calculate_row_heights_and_column_widths()

        plt.ioff()
        fig = plt.figure(figsize=(self.width, self.height), dpi=dpi)
        grid = gridspec.GridSpec(
            len(row_heights),
            len(column_widths),
            width_ratios=column_widths,
            height_ratios=row_heights,
            wspace=self.wspace,
            hspace=self.hspace)

        axes = dict()
        for artist_spec in self._subplot_artist_specs:
            subplot_spec = grid[
                artist_spec.index_grid_top:(artist_spec.index_grid_top + artist_spec.num_rows),
                artist_spec.index_grid_left:(artist_spec.index_grid_left + artist_spec.num_columns)]
            artist_spec.subplot_artist.add_axes(fig, subplot_spec, axes)

        return SubplotFigure(fig, grid), axes

    def add_axes(self, fig, subplot_spec, axes_result):
        for artist, subplot_spec_for_artist in self.prepare_subplots(subplot_spec):
            artist.add_axes(fig, subplot_spec_for_artist, axes_result)

    def render_subplots(self, axes):
        for artist in self._subplot_artist_specs:
            artist.render_subplots(axes)


class SubplotArtistBase(object):

    def __init__(
            self,
            width=None,
            height=None,
            subplot_artist_grid=None):

        self._subplot_artist_grid = subplot_artist_grid

        if subplot_artist_grid is not None:
            width = max(width, subplot_artist_grid.width) if width is not None else subplot_artist_grid.width
            height = max(height, subplot_artist_grid.height if height is not None else subplot_artist_grid.height)
        else:
            if width is None:
                raise ValueError('width must be specified if subplot_artist_grid is not specified')
            if height is None:
                raise ValueError('height must be specified if subplot_artist_grid is not specified')

        self._width = width
        self._height = height

    def add_axes(self, fig, subplot_spec, axes_result):
        if self._subplot_artist_grid is not None:
            return self._subplot_artist_grid.add_axes(fig, subplot_spec, axes_result)
        axes_result[self] = self._add_subplot(fig, subplot_spec)

    def render_subplots(self, axes):
        if self._subplot_artist_grid is not None:
            self._subplot_artist_grid.render_subplots(axes)
        else:
            self.render(axes[self])

    def _add_subplot(self, fig, subplot_spec):
        return fig.add_subplot(subplot_spec)

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def subplot_artist_grid(self):
        return self._subplot_artist_grid

    def render(self, axes):
        if isinstance(axes, (dict, list, tuple)):
            for ax in axes:
                ax.set_facecolor('red')
        else:
            axes.set_facecolor('red')


class SubplotArtist(SubplotArtistBase):

    def __init__(self, width=None, height=None, colorbar_artist=None, render_fn=None, **render_kwargs):
        super(SubplotArtist, self).__init__(width=width, height=height)
        self.render_fn = render_fn
        self.render_kwargs = render_kwargs
        self.colorbar_artist = colorbar_artist

    def render(self, axes):
        if self.render_fn is None:
            axes_image = super(SubplotArtist, self).render(axes)
        else:
            render_kwargs = {} if self.render_kwargs is None else self.render_kwargs
            axes_image = self.render_fn(axes, **render_kwargs)
        if self.colorbar_artist is not None:
            self.colorbar_artist.set_mappable(axes_image, axes)
        return axes_image


class SubplotArtist3D(SubplotArtist):

    def __init__(self, width=None, height=None, colorbar_artist=None, render_fn=None, **render_kwargs):
        super(SubplotArtist3D, self).__init__(
            width=width, height=height, colorbar_artist=colorbar_artist, render_fn=render_fn, **render_kwargs)

    def _add_subplot(self, fig, subplot_spec):
        axes = fig.add_subplot(subplot_spec, projection='3d')
        return axes


def get_margined_limits(data_to_limit, min_margin=None, max_margin=None):
    if min_margin is None:
        min_margin = 0.05
    if max_margin is None:
        max_margin = 0.05
    max_d = numpy.nanmax(data_to_limit)
    min_d = numpy.nanmin(data_to_limit)
    span_d = max_d - min_d
    if span_d == 0:
        if min_d == 0:
            # everything is 0
            return -min_margin, max_margin
        elif min_d > 0:
            # everything is some positive constant
            return (1 - min_margin) * min_d, (1 + max_margin) * min_d
        # everything is some negative constant
        return (1 + min_margin) * min_d, (1 - max_margin) * min_d
    return min_d - min_margin * span_d, max_d + max_margin * span_d


def matshow(ax, data, aspect='auto', **kwargs):
    return ax.matshow(data, aspect=aspect, **kwargs)


def bar3d_from_heatmap(ax, heatmap, dx=1, dy=1, **kwargs):
    x = numpy.arange(heatmap.shape[0])
    y = numpy.arange(heatmap.shape[1])
    x, y = numpy.meshgrid(x, y)
    return ax.bar3d(
        numpy.ravel(x),
        numpy.ravel(y),
        0,
        dx=dx,
        dy=dy,
        dz=numpy.ravel(heatmap),
        **kwargs)


class MatrixArtist(SubplotArtist):

    def __init__(self, matrix, width=None, height=None, colorbar_artist=None, **render_kwargs):
        super(MatrixArtist, self).__init__(
            width=width, height=height, colorbar_artist=colorbar_artist, render_kwargs=render_kwargs)
        self.matrix = matrix

    def render(self, axes):
        matshow(axes, self.matrix, **self.render_kwargs)


render_subplot_artists = SubplotArtistGrid.render_artists
make_figure = SubplotArtistGrid.artists_figure
