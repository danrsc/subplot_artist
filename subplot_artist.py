from numbers import Number, Integral
import numpy
from matplotlib import rcParams
from matplotlib import gridspec
# noinspection PyUnresolvedReferences
import mpl_toolkits.mplot3d
from .pyplot_lazy import pyplot_lazy_import
from six import itervalues


__all__ = ['SubplotArtistGrid', 'SubplotArtist', 'SubplotArtist3D', 'get_margined_limits']


class _SubplotArtistSpec:

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


class _RemoveChoice:

    def __init__(self):
        pass

    top = 'top'
    bottom = 'bottom'
    left = 'left'
    right = 'right'
    all = 'all'


class SubplotArtistGrid:

    def __init__(
            self,
            num_rows=None,
            num_columns=None,
            iterable=None,
            wspace=None,
            hspace=None):
        self.__current_row = 0
        self.__current_column = 0
        self.__num_fixed_columns = -1
        self.__num_fixed_rows = -1
        self.__subplot_artist_specs = list()
        self.__grid = []
        self.wspace = wspace if wspace is not None else rcParams['figure.subplot.wspace']
        self.hspace = hspace if hspace is not None else rcParams['figure.subplot.hspace']

        if num_rows is not None:
            if not isinstance(num_rows, Integral):
                raise ValueError('If num_rows is specified, then it must be an int')
            if num_rows >= 1:
                self.__num_fixed_rows = num_rows
        if num_columns is not None:
            if not isinstance(num_columns, Integral):
                raise ValueError('If num_columns is specified, then it must be an int')
            if num_columns >= 1:
                self.__num_fixed_columns = num_columns

        if iterable is not None:
            if SubplotArtistGrid.__is_spec(iterable) or SubplotArtistGrid.__is_artist(iterable):
                if SubplotArtistGrid.__is_spec(iterable):
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
                        if SubplotArtistGrid.__is_spec(item) or SubplotArtistGrid.__is_artist(item):
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
        if self.__grid is None:
            return 0
        return len(self.__grid)

    @property
    def column_count(self):
        if self.__grid is None or len(self.__grid) == 0:
            return 0
        return len(self.__grid[0])

    @property
    def current_row(self):
        return self.__current_row

    @property
    def current_column(self):
        return self.__current_column

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
        for artist_spec in self.__subplot_artist_specs:
            yield artist_spec.subplot_artist

    def get_is_empty_grid(self):
        is_empty = numpy.full((self.row_count, self.column_count), True, dtype=bool)
        for artist_spec in self.__subplot_artist_specs:
            is_empty[
                artist_spec.index_grid_top:(artist_spec.index_grid_top + artist_spec.num_rows),
                artist_spec.index_grid_left:(artist_spec.index_grid_left + artist_spec.num_columns)] = False
        return is_empty

    def check_grid(self):
        if len(self.__grid) > 0 and len(self.__grid[-1]) != len(self.__grid[0]):
            raise RuntimeError('wrong number of columns in current row: {0}'.format(len(self.__grid) - 1))

    @staticmethod
    def __is_spec(x):
        return (
            isinstance(x, tuple) and
            len(x) == 3 and
            SubplotArtistGrid.__is_artist(x[0]) and
            isinstance(x[1], Integral) and
            isinstance(x[2], Integral))

    @staticmethod
    def __is_artist(x):
        return (
            hasattr(x, 'width') and
            hasattr(x, 'height') and
            hasattr(x, 'render_subplots'))

    def append_subplot_artist(self, artist):

        if SubplotArtistGrid.__is_spec(artist):
            artist, num_rows, num_columns = artist
        else:
            num_rows = 1
            num_columns = 1

        # noinspection PyTypeChecker
        if 0 <= self.__num_fixed_columns < self.current_column + num_columns:
            # noinspection PyTypeChecker
            if num_columns > self.__num_fixed_columns:
                raise ValueError('artist out of bounds for fixed number of columns')
            # noinspection PyTypeChecker
            if 0 <= self.__num_fixed_rows <= self.current_row + 1:
                raise ValueError('new row is required, but would be out of bounds')
            self.next_row()

        # noinspection PyTypeChecker
        if 0 <= self.__num_fixed_rows < self.current_row + num_rows:
            raise ValueError('artist out of bounds for fixed number of rows')

        artist_spec = None
        for i in range(self.current_row, self.current_row + num_rows):
            if i == len(self.__grid):
                self.__grid.append(list())
            if i == self.current_row:
                while self.current_column < len(self.__grid[i]) and \
                        self.__grid[i][self.current_column] is not None:
                    self.__current_column += 1
                artist_spec = _SubplotArtistSpec(
                    artist,
                    index_grid_left=self.current_column,
                    index_grid_top=self.current_row,
                    num_rows=num_rows,
                    num_columns=num_columns)
            else:
                while len(self.__grid[i]) < self.current_column:
                    self.__grid[i].append(None)
            for j in range(self.current_column, self.current_column + num_columns):
                if j == len(self.__grid[i]):
                    self.__grid[i].append(artist_spec)
                elif self.__grid[i][j] is None:
                    self.__grid[i][j] = artist_spec
                else:
                    raise Exception('bad grid')

        self.__subplot_artist_specs.append(artist_spec)
        self.__current_column += artist_spec.num_columns

    def next_row(self):

        # noinspection PyTypeChecker
        if 0 <= self.__num_fixed_rows <= self.current_row + 1:
            raise RuntimeError('Adding a row would exceed the number of fixed rows')

        if self.current_row < len(self.__grid):
            while self.current_column < len(self.__grid[self.current_row]) \
                    and self.__grid[self.current_row][self.current_column] is not None:
                self.__current_column += 1
        if len(self.__grid) > 0 and self.current_column != len(self.__grid[0]):
            raise RuntimeError('wrong number of columns in current row: {0}, expected {1}, got {2}'.format(
                self.current_row, len(self.__grid[0]), self.current_column))
        self.__current_column = 0
        self.__current_row += 1

    def fix_rows(self):
        self.__num_fixed_rows = self.row_count

    def fix_columns(self):
        self.__num_fixed_columns = self.column_count

    def __setitem__(self, key, value):

        if not isinstance(key, tuple):
            raise IndexError('key must be an (index_row, index_column) pair')
        try:
            row_key, column_key = key
        except ValueError:
            raise ValueError('key must be an (index_row, index_column) pair')

        if not SubplotArtistGrid.__is_artist(value):
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
        if 0 <= self.__num_fixed_columns < col2:
            raise IndexError('column index out of range: {0}'.format(column_key))
        # noinspection PyTypeChecker
        if 0 <= self.__num_fixed_rows < row2:
            raise IndexError('row index out of range: {0}'.format(row_key))

        if self.column_count < col2:
            for i in range(self.row_count):
                while len(self.__grid[i]) < col2:
                    self.__grid[i].append(None)

        while self.row_count < row2:
            self.__grid.append([None] * max(self.column_count, col2))

        new_spec = _SubplotArtistSpec(value, col1, row1, row2 - row1, col2 - col1)
        self.__subplot_artist_specs.append(new_spec)

        to_modify = dict()

        for index_row in range(row1, row2):
            for index_column in range(col1, col2):
                artist_spec = self.__grid[index_row][index_column]
                if artist_spec is not None:
                    to_modify[(artist_spec.index_grid_top, artist_spec.index_grid_left)] = artist_spec
                self.__grid[index_row][index_column] = new_spec

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
                        self.__grid[index_row][index_column] = None
                artist_spec.update(num_columns=new_num_columns)
            elif remove_choice == _RemoveChoice.bottom:
                new_num_rows = new_spec.index_grid_top - artist_spec.index_grid_top
                for index_row in range(artist_spec.index_grid_top + new_num_rows,
                                       artist_spec.index_grid_top + artist_spec.num_rows):
                    for index_column in range(artist_spec.index_grid_left,
                                              artist_spec.index_grid_left + artist_spec.num_columns):
                        self.__grid[index_row][index_column] = None
                artist_spec.update(num_rows=new_num_rows)
            elif remove_choice == _RemoveChoice.left:
                new_left = new_spec.index_grid_left + new_spec.num_columns
                for index_row in range(artist_spec.index_grid_top, artist_spec.index_grid_top + artist_spec.num_rows):
                    for index_column in range(artist_spec.index_grid_left, new_left):
                        self.__grid[index_row][index_column] = None
                artist_spec.update(index_grid_left=new_left,
                                   num_columns=artist_spec.index_grid_left + artist_spec.num_columns - new_left)
            elif remove_choice == _RemoveChoice.top:
                new_top = new_spec.index_grid_top + new_spec.num_rows
                for index_row in range(artist_spec.index_grid_top, new_top):
                    for index_column in range(artist_spec.index_grid_left,
                                              artist_spec.index_grid_left + artist_spec.num_columns):
                        self.__grid[index_row][index_column] = None
                artist_spec.update(index_grid_top=new_top,
                                   num_rows=artist_spec.index_grid_top + artist_spec.num_rows - new_top)
            elif remove_choice == _RemoveChoice.all:
                self.__subplot_artist_specs.remove(artist_spec)
            else:
                raise RuntimeError('Bad code - unknown remove_choice: {0}'.format(remove_choice))

    def calculate_row_heights_and_column_widths(self):
        column_widths = [0] * self.column_count
        row_heights = [0] * self.row_count
        for artist_spec in self.__subplot_artist_specs:
            if artist_spec.num_rows == 1:
                row_heights[artist_spec.index_grid_top] = max(
                    row_heights[artist_spec.index_grid_top], artist_spec.height)
            if artist_spec.num_columns == 1:
                column_widths[artist_spec.index_grid_left] = max(
                    column_widths[artist_spec.index_grid_left], artist_spec.width)

        for artist_spec in self.__subplot_artist_specs:
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
             ) for artist_spec in self.__subplot_artist_specs]

    def render(
            self,
            output_figure_path,
            is_tight_layout=False,
            tight_layout_pad=1.08,
            tight_layout_h_pad=None,
            tight_layout_w_pad=None):
        # from the way tight_layout / GridSpec work together, it seems like it might be a bad idea
        # to try to use tight_layout to adjust a nested GridSpec, so we only allow tight_layout in the
        # render function, where we are the top level grid
        row_heights, column_widths = self.calculate_row_heights_and_column_widths()
        plt = pyplot_lazy_import.pyplot
        plt.ioff()
        fig = plt.figure(figsize=(self.width, self.height))
        grid = gridspec.GridSpec(
            len(row_heights),
            len(column_widths),
            width_ratios=column_widths,
            height_ratios=row_heights,
            wspace=self.wspace,
            hspace=self.hspace)
        for artist_spec in self.__subplot_artist_specs:
            subplot_spec = grid[
                artist_spec.index_grid_top:(artist_spec.index_grid_top + artist_spec.num_rows),
                artist_spec.index_grid_left:(artist_spec.index_grid_left + artist_spec.num_columns)]
            artist_spec.subplot_artist.render_subplots(fig, subplot_spec)
        if is_tight_layout:
            grid.tight_layout(
                fig, pad=tight_layout_pad, h_pad=tight_layout_h_pad, w_pad=tight_layout_w_pad)
        if output_figure_path is None:
            # this will block until the figures are closed
            plt.show()
        else:
            fig.savefig(output_figure_path, bbox_inches='tight')
            plt.close(fig)
        del fig

    def render_subplots(self, fig, subplot_spec):
        for artist, subplot_spec_for_artist in self.prepare_subplots(subplot_spec):
            artist.render_subplots(fig, subplot_spec_for_artist)


class SubplotArtist:

    def __init__(
            self,
            width=None,
            height=None,
            subplot_artist_grid=None,
            title=None,
            x_label=None,
            y_label=None,
            bind_data=None):

        self.__subplot_artist_grid = subplot_artist_grid
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self._bind_data = bind_data

        if width is not None and not isinstance(width, Number):
            raise ValueError('width must be a number')

        if height is not None and not isinstance(height, Number):
            raise ValueError('height must be a number')

        if subplot_artist_grid is not None:
            if not isinstance(subplot_artist_grid, SubplotArtistGrid):
                raise ValueError('subplot_artist_grid must be an instance of SubplotArtistGrid')
            width = max(width, subplot_artist_grid.width) if width is not None else subplot_artist_grid.width
            height = max(height, subplot_artist_grid.height if height is not None else subplot_artist_grid.height)
        else:
            if width is None:
                raise ValueError('width must be specified if subplot_artist_grid is not specified')
            if height is None:
                raise ValueError('height must be specified if subplot_artist_grid is not specified')

        self.__width = width
        self.__height = height

    def render_subplots(self, fig, subplot_spec):
        if self.__subplot_artist_grid is not None:
            self.__subplot_artist_grid.render_subplots(fig, subplot_spec)
        else:
            axes = self._add_subplot(fig, subplot_spec)
            if self.x_label is not None:
                axes.set_xlabel(self.x_label)
            if self.y_label is not None:
                axes.set_ylabel(self.y_label)
            if self.title is not None:
                axes.set_title(self.title)
            bound_data = self._bind()
            self.render(axes, bound_data)

    def _add_subplot(self, fig, subplot_spec):
        return fig.add_subplot(subplot_spec)

    def _bind(self):
        return self.bind_data() if self.bind_data is not None else None

    def _get_bind_data(self):
        return self._bind_data

    def _set_bind_data(self, value):
        self._bind_data = value

    @property
    def bind_data(self):
        return self._get_bind_data()

    @bind_data.setter
    def bind_data(self, value):
        self._set_bind_data(value)

    @staticmethod
    def _extract_bound_data(bound_data, *keys):
        result = map(lambda key: bound_data[key] if bound_data is not None and key in bound_data else None, keys)
        if len(keys) == 1:
            result = next(result)
        return result

    @property
    def width(self):
        return self.__width

    @property
    def height(self):
        return self.__height

    @property
    def subplot_artist_grid(self):
        return self.__subplot_artist_grid

    def render(self, axes, bound_data):
        axes.set_axis_bgcolor('red')


class SubplotArtist3D(SubplotArtist):

    def __init__(
            self,
            width=None,
            height=None,
            title=None,
            x_label=None,
            y_label=None,
            z_label=None,
            init_elevation=30,
            init_azimuth=145,
            bind_data=None):
        SubplotArtist.__init__(
            self,
            width=width,
            height=height,
            title=title,
            x_label=x_label,
            y_label=y_label,
            bind_data=bind_data)

        self.z_label = z_label
        self.init_elevation = init_elevation
        self.init_azimuth = init_azimuth

    def _add_subplot(self, fig, subplot_spec):
        axes = fig.add_subplot(subplot_spec, projection='3d')
        axes.view_init(elev=self.init_elevation, azim=self.init_azimuth)
        if self.z_label is not None:
            axes.set_zlabel(self.z_label)
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