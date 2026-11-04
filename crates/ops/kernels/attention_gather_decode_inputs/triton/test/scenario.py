"""One decode step's inputs, as the operator's contract describes them.

Shared by the two checks beside this file: the interpreter check runs the kernel's own
Python on the CPU, the GPU check runs the compiled program on a device, and both
compare against :meth:`Scenario.reference`, which is written from the operator's
documentation rather than from either kernel.

A call arrives as a batch of `rows` decode rows, each naming a request slot; slot 0 is
the caller's padding slot, so a row whose slot is 0 is not live yet still occupies its
place. The request side of the pool holds each slot's cache length, its installed start
page per group, its next token id and its next position per axis, and each table is one
`[group, block size, window]` row. The outputs are the batch's fixed-capacity buffers,
which are wider than the call: `max_rows` rows of a `columns`-wide block table.
"""

import attention_gather_decode_inputs as kernels
import numpy as np
import torch


class Scenario:
    """One call: the request pool, the tables, and the buffers the call fills."""

    def __init__(
        self,
        rows=3,
        max_rows=4,
        columns=4,
        tables=1,
        slots=8,
        request_width=6,
        position_axes=1,
        cache=None,
        installed=None,
        window=-1,
        block_size=4,
        token_padding=2,
        position_padding=1,
        axis_stride=None,
        row_padding=1,
        table_padding=5,
        page_padding=1,
        address_padding=3,
        block=4,
        seed=0,
    ):
        rng = np.random.default_rng(seed)
        self.rows = rows
        self.max_rows = max_rows
        self.columns = columns
        self.tables = tables
        self.slots = slots
        self.request_width = request_width
        self.position_axes = position_axes
        self.block = block
        self.block_size = block_size
        self.row_block = 1 << (max_rows - 1).bit_length()

        # The rows name slots; the first row is a padding row when there are
        # two or more, which is how the caller pads a batch.
        self.pool = rng.integers(1, slots, size=rows, dtype=np.int32)
        if rows > 1:
            self.pool[0] = 0

        self.cache = (
            np.asarray(cache, dtype=np.int32)
            if cache is not None
            else np.zeros(max_rows, dtype=np.int32)
        )
        self.cache = np.concatenate(
            [self.cache, np.zeros(max_rows - self.cache.size, dtype=np.int32)]
        )[:max_rows]
        # The request side is indexed by slot, so it is the slots' state.
        self.slot_cache = rng.integers(
            0, request_width * block_size, size=slots
        )
        self.slot_cache = self.slot_cache.astype(np.int32)
        for index, slot in enumerate(self.pool.tolist()):
            if slot != 0:
                self.slot_cache[slot] = int(self.cache[index])

        # One group; a slot's installed start page is what its first page is
        # measured against.
        self.start_group_stride = slots + 1
        self.installed = rng.integers(0, 3, size=self.start_group_stride)
        self.installed = self.installed.astype(np.int32)
        if installed is not None:
            self.installed[:] = np.asarray(installed, dtype=np.int32)

        self.unit_row_stride = request_width
        self.unit_table_stride = slots * request_width + 3
        self.units = rng.integers(
            -1, 40, size=tables * self.unit_table_stride
        ).astype(np.int32)

        self.shapes = np.array(
            [[0, block_size, window]] * tables, dtype=np.int32
        )

        # The request side of the pool is one row per slot, so the strides that
        # reach a row are the caller's padding, not the row count.
        self.request_token_stride = 2 + token_padding
        self.tokens = rng.integers(
            2, 200, size=slots * self.request_token_stride
        ).astype(np.int32)

        self.request_position_stride = position_axes * 2 + position_padding
        self.axis_stride = max_rows + 2 if axis_stride is None else axis_stride
        self.request_positions = rng.integers(
            0,
            5000,
            size=slots * self.request_position_stride,
        ).astype(np.int32)

        self.position_axis_stride = self.axis_stride
        self.block_table_row_stride = columns + row_padding
        self.block_table_stride = (
            max_rows * self.block_table_row_stride + table_padding
        )
        self.start_page_stride = max_rows + page_padding
        self.write_stride = max_rows + address_padding

        # The buffers the call fills start from a value no call may produce, so a
        # cell the kernel forgets to write cannot pass by accident.
        self.initial = {
            "input_ids": -7,
            "positions": -7,
            "block_tables": -7,
            "start_pages": -7,
            "cache_lengths": -7,
            "query_lengths": -7,
            "query_offsets": -7,
            "prefix_offsets": -7,
            "slot_mapping": -7,
        }

    def buffers(self):
        """The outputs, sized as the operator declares them and unfilled."""
        return {
            "input_ids": np.full(
                self.max_rows, self.initial["input_ids"], np.int32
            ),
            "positions": np.full(
                self.position_axes * self.axis_stride,
                self.initial["positions"],
                np.int32,
            ),
            "block_tables": np.full(
                self.tables * self.block_table_stride,
                self.initial["block_tables"],
                np.int32,
            ),
            "start_pages": np.full(
                self.tables * self.start_page_stride,
                self.initial["start_pages"],
                np.int32,
            ),
            "cache_lengths": np.full(
                self.max_rows, self.initial["cache_lengths"], np.int32
            ),
            "query_lengths": np.full(
                self.max_rows, self.initial["query_lengths"], np.int32
            ),
            "query_offsets": np.full(
                self.max_rows + 1, self.initial["query_offsets"], np.int32
            ),
            "prefix_offsets": np.full(
                self.max_rows + 1, self.initial["prefix_offsets"], np.int32
            ),
            "slot_mapping": np.full(
                self.tables * self.write_stride,
                self.initial["slot_mapping"],
                np.int32,
            ),
        }

    def launch(self, device="cpu"):
        """The keyword arguments of one launch, and the outputs it fills."""

        def tensor(array):
            return torch.from_numpy(array).to(device)

        outputs = {
            name: tensor(value) for name, value in self.buffers().items()
        }
        arguments = {
            "request_pool_indices": tensor(self.pool),
            "request_unit_tables": tensor(self.units),
            "request_start_pages": tensor(self.installed),
            "table_shapes": tensor(self.shapes),
            "request_cache_lengths": tensor(self.slot_cache),
            "request_tokens": tensor(self.tokens),
            "request_positions": tensor(self.request_positions),
            "input_ids": outputs["input_ids"],
            "positions": outputs["positions"],
            "block_tables": outputs["block_tables"],
            "start_pages": outputs["start_pages"],
            "cache_lengths": outputs["cache_lengths"],
            "query_lengths": outputs["query_lengths"],
            "query_offsets": outputs["query_offsets"],
            "prefix_offsets": outputs["prefix_offsets"],
            "slot_mapping": outputs["slot_mapping"],
            "max_rows": self.max_rows,
            "columns": self.columns,
            "unit_table_stride": self.unit_table_stride,
            "request_token_stride": self.request_token_stride,
            "request_position_stride": self.request_position_stride,
            "position_axis_stride": self.position_axis_stride,
            "block_table_stride": self.block_table_stride,
            "start_page_stride": self.start_page_stride,
            "write_stride": self.write_stride,
            "UNIT_ROW_STRIDE": self.unit_row_stride,
            "REQUEST_WIDTH": self.request_width,
            "START_GROUP_STRIDE": self.start_group_stride,
            "POSITION_AXES": self.position_axes,
            "BLOCK_TABLE_ROW_STRIDE": self.block_table_row_stride,
            "ROW_BLOCK": self.row_block,
            "BLOCK": self.block,
        }
        return arguments, outputs

    def run(self, device="cpu"):
        """Launch the package's kernel over these tensors on `device`."""
        arguments, outputs = self.launch(device)
        cell_programs = -(-(self.max_rows * self.columns) // self.block)
        kernels._attention_gather_decode_inputs_kernel[
            (1 + cell_programs, self.tables)
        ](**arguments)
        return {name: value.cpu().numpy() for name, value in outputs.items()}

    def reference(self):
        """The outputs the operator's documentation says one call produces."""
        live = self.pool != 0
        cache = np.where(live, self.slot_cache[self.pool], 0).astype(np.int64)
        installed = np.where(live, self.installed[self.pool], 0).astype(
            np.int64
        )
        # The capacity's rows past the call's carry no cached tokens, which is what
        # makes the offsets over the whole capacity the call's offsets.
        capacity_cache = np.zeros(self.max_rows, dtype=np.int64)
        capacity_cache[: self.rows] = cache
        outputs = {name: value.copy() for name, value in self.buffers().items()}

        for table in range(self.tables):
            group, block_size, window = self.shapes[table].tolist()
            units = self.units[
                table * self.unit_table_stride : (table + 1)
                * self.unit_table_stride
            ]
            if window >= 0:
                first = np.maximum(cache - window, 0) // block_size
            else:
                first = np.zeros_like(cache)
            base = table * self.start_page_stride
            outputs["start_pages"][base : base + self.rows] = np.where(
                live, first, 0
            )

            column = cache // block_size - installed
            appendable = live & (column >= 0) & (column < self.request_width)
            unit = units[
                self.pool * self.unit_row_stride
                + np.clip(column, 0, self.request_width - 1)
            ].astype(np.int64)
            base = table * self.write_stride
            outputs["slot_mapping"][base : base + self.rows] = np.where(
                appendable & (unit >= 0),
                unit * block_size + cache % block_size,
                -1,
            )

            cells = np.arange(self.columns)
            source = (first - installed)[:, None] + cells[None, :]
            selected = (
                live[:, None] & (source >= 0) & (source < self.request_width)
            )
            page = units[
                self.pool[:, None] * self.unit_row_stride
                + np.clip(source, 0, self.request_width - 1)
            ]
            base = table * self.block_table_stride
            for row in range(self.rows):
                start = base + row * self.block_table_row_stride
                outputs["block_tables"][start : start + self.columns] = (
                    np.where(selected[row], page[row], 0)
                )

        outputs["input_ids"][: self.rows] = np.where(
            live, self.tokens[self.pool * self.request_token_stride], 1
        )
        for axis in range(self.position_axes):
            base = axis * self.position_axis_stride
            if axis == 0:
                outputs["positions"][base : base + self.rows] = np.where(
                    live,
                    self.request_positions[
                        self.pool * self.request_position_stride
                    ],
                    0,
                )
            else:
                outputs["positions"][base : base + self.rows] = 0
        outputs["cache_lengths"][: self.rows] = cache
        outputs["query_lengths"][: self.rows] = 1

        # Exclusive offsets over the full capacity of rows: every row has one
        # query token and a padded row adds no cached tokens.
        outputs["query_offsets"][:] = np.arange(self.max_rows + 1)
        outputs["prefix_offsets"][0] = 0
        outputs["prefix_offsets"][1:] = np.cumsum(capacity_cache)
        return outputs

    def written(self, name, size):
        """The cells of `name` one call must have written."""
        mask = np.zeros(size, dtype=bool)
        if name == "positions":
            for axis in range(self.position_axes):
                base = axis * self.position_axis_stride
                mask[base : base + self.rows] = True
        elif name == "block_tables":
            for table in range(self.tables):
                base = table * self.block_table_stride
                for row in range(self.rows):
                    start = base + row * self.block_table_row_stride
                    mask[start : start + self.columns] = True
        elif name == "start_pages":
            for table in range(self.tables):
                base = table * self.start_page_stride
                mask[base : base + self.rows] = True
        elif name == "slot_mapping":
            for table in range(self.tables):
                base = table * self.write_stride
                mask[base : base + self.rows] = True
        elif name in ("input_ids", "cache_lengths", "query_lengths"):
            mask[: self.rows] = True
        elif name in ("query_offsets", "prefix_offsets"):
            mask[: self.max_rows + 1] = True
        return mask
