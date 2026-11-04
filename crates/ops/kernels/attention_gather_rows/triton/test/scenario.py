"""One call's attention metadata, as the gather operator's contract describes it.

Shared by the two checks beside this file: the interpreter check runs the kernel's own
Python on the CPU, the GPU check runs the compiled program on a device, and both
compare against :meth:`Scenario.reference`, which is written from the operator's
documentation rather than from either kernel.

Every scenario is built the way a call arrives: an int64-free int32 column array of
`ROW_SECTIONS` shared sections plus one per table and then a row per token, the
slots' resident unit tables, each slot's installed start page, and one table shape
per table. What varies between scenarios is what can break the program: a storage
that is not contiguous, slots that installed fewer units than the request width, a
windowed table, and a token section that ends exactly at a row's edge.
"""

import attention_gather_rows as kernels
import numpy as np
import torch

#: The shared sections of the host column array, as the operator numbers them.
ROW_SECTIONS = 6


class Scenario:
    """One call's tensors, laid out the way the operator's contract describes."""

    def __init__(
        self,
        *,
        rows,
        tokens,
        tables,
        width,
        block,
        slots=8,
        groups=1,
        request_width=8,
        block_size=2,
        window=-1,
        row_padding=0,
        table_padding=0,
        token_padding=1,
        seed=0,
    ):
        generator = np.random.default_rng(seed)
        self.rows = rows
        self.tokens = tokens
        self.tables = tables
        self.width = width
        self.block = block
        self.block_size = block_size
        self.window = window
        self.request_width = request_width
        self.groups = groups
        self.slots = slots

        # The columns: one section per shared column of `max_rows + 1` elements, one per
        # table of the row's selected page count, then a row per token.
        self.max_rows = rows + 2
        self.row_stride = self.max_rows + 1
        self.token_section = (ROW_SECTIONS + tables) * self.row_stride
        size = self.token_section + tokens + token_padding
        self.columns = np.zeros(size, dtype=np.int32)

        self.slot_of = generator.integers(1, slots, size=rows).astype(np.int32)
        self.prefix = (generator.integers(0, 8, size=rows) * block_size).astype(
            np.int32
        )
        self.query = generator.integers(1, 4, size=rows).astype(np.int32)
        self.writes = generator.integers(0, 2, size=rows).astype(np.int32)
        self.query_offsets = np.concatenate(
            [[0], np.cumsum(self.query)]
        ).astype(np.int32)
        self.query_offsets = self.query_offsets[: rows + 1]
        self.prefix_offsets = np.concatenate(
            [[0], np.cumsum(self.prefix)]
        ).astype(np.int32)
        self.prefix_offsets = self.prefix_offsets[: rows + 1]
        # The tokens are the rows' queries in order, which is what makes `token -
        # start` a position inside the row.
        token_rows = []
        for row in range(rows):
            for _ in range(int(self.query[row])):
                if len(token_rows) == tokens:
                    break
                token_rows.append(row)
        while len(token_rows) < tokens:
            token_rows.append(rows - 1)
        self.token_rows = np.array(token_rows[:tokens], dtype=np.int32)

        def section(index):
            start = index * self.row_stride
            return self.columns[start : start + self.row_stride]

        section(0)[:rows] = self.slot_of
        section(1)[:rows] = self.prefix
        section(2)[:rows] = self.query
        section(3)[:rows] = self.writes
        section(4)[: rows + 1] = self.query_offsets
        section(5)[: rows + 1] = self.prefix_offsets

        # A row selects as many pages as its position over the block size needs; the
        # first row deliberately selects fewer than the width, so the zero padding past
        # a row's pages is exercised.
        self.selected = generator.integers(
            0, width + 1, size=(tables, rows)
        ).astype(np.int32)
        for table in range(tables):
            section(ROW_SECTIONS + table)[:rows] = self.selected[table]
        self.columns[self.token_section : self.token_section + tokens] = (
            self.token_rows
        )

        # The resident tables: one per table, one row per slot.
        self.unit_table_stride = slots * request_width + 3
        self.unit_row_stride = request_width
        self.units = generator.integers(
            -1, 4, size=tables * self.unit_table_stride
        ).astype(np.int32)
        self.start_group_stride = slots + 1
        self.start_pages_request = generator.integers(
            0, 3, size=groups * self.start_group_stride
        ).astype(np.int32)

        self.table_shapes = np.zeros((tables, 3), dtype=np.int32)
        for table in range(tables):
            self.table_shapes[table] = (table % groups, block_size, window)

        # The outputs: rows shorter than the storage's rows and tables further apart
        # than the rows need, which is what "not contiguous" means here.
        self.block_table_row_stride = width + row_padding
        self.block_table_stride = (
            rows * self.block_table_row_stride + table_padding
        )
        self.block_tables = np.full(
            tables * self.block_table_stride, -7, dtype=np.int32
        )
        self.start_page_stride = rows + 1
        self.start_pages = np.full(
            tables * self.start_page_stride, -7, dtype=np.int32
        )
        self.write_stride = tokens + 2
        self.slot_mapping = np.full(
            tables * self.write_stride, -7, dtype=np.int32
        )
        self.cache_lengths = np.full(rows, -7, dtype=np.int32)
        self.query_lengths = np.full(rows, -7, dtype=np.int32)
        self.query_offsets_out = np.full(rows + 1, -7, dtype=np.int32)
        self.prefix_offsets_out = np.full(rows + 1, -7, dtype=np.int32)

        self.cell_programs = (rows * width + block - 1) // block
        self.row_block = 1 << (self.row_stride - 1).bit_length()

    def reference(self):
        """The operator's outputs, computed plainly from its documentation."""
        cache_lengths = np.zeros(self.rows, dtype=np.int32)
        query_lengths = np.zeros(self.rows, dtype=np.int32)
        query_offsets = np.zeros(self.rows + 1, dtype=np.int32)
        prefix_offsets = np.zeros(self.rows + 1, dtype=np.int32)
        start_pages = np.zeros(
            self.tables * self.start_page_stride, dtype=np.int32
        )
        block_tables = np.zeros(
            self.tables * self.block_table_stride, dtype=np.int32
        )
        slot_mapping = np.zeros(self.tables * self.write_stride, dtype=np.int32)

        for table in range(self.tables):
            group, block_size, window = (
                int(value) for value in self.table_shapes[table]
            )
            for row in range(self.rows):
                prefix = int(self.prefix[row])
                first = (
                    max(prefix - window, 0) // block_size if window >= 0 else 0
                )
                start_pages[table * self.start_page_stride + row] = first
                slot = int(self.slot_of[row])
                installed = int(
                    self.start_pages_request[
                        group * self.start_group_stride + slot
                    ]
                )
                pages = int(self.selected[table][row])
                for cell in range(self.width):
                    source = first - installed + cell
                    value = 0
                    if cell < pages and 0 <= source < self.request_width:
                        value = int(
                            self.units[
                                table * self.unit_table_stride
                                + slot * self.unit_row_stride
                                + source
                            ]
                        )
                    block_tables[
                        table * self.block_table_stride
                        + row * self.block_table_row_stride
                        + cell
                    ] = value
            for token in range(self.tokens):
                row = int(self.token_rows[token])
                slot = int(self.slot_of[row])
                prefix = int(self.prefix[row])
                start = int(self.query_offsets[row])
                writable = int(self.writes[row])
                installed = int(
                    self.start_pages_request[
                        group * self.start_group_stride + slot
                    ]
                )
                position = prefix + token - start
                column = position // block_size - installed
                unit = -1
                if writable and 0 <= column < self.request_width:
                    unit = int(
                        self.units[
                            table * self.unit_table_stride
                            + slot * self.unit_row_stride
                            + column
                        ]
                    )
                address = -1
                if writable and unit >= 0:
                    address = unit * block_size + position % block_size
                slot_mapping[table * self.write_stride + token] = address
            if table == 0:
                for row in range(self.rows):
                    cache_lengths[row] = int(self.prefix[row])
                    query_lengths[row] = int(self.query[row])
                for row in range(self.rows + 1):
                    query_offsets[row] = int(self.query_offsets[row])
                    prefix_offsets[row] = int(self.prefix_offsets[row])
        return {
            "cache_lengths": cache_lengths,
            "query_lengths": query_lengths,
            "query_offsets": query_offsets,
            "prefix_offsets": prefix_offsets,
            "start_pages": start_pages,
            "block_tables": block_tables,
            "slot_mapping": slot_mapping,
        }

    def run(self, device="cpu"):
        """Launch the package's kernel over these tensors on `device`."""

        def as_tensor(array):
            return torch.from_numpy(array).to(device)

        outputs = {
            "block_tables": as_tensor(self.block_tables.copy()),
            "start_pages": as_tensor(self.start_pages.copy()),
            "cache_lengths": as_tensor(self.cache_lengths.copy()),
            "query_lengths": as_tensor(self.query_lengths.copy()),
            "query_offsets": as_tensor(self.query_offsets_out.copy()),
            "prefix_offsets": as_tensor(self.prefix_offsets_out.copy()),
            "slot_mapping": as_tensor(self.slot_mapping.copy()),
        }
        grid = (
            1
            + self.cell_programs
            + (self.tokens + self.block - 1) // self.block,
            self.tables,
            1,
        )
        kernels._attention_gather_rows_kernel[grid](
            as_tensor(self.columns.copy()),
            as_tensor(self.units.copy()),
            as_tensor(self.start_pages_request.copy()),
            as_tensor(self.table_shapes.copy()),
            outputs["block_tables"],
            outputs["start_pages"],
            outputs["cache_lengths"],
            outputs["query_lengths"],
            outputs["query_offsets"],
            outputs["prefix_offsets"],
            outputs["slot_mapping"],
            self.rows,
            self.tokens,
            self.cell_programs,
            self.width,
            ROW_STRIDE=self.row_stride,
            TOKEN_SECTION=self.token_section,
            UNIT_TABLE_STRIDE=self.unit_table_stride,
            UNIT_ROW_STRIDE=self.unit_row_stride,
            REQUEST_WIDTH=self.request_width,
            START_GROUP_STRIDE=self.start_group_stride,
            BLOCK_TABLE_STRIDE=self.block_table_stride,
            BLOCK_TABLE_ROW_STRIDE=self.block_table_row_stride,
            START_PAGE_STRIDE=self.start_page_stride,
            WRITE_STRIDE=self.write_stride,
            ROW_BLOCK=self.row_block,
            BLOCK=self.block,
        )
        return {name: value.cpu().numpy() for name, value in outputs.items()}

    def written(self, name, size):
        """The index mask of the cells `name` must contain after one call."""
        mask = np.zeros(size, dtype=bool)
        for table in range(self.tables):
            if name == "start_pages":
                base = table * self.start_page_stride
                mask[base : base + self.rows] = True
            elif name == "block_tables":
                base = table * self.block_table_stride
                for row in range(self.rows):
                    start = base + row * self.block_table_row_stride
                    mask[start : start + self.width] = True
            elif name == "slot_mapping":
                base = table * self.write_stride
                mask[base : base + self.tokens] = True
            elif name in ("cache_lengths", "query_lengths"):
                if table == 0:
                    mask[: self.rows] = True
            elif name in ("query_offsets", "prefix_offsets"):
                if table == 0:
                    mask[: self.rows + 1] = True
        return mask

    def untouched(self):
        """The output cells this call must leave alone: the rows past the call's."""
        missed = []
        for table in range(self.tables):
            for row in range(
                self.rows, min(self.max_rows, self.start_page_stride)
            ):
                offset = table * self.start_page_stride + row
                if self.start_pages[offset] != -7:
                    missed.append(f"start_pages beyond the rows at {offset}")
        return missed
