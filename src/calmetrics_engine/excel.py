"""On-demand Excel export. Mathematics, planning and references are native.

Importing calmetrics_engine does not import this optional writer or XlsxWriter.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path

from ._native.excel import ExcelExportError, ExcelPlan, coverage, plan, plan_operator

__all__ = ["ExcelExportError", "ExcelPlan", "coverage", "plan", "plan_operator", "export"]


def export(export_plan: ExcelPlan, destination: str | os.PathLike) -> dict:
    """Atomically serialize a frozen native plan; never launch Excel implicitly.

    The optional dependency is installed with ``pip install calmetrics-engine[excel]``.
    Returned verification remains NOT_RECALCULATED until a separate real-engine test.
    """
    import xlsxwriter

    destination = Path(destination).absolute()
    if destination.suffix.lower() != ".xlsx":
        raise ValueError("destination must end in .xlsx")
    metadata = export_plan.metadata()
    started = time.monotonic()
    def checkpoint():
        export_plan.check_cancelled()
        if time.monotonic() - started > metadata["timeout_seconds"]:
            raise ExcelExportError("TIMEOUT: Excel export")
    checkpoint()
    references = export_plan.reference()
    checkpoint()
    # Both the stream and incomplete ZIP are private, adjacent temporary files.
    # No output replaces the destination until the workbook has closed fully.
    with tempfile.TemporaryDirectory(prefix=".calmetrics-excel-", dir=destination.parent) as temp:
        stream = Path(temp) / "cells.jsonl"
        artifact = Path(temp) / "workbook.xlsx"
        export_plan.write_cells(str(stream))
        with xlsxwriter.Workbook(artifact, {
            "constant_memory": True,
            "strings_to_formulas": False,
            "strings_to_urls": False,
            "tmpdir": temp,
        }) as workbook:
            workbook.set_calc_mode("auto")
            workbook.set_properties({"title": "CalMetricsEngine formula reproduction",
                                     "comments": metadata["identity"]})
            header = workbook.add_format({"bold": True, "font_color": "#17365D"})
            numeric = workbook.add_format({"num_format": "0.###############"})
            sheets = {name: workbook.add_worksheet(name) for name in metadata["sheets"]}
            for name, sheet in sheets.items():
                sheet.set_column(0, 0, 38, header)
                sheet.set_column(1, 4, 25, numeric)
                sheet.freeze_panes(0, 1)
                if name == "Readme":
                    sheet.set_column(0, 0, 72)
                elif name.startswith("Nodes"):
                    sheet.set_column(2, 6, 28)
            count = 0
            with stream.open(encoding="utf-8") as source:
                for line in source:
                    if count % 4096 == 0:
                        checkpoint()
                        if sum(entry.stat().st_size for entry in Path(temp).iterdir() if entry.is_file()) > 2 * metadata["max_stream_bytes"]:
                            raise ExcelExportError("OVER_BUDGET: temporary files")
                    name, row, column, kind, value = json.loads(line)
                    sheet = sheets[name]
                    if kind == "reference":
                        root, index = map(int, value.split(":"))
                        reference = references[root]
                        if reference.get("statuses") and reference["statuses"][index]:
                            kind, value = "text", "ERROR:STATUS:" + str(reference["statuses"][index])
                        elif reference["error"]:
                            kind, value = "text", "ERROR:" + reference["error"]
                        else:
                            value = reference["values"][index]
                            dtype = metadata["outputs"][root]["dtype"]
                            if dtype == "bool":
                                kind = "boolean"
                            elif isinstance(value, float) and not math.isfinite(value):
                                kind, value = "text", ("NaN" if math.isnan(value) else "+Inf" if value > 0 else "-Inf")
                            else:
                                kind = "number"
                    if kind == "formula":
                        # An explicit marker prevents a cached C++ value from
                        # masquerading as an actual spreadsheet calculation.
                        sheet.write_formula(row, column, "=" + value, None, "NOT_RECALCULATED")
                    elif kind == "number":
                        sheet.write_number(row, column, float(value))
                    elif kind == "boolean":
                        sheet.write_boolean(row, column, bool(int(value)))
                    else:
                        sheet.write_string(row, column, str(value))
                    count += 1
            if count != metadata["cells"]:
                raise ExcelExportError("EXPORT_PLAN_MISMATCH: written cells")
        checkpoint()
        if artifact.stat().st_size > metadata["max_stream_bytes"]:
            raise ExcelExportError("OVER_BUDGET: workbook file bytes")
        os.replace(artifact, destination)
    return {**metadata, "path": str(destination), "bytes": destination.stat().st_size}
