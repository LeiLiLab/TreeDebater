"""Reserve numeric run paths without reusing artifacts or racing other launches."""

import re
from pathlib import Path


def reserve_run_path(base_dir, suffix="log"):
    directory = Path(base_dir)
    directory.mkdir(parents=True, exist_ok=True)
    used = [
        int(match.group(1))
        for entry in directory.iterdir()
        if (match := re.match(r"^(\d+)(?:[._]|$)", entry.name))
    ]
    number = max(used, default=0) + 1
    while True:
        path = directory / f"{number}.{suffix}"
        try:
            # Reserve immediately, before lazy logging or model initialization.
            with path.open("x", encoding="utf-8"):
                pass
        except FileExistsError:
            number += 1
            continue
        return str(path)
