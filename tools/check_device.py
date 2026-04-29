from __future__ import annotations

import json

from _bootstrap import add_src_to_path

add_src_to_path()

from vfi.device import describe_torch_devices


def main() -> None:
    print(json.dumps(describe_torch_devices(), indent=2))


if __name__ == "__main__":
    main()

