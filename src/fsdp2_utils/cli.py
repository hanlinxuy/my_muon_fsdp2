"""FSDP2 工具库 - 命令行接口。"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    """主 CLI 入口点。"""
    parser = argparse.ArgumentParser(
        description="FSDP2 工具库 - 通用 PyTorch FSDP2 工具",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version="%(prog)s 0.2.0",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="显示包信息",
    )

    args = parser.parse_args()

    if args.info:
        print("FSDP2 工具库")
        print("版本: 0.2.0")
        print("描述: 通用 PyTorch FSDP2 工具库")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
