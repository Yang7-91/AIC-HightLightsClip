"""允许执行 ``python -m video_highlight.stage5_export``。"""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
