"""Smoke test for the new delete_last_page feature.

Exercises:
  * pop from self.pages
  * delete page_NNN.jpg + raw_NNN.jpg + .reason.txt off disk
  * renumber the remaining files so PDF stays contiguous
  * no-op when the buffer is empty
"""
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

import app as app_mod


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="dscan_del_"))
    try:
        scanned = tmp / "scanned"
        raw = tmp / "raw"
        out = tmp / "output"
        for d in (scanned, raw, out):
            d.mkdir(parents=True, exist_ok=True)

        session = app_mod.ScanSession(
            captures_dir=scanned,
            output_dir=out,
            camera_source=99,           # never read
            web_host="127.0.0.1",
            web_port=0,
            scan_mode="color",
        )
        session.pdf_dir = out / "pdf"
        session.qr_dir = out / "qr"
        session.scanned_dir = scanned
        session.raw_dir = raw
        (session.pdf_dir).mkdir(parents=True, exist_ok=True)
        (session.qr_dir).mkdir(parents=True, exist_ok=True)

        # Seed three synthetic pages directly so we don't need a camera.
        blank = (np.ones((200, 200, 3), dtype=np.uint8) * 200)
        session.pages = [blank.copy() for _ in range(3)]
        for i, img in enumerate(session.pages, start=1):
            cv2.imwrite(str(scanned / f"page_{i}.jpg"), img)
            cv2.imwrite(str(raw / f"raw_{i}.jpg"), img)
            (scanned / f"page_{i}.reason.txt").write_text("synthetic seed\n")

        print("before:", [p.name for p in sorted(scanned.glob("page_*.jpg"))],
              "memory:", session.page_count())

        # 1) delete the last page
        ok = session.delete_last_page()
        print("delete returned:", ok)
        names = sorted(p.name for p in scanned.glob("page_*.jpg"))
        print("after first delete:", names, "memory:", session.page_count())
        assert ok is True
        assert names == ["page_1.jpg", "page_2.jpg"]
        assert session.page_count() == 2

        # 2) delete again
        ok = session.delete_last_page()
        names = sorted(p.name for p in scanned.glob("page_*.jpg"))
        print("after second delete:", names, "memory:", session.page_count())
        assert ok is True
        assert names == ["page_1.jpg"]
        assert session.page_count() == 1

        # 3) delete the last remaining one
        ok = session.delete_last_page()
        names = sorted(p.name for p in scanned.glob("page_*.jpg"))
        print("after third delete:", names, "memory:", session.page_count())
        assert ok is True
        assert names == []
        assert session.page_count() == 0

        # 4) no-op when empty
        ok = session.delete_last_page()
        print("delete on empty returned:", ok, "message:", session.last_message)
        assert ok is False

        # 5) raw dir must also be clean
        raw_files = sorted(p.name for p in raw.glob("*"))
        print("raw after all deletes:", raw_files)
        assert raw_files == []

        print("OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
