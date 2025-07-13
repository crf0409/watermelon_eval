from pathlib import Path

from toolbox.data.io import Record, scan_raw, to_imf


def test_scan(tmp_path):
    fold = tmp_path / "fold_0"
    fold.mkdir()
    (fold / "audio.wav").write_text("abc")
    (fold / "image.jpg").write_text("img")
    (fold / "meta.json").write_text('{"id":"0","brix":1.0}')
    records = scan_raw(tmp_path)
    assert len(records) == 1
    assert records[0].brix == 1.0
