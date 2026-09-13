from pathlib import Path

from batch_process import batch


def test_batch_directories_are_relative_to_batch_module():
    base_dir = Path(batch.__file__).resolve().parent

    assert Path(batch.UPLOAD_FOLDER) == base_dir / "upload"
    assert Path(batch.PROCESSED_FOLDER) == base_dir / "processed"
    assert Path(batch.FAILED_FOLDER) == base_dir / "failed"
