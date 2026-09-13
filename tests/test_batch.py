from pathlib import Path
from unittest.mock import Mock

from batch_process import batch


def test_batch_directories_are_relative_to_batch_module():
    base_dir = Path(batch.__file__).resolve().parent

    assert Path(batch.UPLOAD_FOLDER) == base_dir / "upload"
    assert Path(batch.PROCESSED_FOLDER) == base_dir / "processed"
    assert Path(batch.FAILED_FOLDER) == base_dir / "failed"


def test_process_file_calls_api_and_uses_idempotent_insert(tmp_path, monkeypatch):
    csv_file = tmp_path / "returns.csv"
    csv_file.write_text("label,pixel1,pixel2\n0,0,255\n")

    response = Mock()
    response.json.return_value = {
        "prediction": 0,
        "label": "T-shirt/top",
        "confidence": 0.76,
    }
    response.raise_for_status.return_value = None
    post = Mock(return_value=response)
    monkeypatch.setattr(batch.requests, "post", post)

    cursor = Mock()
    connection = Mock()
    connection.cursor.return_value = cursor

    batch.process_file(csv_file, connection)

    post.assert_called_once_with(
        batch.API_URL, json={"pixels": [0, 255]}, timeout=30
    )
    sql = cursor.execute.call_args.args[0]
    assert "ON CONFLICT (source_file, row_index) DO NOTHING" in sql
    connection.commit.assert_called_once()
    cursor.close.assert_called_once()


def test_run_batch_moves_failed_file(tmp_path, monkeypatch):
    upload = tmp_path / "upload"
    processed = tmp_path / "processed"
    failed = tmp_path / "failed"
    upload.mkdir()
    source = upload / "broken.csv"
    source.write_text("pixel1\n0\n")

    monkeypatch.setattr(batch, "UPLOAD_FOLDER", str(upload))
    monkeypatch.setattr(batch, "PROCESSED_FOLDER", str(processed))
    monkeypatch.setattr(batch, "FAILED_FOLDER", str(failed))

    connection = Mock()
    monkeypatch.setattr(batch, "get_db", Mock(return_value=connection))
    monkeypatch.setattr(
        batch, "process_file", Mock(side_effect=RuntimeError("API unavailable"))
    )

    batch.run_batch()

    assert not source.exists()
    assert (failed / "broken.csv").exists()
    connection.rollback.assert_called_once()
    connection.close.assert_called_once()
