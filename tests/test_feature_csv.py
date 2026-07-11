from __future__ import annotations

from pathlib import Path

from xnerf.datasets.build_dataset import build_manifest, generate_cache_from_manifest
from xnerf.datasets.loaders import MalwareManifestDataset
from xnerf.utils.io import read_jsonl


def test_headerless_feature_csv_becomes_per_row_samples(tmp_path: Path):
    csv_dir = tmp_path / "raw" / "AndMal2020" / "static" / "benign"
    csv_dir.mkdir(parents=True)
    csv_path = csv_dir / "CCCS Ben0.csv"
    csv_path.write_text(
        "shaaaa111,1,2,3,4,5\n"
        "shabbb222,10,20,30,40,50\n",
        encoding="utf-8",
    )
    out = tmp_path / "processed" / "manifest.jsonl"

    build_manifest(tmp_path, out, make_splits=False)

    rows = read_jsonl(out)
    assert len(rows) == 2
    assert rows[0]["data_type"] == "feature_csv"
    assert rows[0]["label"] == 0
    assert rows[0]["feature_dim"] == 5

    ds = MalwareManifestDataset(out)
    item = ds[0]
    assert item["memory_trace"].shape == (512, 8)
    assert item["binary_image"].sum().item() == 0


def test_manifest_only_skips_feature_cache_then_generates_selected_cache(tmp_path: Path):
    csv_dir = tmp_path / "raw" / "AndMal2020" / "static" / "benign"
    csv_dir.mkdir(parents=True)
    csv_path = csv_dir / "features.csv"
    csv_path.write_text("sha111,1,2,3\nsha222,4,5,6\n", encoding="utf-8")
    out = tmp_path / "processed" / "manifest.jsonl"

    build_manifest(tmp_path, out, make_splits=False, manifest_only=True)

    rows = read_jsonl(out)
    assert len(rows) == 2
    assert rows[0]["feature_path"] == ""
    assert not (tmp_path / "cache").exists()

    generate_cache_from_manifest(tmp_path, out)

    rows = read_jsonl(out)
    assert rows[0]["feature_path"]
    assert Path(rows[0]["feature_path"]).exists()
    item = MalwareManifestDataset(out, require_cache=True)[0]
    assert item["memory_trace"].shape == (512, 8)


def test_split_manifest_reuses_parent_generated_cache(tmp_path: Path):
    csv_dir = tmp_path / "raw" / "AndMal2020" / "static" / "benign"
    csv_dir.mkdir(parents=True)
    csv_path = csv_dir / "features.csv"
    csv_path.write_text("sha111,1,2,3\nsha222,4,5,6\n", encoding="utf-8")
    parent = tmp_path / "processed" / "manifest_balanced.jsonl"
    split = tmp_path / "processed" / "train_manifest.jsonl"

    build_manifest(tmp_path, parent, make_splits=False, manifest_only=True)
    split.write_text(parent.read_text(encoding="utf-8"), encoding="utf-8")

    first = generate_cache_from_manifest(tmp_path, parent)
    second = generate_cache_from_manifest(tmp_path, parent)

    assert first["generated_feature_tensors"] == 2
    assert second["skipped_existing"] == 2
    assert read_jsonl(split)[0]["feature_path"] == ""
    item = MalwareManifestDataset(split, require_cache=True)[0]
    assert item["memory_trace"].shape == (512, 8)


def test_headered_feature_csv_uses_label_and_numeric_columns(tmp_path: Path):
    csv_dir = tmp_path / "raw" / "AndMal2020" / "dynamic"
    csv_dir.mkdir(parents=True)
    csv_path = csv_dir / "dynamic.csv"
    csv_path.write_text(
        "sha256,label,family,api_count,net_count,text_col\n"
        "abc123,benign,cleanfam,1,2,ignore\n"
        "def456,malicious,badfam,5,7,ignore\n",
        encoding="utf-8",
    )
    out = tmp_path / "processed" / "manifest.jsonl"

    build_manifest(tmp_path, out, make_splits=False)

    rows = read_jsonl(out)
    assert len(rows) == 2
    assert rows[0]["label"] == 0
    assert rows[0]["family"] == "cleanfam"
    assert rows[0]["feature_dim"] == 2
    assert rows[1]["label"] == 1


def test_public_labels_csv_is_metadata_not_sample(tmp_path: Path):
    raw = tmp_path / "raw" / "cape"
    raw.mkdir(parents=True)
    (raw / "public_labels.csv").write_text(
        "sha256,label,family\n"
        "abc123,malicious,trojan\n",
        encoding="utf-8",
    )
    feature_dir = tmp_path / "raw" / "AndMal2020" / "static"
    feature_dir.mkdir(parents=True)
    (feature_dir / "features.csv").write_text("abc123,1,2,3\n", encoding="utf-8")
    out = tmp_path / "processed" / "manifest.jsonl"

    build_manifest(tmp_path, out, make_splits=False)

    rows = read_jsonl(out)
    assert len(rows) == 1
    assert rows[0]["label"] == 1
    assert rows[0]["family"] == "trojan"


def test_api_name_sequence_csv_becomes_dynamic_rows(tmp_path: Path):
    csv_dir = tmp_path / "raw" / "MalBehavD-V1"
    csv_dir.mkdir(parents=True)
    csv_path = csv_dir / "MalBehavD-V1-dataset.csv"
    csv_path.write_text(
        "sha256,labels,0,1,2\n"
        "abc1234567890123,0,LdrLoadDll,NtCreateFile,RegOpenKeyExW\n"
        "def1234567890123,1,NtClose,,GetSystemInfo\n",
        encoding="utf-8",
    )
    out = tmp_path / "processed" / "manifest.jsonl"

    build_manifest(tmp_path, out, make_splits=False)

    rows = read_jsonl(out)
    assert len(rows) == 2
    assert rows[0]["data_type"] == "api_sequence_csv"
    assert rows[0]["api_call_count"] == 3
    assert rows[0]["label"] == 0
    assert rows[1]["api_call_count"] == 2


def test_numeric_api_sequence_csv_becomes_dynamic_rows(tmp_path: Path):
    csv_dir = tmp_path / "raw" / "MalwareAnalysisDatasetsAPICallSequences"
    csv_dir.mkdir(parents=True)
    csv_path = csv_dir / "dynamic_api_call_sequence_per_malware_100_0_306.csv"
    csv_path.write_text(
        "hash,t_0,t_1,t_2,malware\n"
        "071e8c3f8922e186e57548cd4c703a5d,112,274,158,1\n",
        encoding="utf-8",
    )
    out = tmp_path / "processed" / "manifest.jsonl"

    build_manifest(tmp_path, out, make_splits=False)

    rows = read_jsonl(out)
    assert len(rows) == 1
    assert rows[0]["data_type"] == "api_sequence_csv"
    assert rows[0]["api_call_count"] == 3
    assert rows[0]["label"] == 1


def test_malapi_text_sequences_use_line_labels(tmp_path: Path):
    raw_dir = tmp_path / "raw" / "MalAPI2019" / "Mal-API-2019" / "mal-api-2019"
    raw_dir.mkdir(parents=True)
    (raw_dir.parent / "labels.csv").write_text("Trojan\nBackdoor\n", encoding="utf-8")
    (raw_dir / "all_analysis_data.txt").write_text(
        "ldrloaddll ldrgetprocedureaddress ntclose\n"
        "getsysteminfo ntcreatefile\n",
        encoding="utf-8",
    )
    out = tmp_path / "processed" / "manifest.jsonl"

    build_manifest(tmp_path, out, make_splits=False)

    rows = read_jsonl(out)
    assert len(rows) == 2
    assert rows[0]["data_type"] == "api_sequence_txt"
    assert rows[0]["family"] == "Trojan"
    assert rows[1]["api_call_count"] == 2
