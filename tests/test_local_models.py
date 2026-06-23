"""Tests for local model dirs, path validation, and LoRA CLI translation."""

import sys
from pathlib import Path
from unittest import mock

import pytest
from fastapi import HTTPException

# Reuse the import dance from test_failure_classification (both the client and
# the backend ship a regular package named `app`).
from tests.test_failure_classification import client_main


@pytest.fixture
def model_dirs(tmp_path):
    """Two allowed dirs with a model and an adapter inside."""
    models = tmp_path / "models"
    loras = tmp_path / "loras"
    (models / "my-ckpt").mkdir(parents=True)
    (loras / "my-adapter").mkdir(parents=True)
    with mock.patch.object(
        client_main.settings, "model_dirs", f"{models},{loras}"
    ):
        yield {"models": models, "loras": loras}


class TestValidateHostPath:
    def test_valid_path(self, model_dirs):
        path = str(model_dirs["models"] / "my-ckpt")
        assert client_main._validate_host_path(path) == path

    def test_outside_allowlist_rejected(self, model_dirs, tmp_path):
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        with pytest.raises(HTTPException) as excinfo:
            client_main._validate_host_path(str(outside))
        assert excinfo.value.status_code == 400
        assert "allowed model" in excinfo.value.detail

    def test_traversal_escape_rejected(self, model_dirs, tmp_path):
        sneaky = str(model_dirs["models"] / ".." / "..")
        with pytest.raises(HTTPException):
            client_main._validate_host_path(sneaky)

    def test_missing_path_rejected(self, model_dirs):
        with pytest.raises(HTTPException) as excinfo:
            client_main._validate_host_path(str(model_dirs["models"] / "nope"))
        assert "does not exist" in excinfo.value.detail

    def test_no_dirs_configured_still_rejects_outside_paths(self):
        # The managed .models dir is always allowed, so the allowlist is never
        # empty — but arbitrary paths outside it are still rejected.
        with mock.patch.object(client_main.settings, "model_dirs", ""):
            with pytest.raises(HTTPException) as excinfo:
                client_main._validate_host_path("/anything")
        assert excinfo.value.status_code == 400
        assert "allowed model" in excinfo.value.detail


class TestLoraArgsToCli:
    def test_empty(self):
        assert client_main._lora_args_to_cli(None) == []
        assert client_main._lora_args_to_cli([]) == []

    def test_hub_adapter_passthrough(self):
        tokens = client_main._lora_args_to_cli(
            [{"name": "a", "path": "org/lora-adapter"}]
        )
        assert tokens == ["--enable-lora", "--lora-modules", "a=org/lora-adapter"]

    def test_local_path_validated_and_used(self, model_dirs):
        path = str(model_dirs["loras"] / "my-adapter")
        tokens = client_main._lora_args_to_cli([{"name": "a", "path": path}])
        assert tokens == ["--enable-lora", "--lora-modules", f"a={path}"]

    def test_local_path_outside_allowlist_rejected(self, model_dirs, tmp_path):
        outside = tmp_path / "rogue"
        outside.mkdir()
        with pytest.raises(HTTPException):
            client_main._lora_args_to_cli([{"name": "a", "path": str(outside)}])

    def test_duplicate_names_rejected(self):
        with pytest.raises(HTTPException) as excinfo:
            client_main._lora_args_to_cli(
                [{"name": "a", "path": "x/y"}, {"name": "a", "path": "x/z"}]
            )
        assert "Duplicate" in excinfo.value.detail

    def test_missing_fields_rejected(self):
        with pytest.raises(HTTPException):
            client_main._lora_args_to_cli([{"name": "", "path": "x/y"}])


@pytest.fixture
def managed_models_dir(tmp_path):
    """Point the managed .models dir (and its staging area) at a tmp path."""
    models = tmp_path / ".models"
    with mock.patch.object(client_main, "_MODELS_DIR", models), mock.patch.object(
        client_main, "_MODELS_TMP", models / ".tmp"
    ):
        yield models


class TestSanitizeModelName:
    @pytest.mark.parametrize("name", ["my-model", "Llama3.1_ft", "a", "x" * 64])
    def test_valid(self, name):
        assert client_main._sanitize_model_name(f" {name} ") == name

    @pytest.mark.parametrize(
        "name",
        ["", "a/b", "../up", ".hidden", ".tmp", "x" * 65, "sp ace", "semi;colon"],
    )
    def test_invalid(self, name):
        with pytest.raises(HTTPException) as excinfo:
            client_main._sanitize_model_name(name)
        assert excinfo.value.status_code == 400


class TestValidateRelPath:
    @pytest.mark.parametrize("rel", ["config.json", "sub/dir/w.safetensors"])
    def test_valid(self, rel):
        assert str(client_main._validate_rel_path(rel)) == rel

    @pytest.mark.parametrize(
        "rel", ["", "/abs/path", "../escape", "a/../b", "c:\\win"]
    )
    def test_invalid(self, rel):
        with pytest.raises(HTTPException) as excinfo:
            client_main._validate_rel_path(rel)
        assert excinfo.value.status_code == 400

    @pytest.mark.parametrize("rel", ["a//b", "a/./b"])
    def test_redundant_separators_normalize_safely(self, rel):
        # PurePosixPath collapses empty and '.' segments, so these are safe.
        assert str(client_main._validate_rel_path(rel)) == "a/b"


class TestSafeExtract:
    def _make_tar(self, tmp_path, member_name, data=b"x"):
        import io
        import tarfile

        archive = tmp_path / "a.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            info = tarfile.TarInfo(member_name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        return archive

    def test_benign_tar_extracts(self, tmp_path):
        archive = self._make_tar(tmp_path, "model/config.json", b"{}")
        dest = tmp_path / "out"
        dest.mkdir()
        client_main._safe_extract(archive, dest)
        assert (dest / "model" / "config.json").read_bytes() == b"{}"

    def test_tar_traversal_rejected(self, tmp_path):
        archive = self._make_tar(tmp_path, "../evil.txt")
        dest = tmp_path / "out"
        dest.mkdir()
        with pytest.raises(HTTPException) as excinfo:
            client_main._safe_extract(archive, dest)
        assert excinfo.value.status_code == 400
        assert not (tmp_path / "evil.txt").exists()

    def test_tar_symlink_member_skipped(self, tmp_path):
        import tarfile

        archive = tmp_path / "a.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            info = tarfile.TarInfo("link")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tar.addfile(info)
        dest = tmp_path / "out"
        dest.mkdir()
        client_main._safe_extract(archive, dest)
        assert not (dest / "link").exists()

    def test_zip_traversal_rejected(self, tmp_path):
        import zipfile

        archive = tmp_path / "a.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("../evil.txt", "x")
        dest = tmp_path / "out"
        dest.mkdir()
        with pytest.raises(HTTPException):
            client_main._safe_extract(archive, dest)

    def test_zip_extracts(self, tmp_path):
        import zipfile

        archive = tmp_path / "a.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("model/config.json", "{}")
        dest = tmp_path / "out"
        dest.mkdir()
        client_main._safe_extract(archive, dest)
        assert (dest / "model" / "config.json").read_text() == "{}"


class TestManagedDirIntegration:
    def test_allowed_model_dirs_includes_managed_first(self, managed_models_dir):
        with mock.patch.object(client_main.settings, "model_dirs", ""):
            dirs = client_main._allowed_model_dirs()
        assert dirs[0] == managed_models_dir.resolve()

    def test_validate_host_path_accepts_managed_model(self, managed_models_dir):
        model = managed_models_dir / "my-ft"
        model.mkdir(parents=True)
        with mock.patch.object(client_main.settings, "model_dirs", ""):
            assert client_main._validate_host_path(str(model)) == str(model.resolve())

    def test_volumes_mount_managed_dir_ro(self, managed_models_dir):
        with mock.patch.object(client_main.settings, "model_dirs", ""):
            volumes = client_main._volumes()
        key = str(managed_models_dir.resolve())
        assert volumes[key] == {"bind": key, "mode": "ro"}


class TestFinalizeModel:
    def test_rename_and_flatten(self, managed_models_dir):
        staging = managed_models_dir / ".tmp" / ".partial-x"
        (staging / "root-dir").mkdir(parents=True)
        (staging / "root-dir" / "config.json").write_text("{}")
        result = client_main._finalize_model(staging, "my-ft", flatten=True)
        target = managed_models_dir / "my-ft"
        assert (target / "config.json").exists()
        assert result["path"] == str(target)
        assert result["warnings"] == []
        assert not staging.exists()

    def test_warning_without_config_or_weights(self, managed_models_dir):
        staging = managed_models_dir / ".tmp" / ".partial-y"
        staging.mkdir(parents=True)
        (staging / "notes.txt").write_text("hi")
        result = client_main._finalize_model(staging, "odd", flatten=False)
        assert result["warnings"]

    def test_conflict_409_cleans_staging(self, managed_models_dir):
        (managed_models_dir / "taken").mkdir(parents=True)
        staging = managed_models_dir / ".tmp" / ".partial-z"
        staging.mkdir(parents=True)
        with pytest.raises(HTTPException) as excinfo:
            client_main._finalize_model(staging, "taken", flatten=False)
        assert excinfo.value.status_code == 409
        assert not staging.exists()


class TestVolumes:
    def test_allowed_dirs_mounted_ro_same_path(self, model_dirs):
        volumes = client_main._volumes()
        models = str(model_dirs["models"])
        assert volumes[models] == {"bind": models, "mode": "ro"}

    def test_image_digest_fallback(self):
        image = mock.MagicMock()
        image.attrs = {"RepoDigests": []}
        image.id = "sha256:abc"
        container = mock.MagicMock(image=image)
        assert client_main._image_digest(container) == "sha256:abc"

    def test_image_digest_repo_digest(self):
        image = mock.MagicMock()
        image.attrs = {"RepoDigests": ["vllm/vllm-openai@sha256:beef"]}
        container = mock.MagicMock(image=image)
        assert client_main._image_digest(container) == "vllm/vllm-openai@sha256:beef"
