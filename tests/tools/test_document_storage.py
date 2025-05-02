"""Tests for document storage operations."""

import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from edison.tools.document_storage import DocumentStorage
from edison.models import DocumentContent, DocumentSection, DocumentMetdataItem
from edison.errors import StorageError, StorageIOError


@pytest.fixture
def storage_dir(tmp_path):
    """Provides a temporary directory for storage tests."""
    return str(tmp_path / "test_documents")


@pytest.fixture
def storage(storage_dir):
    """Returns a DocumentStorage instance with temp directory."""
    return DocumentStorage(storage_dir)


@pytest.fixture
def sample_document():
    """Creates a sample document for testing."""
    return DocumentContent(
        sections={
            "intro": DocumentSection(
                title="Introduction",
                content="Test content",
                last_modified=datetime(2024, 1, 1),
                version=1,
                context_tokens=100,
            )
        },
        metadata=[DocumentMetdataItem(key="author", value="test")],
        created_at=datetime(2024, 1, 1),
        last_modified=datetime(2024, 1, 1),
        version=1,
    )


def test_storage_init_creates_directory(tmp_path):
    """Test storage directory creation on initialization."""
    storage_path = tmp_path / "new_dir"
    assert not storage_path.exists()

    DocumentStorage(str(storage_path))
    assert storage_path.exists()
    assert storage_path.is_dir()


def test_storage_init_fails_with_invalid_path():
    """Test initialization with invalid path."""
    with pytest.raises(StorageError):
        DocumentStorage("/invalid/:/path")


def test_save_document(storage, sample_document):
    """Test successful document save operation."""
    doc_id = "test_doc"
    storage.save_document(doc_id, sample_document)

    saved_path = Path(storage.storage_dir) / f"{doc_id}.json"
    assert saved_path.exists()

    with saved_path.open() as f:
        saved_data = json.load(f)
        assert len(saved_data["metadata"]) == 1
        assert saved_data["metadata"][0]["key"] == "author"
        assert saved_data["metadata"][0]["value"] == "test"
        assert "intro" in saved_data["sections"]
        assert saved_data["sections"]["intro"]["title"] == "Introduction"


@patch("pathlib.Path.open", side_effect=IOError("mocked error"))
def test_save_document_io_error(mock_open, storage, sample_document):
    """Test document save with IO error."""
    with pytest.raises(StorageIOError):
        storage.save_document("test_doc", sample_document)


def test_load_document(storage, sample_document):
    """Test successful document load operation."""
    doc_id = "test_doc"
    storage.save_document(doc_id, sample_document)

    loaded_doc = storage.load_document(doc_id)
    assert loaded_doc is not None
    assert loaded_doc.metadata == sample_document.metadata
    assert loaded_doc.sections["intro"].title == sample_document.sections["intro"].title
    assert loaded_doc.created_at == sample_document.created_at


def test_load_nonexistent_document(storage):
    """Test loading non-existent document returns None."""
    assert storage.load_document("nonexistent") is None


def test_load_corrupted_document(storage):
    """Test loading corrupted document file."""
    doc_id = "corrupted"
    doc_path = Path(storage.storage_dir) / f"{doc_id}.json"
    doc_path.parent.mkdir(parents=True, exist_ok=True)

    with doc_path.open("w") as f:
        f.write("invalid json")

    with pytest.raises(StorageIOError):
        storage.load_document(doc_id)


def test_list_documents_empty(storage):
    """Test listing documents in empty directory."""
    docs = storage.list_documents()
    assert docs == {}


def test_list_documents_with_content(storage, sample_document):
    """Test listing multiple documents."""
    storage.save_document("doc1", sample_document)
    storage.save_document("doc2", sample_document)

    docs = storage.list_documents()
    assert len(docs) == 2
    assert "doc1" in docs
    assert "doc2" in docs
    assert all("author" in doc and doc["author"] == "test" for doc in docs.values())


def test_list_documents_with_corrupted_file(storage, sample_document):
    """Test listing documents with a corrupted file present."""
    # Save valid document
    storage.save_document("valid_doc", sample_document)

    # Create corrupted document
    corrupt_path = Path(storage.storage_dir) / "corrupt.json"
    with corrupt_path.open("w") as f:
        f.write("invalid json")

    docs = storage.list_documents()
    assert len(docs) == 1
    assert "valid_doc" in docs


def test_document_datetime_serialization(storage, sample_document):
    """Test datetime fields are properly serialized and deserialized."""
    doc_id = "test_datetime"
    storage.save_document(doc_id, sample_document)

    loaded_doc = storage.load_document(doc_id)
    assert loaded_doc.created_at == sample_document.created_at
    assert loaded_doc.last_modified == sample_document.last_modified
    assert (
        loaded_doc.sections["intro"].last_modified
        == sample_document.sections["intro"].last_modified
    )


@pytest.mark.parametrize(
    "doc_id",
    ["normal-id", "with spaces", "with/slashes", "with.dots", "with#special@chars"],
)
def test_document_id_handling(storage, sample_document, doc_id):
    """Test handling of various document ID formats."""
    storage.save_document(doc_id, sample_document)
    loaded_doc = storage.load_document(doc_id)
    assert loaded_doc is not None


def test_load_document_with_missing_dates(storage):
    """Test loading document with missing datetime fields."""
    doc_id = "test_missing_dates"
    doc_path = Path(storage.storage_dir) / f"{doc_id}.json"

    # Create document with missing datetime fields
    test_data = {
        "sections": {
            "intro": {
                "title": "Test",
                "content": "Content",
                "last_modified": None,
                "version": 1,
                "context_tokens": 100,
            }
        },
        "metadata": [],
        "created_at": None,
        "last_modified": None,
        "version": 1,
    }

    doc_path.parent.mkdir(parents=True, exist_ok=True)
    with doc_path.open("w") as f:
        json.dump(test_data, f)

    loaded_doc = storage.load_document(doc_id)
    assert loaded_doc is not None
    assert loaded_doc.created_at is None
    assert loaded_doc.last_modified is None
    assert loaded_doc.sections["intro"].last_modified is None


def test_load_document_with_partial_data(storage):
    """Test loading document with partial data fields."""
    doc_id = "test_partial"
    doc_path = Path(storage.storage_dir) / f"{doc_id}.json"

    # Create document with minimal required fields
    test_data = {"sections": {}, "metadata": [], "version": 1}

    doc_path.parent.mkdir(parents=True, exist_ok=True)
    with doc_path.open("w") as f:
        json.dump(test_data, f)

    loaded_doc = storage.load_document(doc_id)
    assert loaded_doc is not None
    assert loaded_doc.sections == {}
    assert loaded_doc.metadata == []
    assert loaded_doc.version == 1
