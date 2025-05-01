class DocumentError(Exception):
    """Base exception for document operations."""

    pass


class DocumentNotFoundError(DocumentError):
    """Raised when a document is not found."""

    pass


class StorageError(DocumentError):
    """Base exception for storage operations."""

    pass


class StorageIOError(StorageError):
    """Raised when storage I/O operations fail."""

    pass
