from lofi.spec.language import AnnotationError, LanguageError
from lofi.spec.language._errors import AnnotationError as DeclaredAnnotationError


def test_annotation_error_is_the_public_language_refusal() -> None:
    assert AnnotationError is DeclaredAnnotationError
    assert issubclass(AnnotationError, LanguageError)
