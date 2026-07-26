class AronofyException(Exception):
    """
    Base exception class for MedBridge Backend.
    """
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class EntityNotFoundException(AronofyException):
    """
    Raised when a requested resource is missing.
    """
    def __init__(self, entity_name: str, identifier: str):
        super().__init__(f"{entity_name} with identifier '{identifier}' was not found.")
        self.entity_name = entity_name
        self.identifier = identifier


class AuthenticationException(AronofyException):
    """
    Raised when JWT token decoding, validation, or verification fails.
    """
    pass


class AuthorizationException(AronofyException):
    """
    Raised when a user attempts to access resources outside their RBAC permission level.
    """
    pass


class ConsentRequiredException(AronofyException):
    """
    Raised when a clinical data processing task is triggered but patient consent is missing.
    """
    pass


class BusinessRuleValidationException(AronofyException):
    """
    Raised when request data violates clinical business rules.
    """
    pass
