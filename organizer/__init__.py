"""organizer — an extensible, learn-your-structure file organizer."""

from .core import Classifier, Decision, FileContext, Outcome, Policy, run_chain  # noqa: F401

__all__ = ["FileContext", "Decision", "Classifier", "run_chain", "Policy", "Outcome"]
__version__ = "0.1.0"
