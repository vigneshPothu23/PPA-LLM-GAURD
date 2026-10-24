"""Engine 1: polymorphic prompt assembly (independent implementation). See docs/v5/ENGINE1.md."""

from .assembler import (AssembledPrompt, Engine1Error, InvalidInput, PolymorphicAssembler, ResponseCheck,
                        SeparatorCollision)
from .forgery import ForgeryFinding, find_forgery
from .separators import FAMILIES, TAGS, SeparatorGenerator, SeparatorPair
from .templates import TEMPLATES, Template

__all__ = ["AssembledPrompt", "Engine1Error", "FAMILIES", "ForgeryFinding", "InvalidInput", "PolymorphicAssembler",
           "ResponseCheck", "SeparatorCollision", "SeparatorGenerator", "SeparatorPair", "TAGS", "TEMPLATES", "Template",
           "find_forgery"]
