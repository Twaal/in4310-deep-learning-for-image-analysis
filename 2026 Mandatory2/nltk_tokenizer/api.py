# Natural Language Toolkit: Tokenizer Interface
#
# Copyright (C) 2001-2025 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Tokenizer Interface
"""

import types
from abc import ABC, abstractmethod
from typing import Iterator, List, Tuple


class TokenizerI(ABC):
    """
    A processing interface for tokenizing a string.
    Subclasses must define ``tokenize()`` or ``tokenize_sents()`` (or both).
    """

    @abstractmethod
    def tokenize(self, s: str) -> List[str]:
        """
        Return a tokenized copy of *s*.

        :rtype: List[str]
        """
        if overridden(self.tokenize_sents):
            return self.tokenize_sents([s])[0]

    def span_tokenize(self, s: str) -> Iterator[Tuple[int, int]]:
        """
        Identify the tokens using integer offsets ``(start_i, end_i)``,
        where ``s[start_i:end_i]`` is the corresponding token.

        :rtype: Iterator[Tuple[int, int]]
        """
        raise NotImplementedError()

    def tokenize_sents(self, strings: List[str]) -> List[List[str]]:
        """
        Apply ``self.tokenize()`` to each element of ``strings``.  I.e.:

            return [self.tokenize(s) for s in strings]

        :rtype: List[List[str]]
        """
        return [self.tokenize(s) for s in strings]

    def span_tokenize_sents(
        self, strings: List[str]
    ) -> Iterator[List[Tuple[int, int]]]:
        """
        Apply ``self.span_tokenize()`` to each element of ``strings``.  I.e.:

            return [self.span_tokenize(s) for s in strings]

        :yield: List[Tuple[int, int]]
        """
        for s in strings:
            yield list(self.span_tokenize(s))


def overridden(method):
    """
    :return: True if ``method`` overrides some method with the same
        name in a base class.  This is typically used when defining
        abstract base classes or interfaces, to allow subclasses to define
        either of two related methods:

        >>> class EaterI:
        ...     '''Subclass must define eat() or batch_eat().'''
        ...     def eat(self, food):
        ...         if overridden(self.batch_eat):
        ...             return self.batch_eat([food])[0]
        ...         else:
        ...             raise NotImplementedError()
        ...     def batch_eat(self, foods):
        ...         return [self.eat(food) for food in foods]

    :type method: instance method
    """
    if isinstance(method, types.MethodType) and method.__self__.__class__ is not None:
        name = method.__name__
        funcs = [
            cls.__dict__[name]
            for cls in _mro(method.__self__.__class__)
            if name in cls.__dict__
        ]
        return len(funcs) > 1
    else:
        raise TypeError("Expected an instance method.")


def _mro(cls):
    """
    Return the method resolution order for ``cls`` -- i.e., a list
    containing ``cls`` and all its base classes, in the order in which
    they would be checked by ``getattr``.  For new-style classes, this
    is just cls.__mro__.  For classic classes, this can be obtained by
    a depth-first left-to-right traversal of ``__bases__``.
    """
    if isinstance(cls, type):
        return cls.__mro__
    else:
        mro = [cls]
        for base in cls.__bases__:
            mro.extend(_mro(base))
        return mro
