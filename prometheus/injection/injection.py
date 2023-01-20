# -*- coding: utf-8 -*-
# injection.py
# Copyright (C) 2022 Jeffrey Lazar, Stephan Meighen-Berger
# Interface class to the different lepton injectors

from typing import Iterable

from .injection_event import InjectionEvent

class Injection:
    
    def __init__(
        self,
        events: Iterable[InjectionEvent]
    ):
        self._events = events
        self._size = len(events)
        self._current_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_idx >= self._size:
            self._current_idx = 0
            raise StopIteration
        event = self.events[self._current_idx]
        self._current_idx += 1
        return event

    @property
    def events(self):
        return self._events
