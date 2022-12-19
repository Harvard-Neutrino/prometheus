# -*- coding: utf-8 -*-
# injection.py
# Copyright (C) 2022 Jeffrey Lazar, Stephan Meighen-Berger
# Interface class to the different lepton injectors

from abc import ABC, abstractmethod

class Injection(ABC):

    @property
    @abstractmethod
    def injection_energy(self):
        pass

    @property
    @abstractmethod
    def injection_type(self):
        pass

    @property
    @abstractmethod
    def injection_interaction_type(self):
        pass

    @property
    @abstractmethod
    def injection_zenith(self):
        pass

    @property
    @abstractmethod
    def injection_azimuth(self):
        pass

    @property
    @abstractmethod
    def injection_bjorkenx(self):
        pass

    @property
    @abstractmethod
    def injection_bjorkeny(self):
        pass

    @property
    @abstractmethod
    def injection_position_x(self):
        pass

    @property
    @abstractmethod
    def injection_position_y(self):
        pass

    @property
    @abstractmethod
    def injection_position_z(self):
        pass

    @property
    @abstractmethod
    def injection_column_depth(self):
        pass

    @property
    @abstractmethod
    def primary_lepton_1_type(self):
        pass

    @property
    @abstractmethod
    def primary_lepton_1_energy(self):
        pass

    @property
    @abstractmethod
    def primary_lepton_1_position_x(self):
        pass

    @property
    @abstractmethod
    def primary_lepton_1_position_y(self):
        pass

    @property
    @abstractmethod
    def primary_lepton_1_position_z(self):
        pass

    @property
    @abstractmethod
    def primary_lepton_1_direction_theta(self):
        pass

    @property
    @abstractmethod
    def primary_lepton_1_direction_phi(self):
        pass

    @property
    @abstractmethod
    def primary_hadron_1_type(self):
        pass
    
    @property
    @abstractmethod
    def primary_hadron_1_energy(self):
        pass

    @property
    @abstractmethod
    def primary_hadron_1_position_x(self):
        pass

    @property
    @abstractmethod
    def primary_hadron_1_position_y(self):
        pass

    @property
    @abstractmethod
    def primary_hadron_1_position_z(self):
        pass

    @property
    @abstractmethod
    def primary_hadron_1_direction_theta(self):
        pass

    @property
    @abstractmethod
    def primary_hadron_1_direction_phi(self):
        pass

    @property
    @abstractmethod
    def total_energy(self):
        pass

    @abstractmethod
    def serialize_to_awkward(self):
        pass

    @abstractmethod
    def serialize_to_dict(self):
        pass
