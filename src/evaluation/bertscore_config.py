"""Configuration and constants for BERTScore threshold management.

This module defines the parameterizable BERTScore threshold settings and provides
validation for threshold values used across the evaluation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BERTScoreConfig:
    """Configuration for BERTScore threshold evaluation.
    
    Attributes:
        primary_threshold: Primary threshold for diagnosis acceptance (default: 0.85).
                          This is the main cutoff used in standard evaluation.
        sensitivity_thresholds: Range of thresholds to test during sensitivity analysis.
                               Default: [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        min_threshold: Minimum allowable threshold (safety bound).
        max_threshold: Maximum allowable threshold (safety bound).
        enable_sensitivity_analysis: Whether to run full sensitivity analysis pipeline.
        explanation_threshold: Separate threshold for explanation quality assessment.
    """
    
    primary_threshold: float = 0.85
    sensitivity_thresholds: list[float] = field(
        default_factory=lambda: [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    )
    min_threshold: float = 0.50
    max_threshold: float = 0.99
    enable_sensitivity_analysis: bool = False
    explanation_threshold: Optional[float] = 0.80
    
    def __post_init__(self):
        """Validate configuration values after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters."""
        # Validate primary threshold
        if not (self.min_threshold <= self.primary_threshold <= self.max_threshold):
            raise ValueError(
                f"primary_threshold {self.primary_threshold} must be between "
                f"{self.min_threshold} and {self.max_threshold}"
            )
        
        # Validate sensitivity thresholds
        for threshold in self.sensitivity_thresholds:
            if not (self.min_threshold <= threshold <= self.max_threshold):
                raise ValueError(
                    f"sensitivity_threshold {threshold} must be between "
                    f"{self.min_threshold} and {self.max_threshold}"
                )
        
        # Ensure primary threshold is in sensitivity range if not empty
        if self.sensitivity_thresholds and self.primary_threshold not in self.sensitivity_thresholds:
            print(
                f"Warning: primary_threshold {self.primary_threshold} not in "
                f"sensitivity_thresholds. Consider adding it for complete analysis."
            )
        
        # Validate explanation threshold
        if self.explanation_threshold is not None:
            if not (self.min_threshold <= self.explanation_threshold <= self.max_threshold):
                raise ValueError(
                    f"explanation_threshold {self.explanation_threshold} must be between "
                    f"{self.min_threshold} and {self.max_threshold}"
                )
    
    def get_threshold_by_name(self, name: str) -> float:
        """Retrieve threshold value by name.
        
        Args:
            name: One of 'primary', 'explanation', or a float value from sensitivity_thresholds.
            
        Returns:
            The threshold value.
            
        Raises:
            ValueError: If name is not recognized.
        """
        if name == "primary":
            return self.primary_threshold
        elif name == "explanation":
            if self.explanation_threshold is None:
                raise ValueError("explanation_threshold is not configured")
            return self.explanation_threshold
        else:
            try:
                threshold = float(name)
                if threshold in self.sensitivity_thresholds:
                    return threshold
                raise ValueError(f"Threshold {threshold} not in sensitivity_thresholds")
            except ValueError as e:
                raise ValueError(f"Unknown threshold name: {name}") from e
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "primary_threshold": self.primary_threshold,
            "sensitivity_thresholds": self.sensitivity_thresholds,
            "min_threshold": self.min_threshold,
            "max_threshold": self.max_threshold,
            "enable_sensitivity_analysis": self.enable_sensitivity_analysis,
            "explanation_threshold": self.explanation_threshold,
        }


# Default configuration instance
DEFAULT_CONFIG = BERTScoreConfig()
