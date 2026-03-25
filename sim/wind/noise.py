"""
Stochastic Wind Disturbance Model
================================================================================
Implements an Ornstein-Uhlenbeck (OU) process to generate realistic,
time-correlated wind gusts for the quadrotor simulation.
"""

import numpy as np

class WindModel:
    """
    Generator for stochastic wind forces using an Ornstein-Uhlenbeck process.
    
    The OU process is defined by:
        dW = theta * (mean - W) * dt + sigma * dB
    
    where:
        - W     : Current wind force [N]
        - mean  : Long-term average wind force [N]
        - theta : Reversion rate (how quickly it returns to mean)
        - sigma : Volatility (strength of random gusts)
        - dB    : Brownian motion increment (Gaussian noise)
    """

    def __init__(self, dt: float = 0.01, mean=None, theta=0.5, sigma=0.2):
        """
        Initialise the wind model.
        
        Args:
            dt    : Timestep [s]
            mean  : 3D vector for average [x, y, z] wind force [N]
            theta : Reversion strength
            sigma : Maximum gust intensity
        """
        self.dt    = dt
        self.mean  = np.array(mean) if mean is not None else np.zeros(3)
        self.theta = theta
        self.sigma = sigma
        self.enabled = True  # Toggle for the entire wind model
        
        # Current state of the wind force
        self.force = np.copy(self.mean)

    def step(self) -> np.ndarray:
        """
        Advance the wind state by one timestep.
        
        Returns:
            force : Current 3D wind force vector [N]
        """
        if not self.enabled:
            self.force = np.zeros(3)
            return self.force

        dw = np.random.normal(0, np.sqrt(self.dt), size=3)
        self.force += self.theta * (self.mean - self.force) * self.dt + \
                      self.sigma * dw
        return self.force

    def reset(self):
        """Reset the wind force to its mean value."""
        self.force = np.copy(self.mean)

    def set_params(self, mean=None, theta=None, sigma=None):
        """Update the wind parameters."""
        if mean  is not None: self.mean  = np.array(mean)
        if theta is not None: self.theta = theta
        if sigma is not None: self.sigma = sigma
