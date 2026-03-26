import numpy as np
from dataclasses import dataclass

@dataclass
class LemniscateParams:
    width:      float = 2.0    # Total width of the 8 [m]
    height:     float = 1.0    # Total height of the 8 [m]
    loop_time:  float = 10.0   # Time for one full 8 [s]
    z_center:   float = 2.0    # Average altitude [m]

class LemniscateTrajectory:
    def __init__(self, params: LemniscateParams = None):
        self.p = params or LemniscateParams()
        self.omega = 2.0 * np.pi / self.p.loop_time
        self.total_time = 20.0 # Run for 2 loops

    def get_reference(self, t: float) -> np.ndarray:
        # Standard Gerono Lemniscate formulas
        x = self.p.width * np.sin(self.omega * t)
        y = self.p.width * np.sin(self.omega * t) * np.cos(self.omega * t)
        z = self.p.z_center + 0.5 * np.sin(0.5 * self.omega * t)
        
        # Derivatives for velocity
        vx = self.p.width * self.omega * np.cos(self.omega * t)
        vy = self.p.width * self.omega * (np.cos(self.omega * t)**2 - np.sin(self.omega * t)**2)
        vz = 0.25 * self.omega * np.cos(0.5 * self.omega * t)

        ref = np.zeros(12)
        ref[0:3], ref[3:6] = [x, y, z], [vx, vy, vz]
        return ref
    
    def get_position(self, t: float) -> np.ndarray:
        """Returns only the [x, y, z] position at time t."""
        ref = self.get_reference(t)
        return ref[0:3]

    def get_acceleration(self, t: float) -> np.ndarray:
        # Analytical second derivatives for MPC/LQR feedforward
        ax = -self.p.width * (self.omega**2) * np.sin(self.omega * t)
        ay = -4 * self.p.width * (self.omega**2) * np.sin(self.omega * t) * np.cos(self.omega * t)
        az = -0.125 * (self.omega**2) * np.sin(0.5 * self.omega * t)
        return np.array([ax, ay, az])