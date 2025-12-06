# deephall/monitoring.py
import numpy as np
from typing import List, Dict
import logging
from pathlib import Path

logger = logging.getLogger("deephall")

class VMC_Monitor:
    def __init__(self, window_size: int = 100, output_file: str = None):
        self.energy_history: List[float] = []
        self.acceptance_history: List[float] = []
        self.window_size = window_size
        self.output_file = output_file
        self.stats_file = None
        
        # Initialize CSV file if output_file is provided
        if self.output_file:
            self._init_csv_file()

    def _init_csv_file(self):
        """Initialize the CSV file with headers."""
        self.stats_file = Path(self.output_file)
        with open(self.stats_file, 'w') as f:
            f.write("step,energy_mean,energy_std,energy_variance,acceptance_mean,autocorr_time\n")
    
    def update(self, energy: float, acceptance_rate: float):
        """Update monitoring data."""
        self.energy_history.append(energy)
        self.acceptance_history.append(acceptance_rate)
        
        # Keep only recent history
        if len(self.energy_history) > self.window_size:
            self.energy_history.pop(0)
            self.acceptance_history.pop(0)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get current monitoring statistics."""
        if len(self.energy_history) < 10:
            return {}
        
        energies = np.array(self.energy_history)
        acceptances = np.array(self.acceptance_history)
        
        return {
            'energy_mean': np.mean(energies),
            'energy_std': np.std(energies),
            'energy_variance': np.var(energies),
            'acceptance_mean': np.mean(acceptances),
            'autocorr_time': self._compute_autocorr_time(energies)
        }
    
    def log_statistics(self, step: int):
        """Log statistics to CSV file."""
        if self.stats_file and len(self.energy_history) >= 10:
            stats = self.get_statistics()
            with open(self.stats_file, 'a') as f:
                f.write(f"{step},{stats.get('energy_mean', 0):.6f},"
                       f"{stats.get('energy_std', 0):.6f},"
                       f"{stats.get('energy_variance', 0):.6f},"
                       f"{stats.get('acceptance_mean', 0):.6f},"
                       f"{stats.get('autocorr_time', 0):.2f}\n")
    
    def _compute_autocorr_time(self, energies: np.ndarray) -> float:
        """Compute autocorrelation time."""
        if len(energies) < 10:
            return 0.0
        
        autocorr = np.correlate(energies, energies, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]
        
        threshold = 1.0 / np.e
        autocorr_time = np.argmax(autocorr < threshold)
        return float(autocorr_time)

    def close(self):
        """Close the monitoring file."""
        if self.stats_file:
            # File is already closed after each write
            pass