import subprocess
import threading
import time
from typing import Optional

import codecarbon
import wandb

MEASUREMENT_INTERVAL = 5  # in seconds
experiment_running = False


def log_power_consumption(power_samples):
    command = ["nvidia-smi", "-q", "-d", "POWER"]
    while experiment_running:
        error_occured = False
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            if result.returncode != 0:
                error_occured = True
        except Exception as e:
            error_occured = True

        if error_occured:
            power_samples = [-1]
            print("Unable to collect power usage samples using nvidia-smi")
            break

        # extract power consumption (in Watt) from the command output
        for output_line in result.stdout.splitlines():
            if "Power Draw" in output_line:
                power_sample = float(output_line.split(": ")[1].split()[0])
                break
        power_samples.append(power_sample)

        time.sleep(MEASUREMENT_INTERVAL)


def compute_kWh(power_samples):
    """power_samples is a list of power measurements (in Watt) at time points"""
    if len(power_samples) == 0 or sum(power_samples) == 0:
        return 0  # avoid division by zero
    average_power = sum(power_samples) / len(power_samples) / 1000  # in kWh
    duration = MEASUREMENT_INTERVAL * len(power_samples)  # in seconds
    return average_power * duration / 3600


class SustainabilityLogger:
    """This class tracks emissions and power consumption using CodeCarbon nvidia-smi."""

    def __init__(
        self, description: Optional[str] = None, log_file: Optional[str] = None
    ):
        self.description = description
        self.log_file = log_file
        self.power_measurements = []

    def __enter__(self):
        global experiment_running
        experiment_running = True

        # start library trackers
        self.codecarbon_tracker = codecarbon.EmissionsTracker(log_level="critical")
        self.codecarbon_tracker.start()

        # start logging kWh with nvidia-smi
        self.power_thread = threading.Thread(
            target=log_power_consumption, args=(self.power_measurements,)
        )
        self.power_thread.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        global experiment_running
        experiment_running = False

        prefix = "sustainability"
        if self.description is not None:
            prefix += f" ({self.description})"

        results = {
            f"{prefix}/CodeCarbon CO2_emissions(kg)": self.codecarbon_tracker.stop(),
            f"{prefix}/NVIDIA power_consumption(kWh)": compute_kWh(
                self.power_measurements
            ),
        }

        self.power_thread.join()

        if wandb.run is not None:  # log to wandb if wandb.init() has been called
            wandb.log(results)

        if self.log_file is not None:
            with open(self.log_file, "w") as f:
                f.write("\n".join(f"{name}: {v}" for name, v in results.items()))
