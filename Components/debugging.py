import torch

import time


def print_gpu_memory_info(device, info=""):
    """Prints the GPU memory information for the specified device.
    NOTE: There are far more advanced inbuild cuda functions to get memory info. Use these!"""
    
    if device.type != 'cuda' :
        return
    
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)

    # Calculate the free memory
    free_memory = total_memory - allocated_memory
    
    print(info)
    print(f"  Total Memory: {total_memory / 1024**2} MB")
    print(f"  Allocated Memory: {allocated_memory / 1024**2} MB")
    print(f"  Free Memory: {free_memory / 1024**2} MB")
    print("_____________________\n")



class DebugTimer:
    """A simple timer class for debugging purposes."""
    
    def __init__(self):
        self.tracker = {}

    def start(self, key):
        """Start the timer for a given key."""

        # Initialize the key if it doesn't exist
        if key not in self.tracker:
            self.tracker[key] = {"start": [], "end": []}

        # Record the start time
        self.tracker[key]["start"].append(time.perf_counter())
    
    def stop(self, key):
        """Stop the timer for a given key."""
        # Record the end time
        self.tracker[key]["end"].append(time.perf_counter())
  
    def report(self):
        """Report the elapsed time for each key."""

        total_time = 0.0
        
        print("\n--- Debug Timer Report ---")
        for key, times in self.tracker.items():
            start_times = torch.tensor(times["start"], dtype=torch.float64)
            end_times = torch.tensor(times["end"], dtype=torch.float64)

            elapsed_times = end_times - start_times
            average_time = elapsed_times.mean().item()

            # Print with full precision
            print(f"{key}")
            print(f"  Average Time: {average_time:.6f} seconds over {len(elapsed_times)} runs\n")

            total_time += average_time

        print(f"Total Time Across All Keys: {total_time:.6f} seconds")
        print("--------------------------\n")
