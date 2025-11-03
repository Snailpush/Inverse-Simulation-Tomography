# Inverse Simulation Tomography

This project estimates the **pose trajectory** and **RI distibutions** of microscopic transparent objects by comparing simulated scattered wavefields with observed wavefields. The system takes 3D tomography data, simulates the light scattering process, and optimizes the pose to minimize the difference from experimental observations.


---


## 🧠 Tasks

* **Wavefield Simulation** with predefined Poses
* **Pose optimization** towards recorded ground truth frames
* **RI Distribution Reconstruction**

---

## ✅ Features

* Differentiable BPM implementation
* Pose optimization via gradient-based methods
* Volume Reconstruction via gradient-based methods
* Configurable data domain adaptaions
* Logging via local files with additional progress tracking via [Weights & Biases](https://wandb.ai/)

---

## 📁 Repository Structure

```
.
├── main/                 # Main Interface for each task
├── Core/                 # Core logic for each task
├── Components/           # Individual reusable modules
├── Auxiliary/            # General utility functions not used directly by the BPM Simulation
    ├── Preprocessing/    # Utilities to convert rough 3D data to usable Voxel Objects
    ├── utils/            # Helper functions for Data Comparison and Config generation
    ├── misc/             # Miscilanious helper functions 
├── Configs/              # JSON config files for experiment management
├── Data/                 # Data files - Not included  
└── Outputs/              # Automatically created to store run outputs
```

---

## ⚙️ Installation
> A `requirements.txt` file is provided, but it has not been thoroughly tested.

```bash
pip install -r requirements.txt
```

Ensure a Python environment with PyTorch and supporting libraries (e.g., NumPy, matplotlib) is available.

---

## 📖 Usage

Detailed usage instructions and example workflows are provided in a separate file: `Guide.pdf`.

This document will cover:

* How to prepare input data
* What each Core script is used for
* How to configure you config files

---

## 🧪 Testing

Currently, no formal tests or demos are included.

Pose optimization and training are handled directly within the scripts in `BPM Simulation/`.

---

## 📈 Results

No benchmark results or visual examples are currently included.

Generated outputs are stored in the `Outputs/` folder and can be analyzed post-run.

---

## 👤 Author

Developed by [Jannis Maron](mailto:Jannis.Maron@uni-siegen.de)

---

## 📄 License

-

---

## 🔗 References & Links

- Tomography data created by: [TIGRE Toolbox](https://github.com/CERN/TIGRE)
