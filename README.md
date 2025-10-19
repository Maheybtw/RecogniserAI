
<div align="center">

# 🧠 RecogniserAI  
**Adaptive control framework for self-regulating deep learning systems**  

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Maheybtw/RecogniserAI/blob/main/LICENSE)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)

</div>

---

### 🧠 Overview
**RecogniserAI** introduces adaptive feedback-driven learning control for neural networks.  
It dynamically tunes internal parameters such as learning rate and efficiency targets in real time, improving convergence stability and energy efficiency.


---

### 💡 Use Cases

RecogniserAI is designed for developers, researchers, and students exploring adaptive optimisation and self-regulating AI systems.  
Its flexible architecture makes it easy to integrate into existing PyTorch workflows or serve as a foundation for experimentation.

**🔬 Research and Academia**
- Study feedback-based control in deep learning
- Prototype self-tuning optimisation systems
- Reproduce and extend adaptive learning experiments

**⚙️ Applied Machine Learning**
- Automatically tune learning rates and efficiency targets in real time  
- Improve model stability during training for difficult tasks (GANs, diffusion, RL)
- Monitor and optimise energy efficiency during large-scale training

**🌱 Sustainable and Efficient AI**
- Track and reduce computational energy use  
- Develop “green AI” solutions with adaptive energy balancing

**🧩 Experimental Optimisation**
- Create models that *learn how to learn* through closed-loop adaptive control  
- Build new training paradigms based on dynamic feedback rather than static schedules

---

> RecogniserAI bridges control theory and deep learning — bringing adaptive intelligence to the optimisation process itself.


### ⚙️ Installation
```bash
# clone the repository
git clone https://github.com/Maheybtw/RecogniserAI.git
cd RecogniserAI

# install in editable mode
pip install -e .
````

Dependencies are listed in `requirements.txt`.

---

### 🚀 Quick Start

Run the built-in CIFAR demo:

```bash
python -m recogniserai.examples
```

Or import directly:

```python
from recogniserai import AdaptiveController, EfficiencyMeter, EnergyTracker

ctrl = AdaptiveController(target_eff=0.85)
```

---

### 📂 Project Structure

```
recogniserai/
├── core/
│   ├── controller.py
│   ├── efficiency.py
│   ├── energy.py
│   └── __init__.py
├── examples/
│   ├── demo_cifar.py
│   └── __main__.py
├── tests/
└── __init__.py
```

---

### 🧩 Features

* Real-time adaptive control of model efficiency
* Feedback loop visualization
* Plug-and-play with existing PyTorch pipelines
* Energy and stability tracking

---

### 📅 Roadmap

* [ ] Add ImageNet-scale benchmark support
* [ ] Implement hybrid CPU/GPU adaptive scheduling
* [ ] Publish documentation site
* [ ] Prepare v1.0 release

---

### 🤝 Contributing

Pull requests are welcome!
For major changes, please open an issue first to discuss what you’d like to modify.


---

### 🧾 Recent Updates

**v0.1.1 — October 2025**
- Fixed repeated prints from module imports and DataLoader workers  
- Unified GPU/CPU detection with clean one-time startup message  
- Restored CUDA support for RTX-class GPUs  
- General stability and formatting improvements for `demo_cifar.py`

> Next up: preparing documentation and adding example visualisations for the adaptive feedback loop.

---


### ⚖️ License

This project is licensed under the [Apache 2.0 License](LICENSE) © 2025 Maheybtw.

````

