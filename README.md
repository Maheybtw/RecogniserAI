# RecogniserAI

**RecogniserAI** is an adaptive control framework for deep learning.  
It dynamically tunes learning behaviour using real-time feedback on efficiency and energy usage.

---

## 🚀 Quick Start

### 1️⃣ Install (development mode)
```bash
git clone https://github.com/Maheybtw/recogniserai.git
cd recogniserai
pip install -e .


Run the demo:
python -m recogniserai.examples

Example usage:
from recogniserai import AdaptiveController, EfficiencyMeter, EnergyTracker

ctrl = AdaptiveController(target_eff=0.75)
print(ctrl)
