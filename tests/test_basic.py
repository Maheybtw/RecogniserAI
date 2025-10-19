from recogniserai.core.controller import AdaptiveController

def test_controller_init():
    ctrl = AdaptiveController(target_eff=0.8)
    assert ctrl is not None
