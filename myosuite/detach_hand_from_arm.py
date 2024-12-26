
import mujoco # tested with Version: 3.2.4
from mujoco import viewer
import time

file_path = "simhive/myo_sim/arm/myoarm.xml"
spec = mujoco.MjSpec.from_file(file_path)

# find body to remove
b_lunate = spec.find_body('lunate')
b_lunate_pos = b_lunate.pos.copy()

# add site to the parent body
b_radius = spec.find_body('radius')
b_radius.add_site(
  name='wrist',
    pos=b_lunate_pos,
    group=3
)

# Remove the body
spec.detach_body(b_lunate)

# compile model and
mj_model = spec.compile()
mj_data = mujoco.MjData(mj_model)
window = viewer.launch_passive(
        mj_model,
        mj_data,
        show_left_ui=False,
        show_right_ui=False,
    )

while window.is_running():
  mujoco.mj_step(mj_model, mj_data)
  window.sync()
  time.sleep(.01)

print("Success. Clear Exit")