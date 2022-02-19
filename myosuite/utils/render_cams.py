""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """


DESC = '''
Helper script to render images offscreen and save using a mujoco model.\n
USAGE:\n
    $ python render_cams.py --model_path franka_sim.xml --cam_names top_cam --cam_names left_cam --cam_names right_cam \n
EXAMPLE:\n
    $ python utils/render_cams.py -m envs/fm/assets/franka_microwave.xml -c top_cam -c left_cam -c right_cam
'''

from mujoco_py import load_model_from_path, MjSim
from PIL import Image
import numpy as np
import click


def render_camera_offscreen(cameras:list, width:int=640, height:int=480, device_id:int=0, sim=None):
    """
    Render images(widthxheight) from a list_of_cameras on the specified device_id.
    """
    imgs = np.zeros((len(cameras), height, width, 3), dtype=np.uint8)
    for ind, cam in enumerate(cameras) :
        img = sim.render(width=width, height=height, mode='offscreen', camera_name=cam, device_id=device_id)
        img = img[::-1, :, : ] # Image has to be flipped
        imgs[ind, :, :, :] = img
    return imgs


@click.command(help=DESC)
@click.option('-m', '--model_path', required=True, type=str, help='model file')
@click.option('-c', '--cam_names', required=True, multiple=True, help=('Camera names for rendering'))
@click.option('-w', '--width', type=int, default=640, help='image width')
@click.option('-h', '--height', type=int, default=480, help='image height')
@click.option('-d', '--device_id', type=int, default=0, help='device id for rendering')

def main(model_path, cam_names, width, height, device_id):
    # render images
    model = load_model_from_path(model_path)
    sim = MjSim(model)
    imgs = render_camera_offscreen(cameras=cam_names, width=width, height=height, device_id=device_id, sim=sim)

    # save images
    for i, cam in enumerate(cam_names):
        image = Image.fromarray(imgs[i])
        image.save(cam+".jpeg")
        print("saved "+cam+".jpeg")

if __name__ == '__main__':
    main()