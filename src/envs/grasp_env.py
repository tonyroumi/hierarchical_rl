import sapien
import numpy as np
import gymnasium as gym
from mani_skill.utils import sapien_utils, common
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.wrappers.record import RecordEpisode
import torch
from transforms3d.euler import euler2quat


@register_env("GraspEnv", max_episode_steps=100)
class GraspEnv(BaseEnv):
    cube_halfsize = [0.02, 0.02, 0.02]

    def __init__(self, *args, robot_uids="panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.5, 0.4, 1.0], target=[-0.5, -0.2, -0.5])

        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 1]))

    def _load_lighting(self, options: dict):
        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_directional_light(
            [1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=2048
        )
        self.scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self
        )
        self.table_scene.build()
        wood_material = sapien.render.RenderMaterial()
        wood_material.base_color_texture = sapien.render.RenderTexture2D("rosewood_veneer1_diff_1k.png")

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(
            half_size = self.cube_halfsize
        )
        builder.add_box_visual(
            half_size=self.cube_halfsize,
            material=sapien.render.RenderMaterial(
                base_color=[1, 0, 0, 1],
            ),
        )
        builder.initial_pose = sapien.Pose(p=[0, 0.1, 0.04], q=[1, 0, 0, 0])
        self.object = builder.build(name="cube")
        

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # using torch.device context manager to auto create tensors
        # on CPU/CUDA depending on self.device, the device the env runs on
        with torch.device(self.device):
            # the number of parallel envs running, it is just 1 in this assignment but recall that all data in maniskill is batched by default
            b = len(env_idx)

            # this initialization will automatically place the table and ground plane in the right place
            # and put the panda arm at the edge of the table
            self.table_scene.initialize(env_idx)

            # Generate cube position
            p_cube = torch.zeros((b, 3))
            p_cube[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1  
            p_cube[..., 2] = self.cube_halfsize[-1]  
            q_cube = [1, 0, 0, 0]

            cube_pose = Pose.create_from_pq(p=p_cube, q=q_cube)
            self.object.set_pose(cube_pose)
#

    def evaluate(self):
        success = torch.zeros((self.num_envs, ), dtype=bool)
        obj_pos = (torch.tensor(self.object.pose.p))
        ee_pos = (torch.tensor(self.agent.controller.get_ee_pose().p))
        dist = torch.norm(obj_pos - ee_pos)
        lifted = obj_pos[2] > 0.1
        grasped = dist < 0.05
        success = lifted & grasped

        return dict(success=success)