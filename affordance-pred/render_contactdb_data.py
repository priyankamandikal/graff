'''
Generate input rgb and gt affordance mask data for supervised affordance prediction model.
Camera is set to first person and hand is made invisible.
Run as:
    debug: python affordance-pred/render_contactdb_data.py --debug --obj apple banana cup cell_phone door_knob flashlight hammer knife light_bulb mouse mug pan scissors stapler teapot toothbrush toothpaste water_bottle
    run: python affordance-pred/render_contactdb_data.py --obj apple banana cup cell_phone door_knob flashlight hammer knife light_bulb mouse mug pan scissors stapler teapot toothbrush toothpaste water_bottle
'''

import numpy as np
from pyquaternion import Quaternion
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.utils import seeding
from mujoco_py import MjSim, functions, load_model_from_path, MjRenderContextOffscreen, MjRenderContext
from mujoco_py.generated import const
import os
from os.path import join, dirname, abspath, exists
import json
from math import exp, pow
import re
import cv2


class GraspV0(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, device_id=0, grasp_attrs_dict=None):

        self.grasp_attrs_dict = grasp_attrs_dict
        self.obj_name = grasp_attrs_dict['obj']
        self.S_grasp_sid = 0
        self.obj_bid = 0
        self.table_bid = 0
        self.device_id = device_id
        self.curr_dir = dirname(abspath(__file__))
        env_dir = join(self.curr_dir, '../envs/mj_envs/dex_manip')
        model_path = join(env_dir, 'assets/contactdb/env_xmls/%s_vhacd.xml' % self.obj_name)
        frame_skip = 5
        self.seed()

        # # update scale value in mesh xml
        # # this lead to problems while running parallel processes since the same sml file is being updated
        # self.scale = self.grasp_attrs_dict['scale']
        # regex = r'scale=\"([^\"]+)\"'
        # replacement = 'scale="{} {} {}"'.format(self.scale, self.scale, self.scale)
        # mesh_fname = join(curr_dir, 'assets/vhacd/%s_meshes.xml'%self.obj_name)
        # with open(mesh_fname, 'r') as f:
        #     mesh_xml = f.read()
        # mesh_xml = re.sub(regex, replacement, mesh_xml)
        # with open(mesh_fname, 'w') as f:
        #     f.write(mesh_xml)

        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = join(dirname(__file__), "assets", model_path)
        if not exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = load_model_from_path(fullpath)
        self.sim = MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        self.frame_size = (self.grasp_attrs_dict['img_res'], self.grasp_attrs_dict['img_res'])
        self.min_depth = {'fixed': 0.9, 'first_person': 0.9, 'egocentric': 0.9, 'egocentric_zoom': 0.7}
        self.max_depth = {'fixed': 1.0, 'first_person': 1.0, 'egocentric': 1.0, 'egocentric_zoom': 1.0}
        
        if self.grasp_attrs_dict['debug']:
            self.tmp = 0
            os.makedirs('images/%s'%str(self.device_id), exist_ok=True)

        # contact
        self.contact_json = join(env_dir, 'assets/contactdb/contacts/%s.json'%self.obj_name)

        # max contact
        with open(self.contact_json, 'r') as f:
            data = json.load(f)
            self.max_geom_idx = data['max']['geom']
            self.max_contact_idx = data['max']['vert']
            geom_indices = data['multicontact']['geom']
            contact_indices = data['multicontact']['vert']

        # multicontact
        self.glyph_size = self.grasp_attrs_dict['glyph_size']
        self.n_verts_multicontact = self.grasp_attrs_dict['n_verts_multicontact']
        self.fps_geom_indices, self.fps_contact_indices = self.get_fps(geom_indices, contact_indices, self.n_verts_multicontact)

        # geoms
        self.objgeom_names = [name for name in self.sim.model.geom_names if 'Obj_mesh_' in name]
        self.nobjgeoms = len(self.objgeom_names)

        # camera
        self.camera_id_first_person = self.model.camera_name2id('first_person')
        self.camera_id_left = self.model.camera_name2id('left')
        self.camera_id_right = self.model.camera_name2id('right')
        self.camera_id_egocentric = self.model.camera_name2id('egocentric')

        self.camera = self.grasp_attrs_dict['camera']

        mujoco_env.MujocoEnv.__init__(self, model_path, frame_skip)

        self.mjr_render_context_offscreen = MjRenderContext(self.sim, offscreen=True, opengl_backend='egl')
        self.sim._render_context_offscreen.vopt.flags[0] = 0 # display convex hull
        # self.sim.add_render_context(self.mjr_render_context_offscreen)

        self.S_grasp_sid = self.sim.model.site_name2id('S_grasp')
        self.obj_bid = self.sim.model.body_name2id('Object')
        self.table_bid = self.sim.model.body_name2id('table')
        utils.EzPickle.__init__(self)

        # set mass of the object
        self.sim.model.body_mass[self.obj_bid] = self.grasp_attrs_dict['mass']

        # set scale of the object
        for obj_mesh_idx in range(self.nobjgeoms):
            obj_geom_idx = self.model.geom_name2id('Obj_mesh_%d'%obj_mesh_idx)
            self.model.geom_pos[obj_geom_idx] *= self.grasp_attrs_dict['scale']
            start_idx = self.model.mesh_vertadr[obj_mesh_idx]
            end_idx = self.model.mesh_vertadr[obj_mesh_idx] + self.model.mesh_vertnum[obj_mesh_idx]
            for ind in range(start_idx, end_idx):
                self.model.mesh_vert[ind] *= self.grasp_attrs_dict['scale']

        # set gravity of the environment
        self.sim.model.opt.gravity[:] = [0, 0, self.grasp_attrs_dict['gravity']]

        functions.mj_setConst(self.sim.model, self.sim.data)


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def get_contact_pos(self):
        geom_mesh_idx = self.max_geom_idx
        geom_idx = self.model.geom_name2id('Obj_mesh_%d'%self.max_geom_idx)
        ind = self.max_contact_idx
        vert = self.model.mesh_vert[self.model.mesh_vertadr[geom_mesh_idx] + ind]
        body_xpos = self.data.body_xpos[self.obj_bid]
        body_xquat = self.data.body_xquat[self.obj_bid].ravel()
        geom_pos = self.model.geom_pos[geom_idx].ravel()
        geom_quat = self.model.geom_quat[geom_idx]
        vert = Quaternion(geom_quat).rotate(vert) + geom_pos
        vert = Quaternion(body_xquat).rotate(vert) + body_xpos
        return vert


    def get_fps(self, geom_indices, contact_indices, K):

        def get_verts(geom_indices, contact_indices):
            contact_verts = []
            for geom_mesh_idx, ind in zip(geom_indices, contact_indices):
                vert = self.model.mesh_vert[self.model.mesh_vertadr[geom_mesh_idx] + ind]
                geom_idx = self.model.geom_name2id('Obj_mesh_%d' % geom_mesh_idx)
                vert = Quaternion(self.model.geom_quat[geom_idx]).rotate(vert) + self.model.geom_pos[geom_idx]
                contact_verts.append(vert)
            return contact_verts

        def calc_distances(p0, points):
            return ((p0 - points) ** 2).sum(axis=1)

        pts = get_verts(geom_indices, contact_indices)
        indices = []
        farthest_pts = np.zeros((K, 3))
        indices.append(0)
        farthest_pts[0] = pts[0]
        distances = calc_distances(farthest_pts[0], pts)
        for i in range(1, K):
            next_point = np.argmax(distances)
            indices.append(next_point)
            farthest_pts[i] = pts[next_point]
            distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
        farthest_geoms = [geom_indices[ele] for ele in indices]
        farthest_indices = [contact_indices[ele] for ele in indices]
        return farthest_geoms, farthest_indices


    def get_multicontact_verts(self, verbose=False):
        body_xpos = self.data.body_xpos[self.obj_bid]
        body_xquat = self.data.body_xquat[self.obj_bid].ravel()
        contact_verts = []
        for geom_mesh_idx, contact_idx in zip(self.fps_geom_indices, self.fps_contact_indices):
            vert = self.model.mesh_vert[self.model.mesh_vertadr[geom_mesh_idx] + contact_idx]
            geom_idx = self.model.geom_name2id('Obj_mesh_%d' % geom_mesh_idx)
            geom_pos = self.model.geom_pos[geom_idx].ravel()
            geom_quat = Quaternion(self.model.geom_quat[geom_idx])
            vert = Quaternion(geom_quat).rotate(vert) + geom_pos
            vert = Quaternion(body_xquat).rotate(vert) + body_xpos
            contact_verts.append(vert)
        if verbose:
            print('multicontact: ', contact_verts)
        return contact_verts


    def step(self, a):
        return np.zeros((128,128,3)), 0, 0, {}


    def render_image(self, cam):
        rgbd_frame = self.sim.render(width=self.frame_size[0], height=self.frame_size[1],
                                     mode='offscreen', camera_name=cam,
                                     depth=True)  # , device_id=self.device_id)
        img_frame = rgbd_frame[0][::-1]  # rgb: (H,W,3)

        if self.grasp_attrs_dict['depth_inp']:
            depth_frame = ((rgbd_frame[1][::-1] - self.min_depth[cam]) / (self.max_depth[cam] - self.min_depth[cam]) * 255).astype(np.uint8)  # (H,W)
            depth_frame = np.expand_dims(depth_frame, axis=-1)  # (H,W,1)

        if self.grasp_attrs_dict['use_contact']:
            contact_pos = self.get_contact_pos()
            self.sim._render_context_offscreen.add_marker(pos=contact_pos, rgba=[0.2, 0.8, 0.2, 1.0],
                                                          size=[0.02, 0.02, 0.02], label="")
        elif self.grasp_attrs_dict['use_multicontact']:
            contact_verts = self.get_multicontact_verts()
            for vert in contact_verts:
                self.sim._render_context_offscreen.add_marker(pos=vert, rgba=[0, 1, 0, 1],
                                                              size=[self.glyph_size, self.glyph_size, self.glyph_size],
                                                              label="")  # , type=const.GEOM_SPHERE)
        aff_frame_rgb = self.sim.render(width=self.frame_size[0], height=self.frame_size[1],
                                        mode='offscreen', camera_name=cam,
                                        depth=False)  # , device_id=self.device_id)
        del self.sim._render_context_offscreen._markers[:]
        aff_frame_hsv = cv2.cvtColor(aff_frame_rgb, cv2.COLOR_RGB2HSV)
        aff_frame = cv2.inRange(aff_frame_hsv, (60, 120, 120), (90, 255, 255))[::-1]
        aff_frame = np.expand_dims(aff_frame, axis=-1)  # (H,W,1)

        return img_frame, aff_frame


    def visualize_hotspots(self, image, mask):
        """Plot affordance as hotspot on top of image."""
        # print(image.dtype, mask.dtype)
        image = image.astype(np.uint8)
        # print(np.unique(mask))
        mask_rgba = cv2.merge((mask * 0, mask, mask * 0, mask))
        mask_rgba_blur = cv2.blur(mask_rgba, (3, 3))
        image_rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        result = cv2.addWeighted(image_rgba, 1.0, mask_rgba_blur, 0.5, 0)
        return result


    def generate_data(self):

        debug = self.grasp_attrs_dict['debug']
        if debug:
            ang_interval = 30
            input_dir = join(self.curr_dir, "data/contactdb/sample/%s/"%self.obj_name)
            gt_dir = join(self.curr_dir, "data/contactdb/sample/%s/"%self.obj_name)
        else:
            ang_interval = 1
            input_dir = join(self.curr_dir, "data/contactdb/input/%s/"%self.obj_name)
            gt_dir = join(self.curr_dir, "data/contactdb/gt/%s/"%self.obj_name)
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)

        if self.obj_name in ['cell_phone', 'stapler', 'teapot', 'toothpaste']:
            angles = range(0, 181, ang_interval)
        else:
            angles = range(180, 361, ang_interval)
        for angle in angles:
            quat = Quaternion(axis=[0, 0, 1], degrees=angle).elements
            self.model.body_quat[self.obj_bid] = quat
            self.sim.forward()
            img_frame, aff_frame = self.render_image(cam=self.grasp_attrs_dict['camera'])
            img_frame = cv2.cvtColor(img_frame.astype(np.float32), cv2.COLOR_BGR2RGB)
            # img_frame = self.visualize_hotspots(img_frame, aff_frame)[35:-45,45:-35,:]
            if debug:
                cv2.imwrite(join(input_dir, "%s_rgb.png"%str(angle).zfill(3)), img_frame)
                cv2.imwrite(join(gt_dir, "%s_aff.png"%str(angle).zfill(3)), aff_frame)
            else:
                cv2.imwrite(join(input_dir, "%s.png"%str(angle).zfill(3)), img_frame)
                cv2.imwrite(join(gt_dir, "%s.png"%str(angle).zfill(3)), aff_frame)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--objs', type=str, nargs='+', help='List of objects')
    parser.add_argument('--camera', type=str, default='first_person', help='Choose from [first_person, left, right, egocentric]')
    parser.add_argument('--img_res', type=int, default=128, help='Resolution of input img to cnn')
    parser.add_argument('--depth_inp', action='store_true', help='Use rgb+d input')
    parser.add_argument('--glyph_size', type=float, default=0.0075, help='Size of each contact point when rendered')
    parser.add_argument('--n_verts_multicontact', type=int, default=100, help='Number of multicontact points to render')
    parser.add_argument('--mass', type=float, default=1, help='Mass of the object')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale the object mesh vertices by this value')
    parser.add_argument('--debug', action='store_true', help='Debug mode: Save image observations')
    args = parser.parse_args()

    for obj in args.objs:
        grasp_attrs_dict = {'obj': obj,
                            'camera': args.camera,
                            'img_res': args.img_res,
                            'use_contact': False,
                            'use_multicontact': True,
                            'glyph_size': args.glyph_size,
                            'n_verts_multicontact': args.n_verts_multicontact,
                            'depth_inp': args.depth_inp,
                            'mass': args.mass,
                            'scale': args.scale,
                            'gravity': -9.81,
                            'debug': args.debug
                            }
        print(obj)
        graspv0 = GraspV0(grasp_attrs_dict=grasp_attrs_dict)
        graspv0.generate_data()
        print("%s completed!"%obj)
