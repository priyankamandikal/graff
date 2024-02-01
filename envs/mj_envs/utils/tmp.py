import os
import numpy as np
from pyquaternion import Quaternion
from mujoco_py import load_model_from_xml, MjSim, MjViewer

dir_path = './envs/resources/meshes/contactdb/original/pan.stl'

MODEL_XML = """
<?xml version="1.0" ?>
<mujoco>
   <asset>
        <texture builtin="flat" name="object_tex" height="32" width="32" rgb1="1 1 1" type="cube"></texture>
        <material name="object_mat" shininess="0.03" specular="0.75" texture="object_tex"></material>
       <mesh file="%s" name="mesh_name" scale="2 2 2"></mesh>
   </asset>

   <worldbody>
   
       <body name="object" pos="0 0 0">
            <geom type='mesh' pos="0 0 0" mesh="mesh_name" name="geom_name" material="object_mat"/>
       </body>

   </worldbody>
</mujoco>
""" % dir_path

class ObjModel():

    def __init__(self):
        self.model = load_model_from_xml(MODEL_XML)
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)
        self.step = 0
        self.obj_bid = self.sim.model.body_name2id('object')

        self.body_xpos = self.sim.data.body_xpos[self.obj_bid].ravel()
        self.body_xmat = self.sim.data.body_xmat[self.obj_bid].ravel()
        self.body_xmat = np.reshape(self.body_xmat, (3,3))
        self.body_xquat = self.sim.data.body_xquat[self.obj_bid].ravel()
        self.body_quat = Quaternion(self.model.body_quat[self.obj_bid])
        self.geom_xpos = self.sim.data.geom_xpos.ravel()
        self.geom_xmat = self.sim.data.geom_xmat
        self.geom_xmat = np.reshape(self.geom_xmat, (3,3))
        self.geom_pos = self.sim.model.geom_pos.ravel()
        self.geom_quat = self.sim.model.geom_quat.ravel()
        print('body_xpos: ', self.body_xpos)
        print('body_xmat: ', self.body_xmat)
        print('body_xquat: ', self.body_xquat)
        print('body_quat: ', self.body_quat)
        print('geom_xpos: ', self.geom_xpos)
        print('geom_xmat: ', self.geom_xmat)
        print('geom_pos: ', self.geom_pos)
        print('geom_quat: ', self.geom_quat)
        print('Number of vertices: ', len(self.model.mesh_vert))
        print()

    def reset_model(self):
        self.model.body_pos[self.obj_bid, 0] = np.random.uniform(low=-0.15, high=0.15)
        self.model.body_pos[self.obj_bid, 1] = np.random.uniform(low=-0.15, high=0.3)
        angle = np.random.uniform(low=0, high=360)
        quat = Quaternion(axis=[0, 0, 1], degrees=angle).elements
        self.model.body_quat[self.obj_bid] = quat
        self.sim.forward()

    def debug(self):
        cnt = 0
        while True:
            cnt += 1
            if cnt % 500 == 0:
                self.reset_model()
                # self.body_xpos = self.sim.data.body_xpos[self.obj_bid].ravel()
                # self.body_xmat = self.sim.data.body_xmat[self.obj_bid].ravel()
                # self.body_xmat = np.reshape(self.body_xmat, (3,3))
                # self.body_xquat = self.sim.data.body_xquat[self.obj_bid].ravel()
                # self.body_quat = Quaternion(self.model.body_quat[self.obj_bid])
                # self.geom_xpos = self.sim.data.geom_xpos.ravel()
                # self.geom_xmat = self.sim.data.geom_xmat
                # self.geom_xmat = np.reshape(self.geom_xmat, (3,3))
                # self.geom_pos = self.sim.model.geom_pos.ravel()
                # self.geom_quat = self.sim.model.geom_quat.ravel()
                print('body_xpos: ', self.body_xpos)
                print('body_xmat: ', self.body_xmat)
                print('body_xquat: ', self.body_xquat)
                print('body_quat: ', self.body_quat)
                print('geom_xpos: ', self.geom_xpos)
                print('geom_xmat: ', self.geom_xmat)
                print('geom_pos: ', self.geom_pos)
                print('geom_quat: ', self.geom_quat)
                print('Number of vertices: ', len(self.model.mesh_vert))
                print()

            # for vert in self.model.mesh_vert[:10]:
            #     # viewer.add_marker(pos=vert, rgba=[0.0, 0.5, 0.5, 1.0],
            #     #                   size=[0.003, 0.003, 0.003])
            #     vert_repos = np.dot(vert, self.geom_xmat_reshaped) + self.geom_xpos
            #     self.viewer.add_marker(pos=vert_repos, rgba=[0.0, 0.8, 0.2, 1.0],
            #                       size=[0.003, 0.003, 0.003],
            #                       label="")

            for ind in [82616, 82617, 82619, 82620, 82632, 82633, 83868, 83881]:
                vert = self.model.mesh_vert[ind]
                vert = Quaternion(self.geom_quat).rotate(vert) + self.geom_pos
                vert = Quaternion(self.body_xquat).rotate(vert) + self.body_xpos
                self.viewer.add_marker(pos=vert, rgba=[0.8, 0.2, 0.2, 1.0],
                                  size=[0.01, 0.01, 0.01],
                                  label="")
            self.viewer.render()


if __name__ == '__main__':
    objmodel = ObjModel()
    objmodel.debug()
