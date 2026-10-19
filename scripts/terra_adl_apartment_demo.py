#!/usr/bin/env python3
"""Run TERRA-4B continuously through a furnished split-level ADL apartment."""
from __future__ import annotations
import json, os
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as imageio
import mujoco
import numpy as np
from myosuite import make_env
from myosuite.envs.waypoint import WaypointTaskCfg
from myosuite.integrations.musclemimic import GaitParameters, MIMIC_SITES, TerrainController, TerrainHeights, build_terrain_fullbody_spec, download_terrain_checkpoint, load_terrain_policy, plan_waypoint_reference
from myosuite.integrations.musclemimic.adl_apartment import adl_apartment_scene, adl_route
from myosuite.terms.waypoint import WaypointRewardCfg
from myosuite.viz.bone_ghost_viz import BoneGhostBodyViz


def main() -> None:
    seed=int(os.environ.get("TERRA_SEED","0")); stochastic=os.environ.get("TERRA_STOCHASTIC","1")!="0"
    out=Path(os.environ.get("TERRA_ADL_OUT","terra_adl_outputs")); out.mkdir(parents=True,exist_ok=True)
    route=adl_route(); model=build_terrain_fullbody_spec(adl_apartment_scene).compile(); probe=mujoco.MjData(model); mujoco.mj_forward(model,probe)
    heights=TerrainHeights(model,top=12.0); height=lambda xy: float(heights(probe,[xy])[0]); gait=GaitParameters(speed=0.25,cycle_s=1.2,clearance=0.12)
    q0=model.qpos0.copy(); q0[:2]=route[0]
    checkpoint=Path(os.environ["TERRA_CHECKPOINT"]) if "TERRA_CHECKPOINT" in os.environ else download_terrain_checkpoint()
    task=WaypointTaskCfg(site_name="pelvis_mimic",waypoints=tuple(map(tuple,route[1:])),arrival_radius=0.65,min_site_height=0.18,reward=WaypointRewardCfg())
    env=make_env("myoFullBodyWaypoint-v0",model=model,task=task,initial_qpos=q0,initial_qvel=np.zeros(model.nv),frame_skip=5,render_mode="rgb_array"); env.reset(seed=seed); base=env.unwrapped

    # TERRA is stochastic.  Instead of committing to one sample for 130 s, keep a
    # tiny ensemble of genuine TERRA policies and select the action whose short
    # MuJoCo rollout best preserves posture while progressing to the next ADL goal.
    # This is receding-horizon action selection, not state reset/teleportation.
    def make_controllers(next_idx:int):
        start=base.data.qpos[:2].copy(); targets=route[1+next_idx:min(len(route),1+next_idx+3)]
        local=np.vstack([start,targets])
        ref,diag=plan_waypoint_reference(model,local,height,base.data.qpos.copy(),sites=MIMIC_SITES,parameters=gait)
        controllers=[]
        for k in range(6):
            pol=load_terrain_policy(checkpoint,model,ref,stochastic=True,seed=seed+97*next_idx+k)
            ctl=TerrainController(pol,model,ref,frame_skip=5); ctl.reset(); controllers.append(ctl)
        return controllers,ref,diag

    controllers,reference,planner=make_controllers(0); replans=0
    option=mujoco.MjvOption(); option.geomgroup[:]=0; option.geomgroup[0]=option.geomgroup[2]=option.geomgroup[3]=1; option.sitegroup[:]=0; option.tendongroup[:]=1
    ghost=BoneGhostBodyViz.from_motion_clip(reference,model,bone_meshes=True,skeleton_rgba=(0.20,0.70,1.0,0.28))
    frames=[]; info={"next_waypoint":0,"solved":False,"failed":False}; min_h=float("inf"); previous_wp=0; max_steps=20000
    trial=mujoco.MjData(model); pelvis_id=model.site("pelvis_mimic").id

    def choose_action(target_xy):
        best=None; best_score=-np.inf
        for ctl in controllers:
            action=ctl(base.data)
            mujoco.mj_copyData(trial,model,base.data)
            lo,hi=model.actuator_ctrlrange.T; trial.ctrl[:]=np.clip(action,lo,hi)
            mujoco.mj_step(model,trial,5)
            pelvis=trial.site_xpos[pelvis_id]
            # Strong fall avoidance, then route progress; discourage explosive states.
            dist=np.linalg.norm(pelvis[:2]-target_xy)
            height_pen=max(0.0,0.72-pelvis[2])
            speed_pen=max(0.0,np.linalg.norm(trial.qvel[:3])-2.0)
            score=-dist-8.0*height_pen-0.15*speed_pen
            if np.isfinite(score) and score>best_score: best_score,best=score,action
        return best

    with mujoco.Renderer(model,360,640,max_geom=model.ngeom+1024) as renderer:
        for step in range(max_steps):
            target=route[min(1+previous_wp,len(route)-1)]
            action=choose_action(target)
            _,_,terminated,truncated,info=base.step(action); min_h=min(min_h,float(base.data.site("pelvis_mimic").xpos[2]))
            reached=int(info["next_waypoint"])
            if reached>previous_wp and not info["solved"]:
                previous_wp=reached; controllers,reference,planner=make_controllers(reached); replans+=1
                ghost=BoneGhostBodyViz.from_motion_clip(reference,model,bone_meshes=True,skeleton_rgba=(0.20,0.70,1.0,0.28))
            if step%10==0:
                cam=mujoco.MjvCamera(); cam.lookat[:]=base.data.qpos[:3]; cam.lookat[2]+=0.35; cam.distance=3.2; cam.azimuth=150; cam.elevation=-12
                renderer.update_scene(base.data,camera=cam,scene_option=option); ghost.draw(min(controllers[0]._frame+50,reference.num_frames-1),renderer.scene); frames.append(renderer.render().copy())
            if info["solved"] or terminated or truncated: break
    final=base.data.qpos[:3].copy(); env.close(); video=out/"terra_adl_apartment.mp4"; preview=out/"terra_adl_apartment_preview.jpg"
    imageio.mimsave(video,frames,fps=10,macro_block_size=1); imageio.imwrite(preview,frames[len(frames)//2],quality=92)
    result={"seed":seed,"controller":"terra_stochastic_ensemble_mpc","ensemble":6,"executed_steps":step+1,"waypoints_reached":int(info["next_waypoint"]),"waypoints_total":len(route)-1,"replans":replans,"solved":bool(info["solved"]),"failed":bool(info["failed"]),"min_pelvis_height":min_h,"final_root_xyz":final.tolist(),"video":str(video),"preview":str(preview)}
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
