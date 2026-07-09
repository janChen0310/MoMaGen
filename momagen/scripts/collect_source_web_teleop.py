# NOTE: imported from the TidyBot dispose-trash pipeline (see docs/tutorials/tidybot-task-pipelines.md).
# Server-specific absolute paths (e.g. dataset/template locations) may need adjusting to your setup.
"""Offscreen browser-streamed teleop for the TidyBot dispose-trash source demo (headless GPU box).
Loads the sampled house_single_floor datagen_picking_up_trash instance (soda can on a kitchen
countertop, trash can on the floor). Drive: grasp the can, carry it to the trash can, drop it in, C=save.
Renders offscreen, serves an MJPEG stream + 20Hz web keyboard input over HTTP."""
import os, io, math, time, threading, collections
os.environ["OMNIGIBSON_HEADLESS"] = "1"; os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ.setdefault("OMNIGIBSON_GPU_ID", "0")
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np, torch as th
try:
    from PIL import Image
    def enc(a):
        b = io.BytesIO(); Image.fromarray(a).save(b, format="JPEG", quality=55); return b.getvalue()
except Exception:
    import cv2
    def enc(a):
        return cv2.imencode(".jpg", a[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 55])[1].tobytes()
import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.envs import DataCollectionWrapper
from omnigibson.macros import gm
from momagen.scripts.collect_tidybot_source_demo import (
    load_tidybot_env_config, CartesianTeleop, POS_STEP, ORI_STEP, BASE_LIN, BASE_ANG)

def _T(x):
    if isinstance(x, th.Tensor):
        return x.detach().to("cpu", th.float32)
    return th.as_tensor(np.asarray(x), dtype=th.float32)

class WebTeleop(CartesianTeleop):
    """Type-safe DLS-IK (control_dict comes back numpy on this env -> cast to CPU torch)."""
    def action(self):
        robot = self.robot
        cd = robot.get_control_dict()
        ac = robot.controllers["arm_" + self.arm]
        di = ac.dof_idx
        q = _T(cd["joint_position"])[di]
        j_eef = _T(cd["eef_%s_jacobian_relative" % self.arm])[:, di]
        pos_rel, quat_rel = robot.get_relative_eef_pose(self.arm)
        dpos = _T(self.target_pos) - _T(pos_rel)
        dori = _T(T.orientation_error(T.quat2mat(_T(self.target_quat)), T.quat2mat(_T(quat_rel))))
        err = th.cat([dpos, dori])
        JT = j_eef.T
        dq = JT @ th.linalg.solve(j_eef @ JT + 1e-4 * th.eye(6), err)
        target_q = q + th.clamp(dq, -0.05, 0.05)
        action = th.zeros(robot.action_dim)
        action[robot.controller_action_idx["arm_" + self.arm]] = _T(ac._reverse_preprocess_command(target_q))
        action[robot.controller_action_idx["base"]] = _T(self.base_cmd)
        action[robot.controller_action_idx["gripper_" + self.arm]] = -1.0 if self.gripper_closed else 1.0
        self.base_cmd = self.base_cmd * 0.9
        return action

REPO = "/root/MoMaGen"
TEMPLATE = REPO + "/momagen/datasets/source_og/r1_picking_up_trash.hdf5"
OUTPUT = REPO + "/momagen/datasets/source_og/tidybot_picking_up_trash.hdf5"
SCENE_INSTANCE = "house_single_floor_task_datagen_picking_up_trash_0_0_template"
CAN = "can_of_soda_595"; TRASH = "trash_can_596"; PORT = 8890
def arr(x): return np.array(x.cpu() if hasattr(x, "cpu") else x, float)

cfg = load_tidybot_env_config(TEMPLATE)
cfg["scene"]["scene_model"] = "house_single_floor"
cfg["scene"]["scene_instance"] = SCENE_INSTANCE
cfg["scene"]["scene_file"] = None
cfg["robots"][0]["grasping_mode"] = "physical"
# The r1_picking_up_trash template carries self_collisions=True (unlike pick_cup/tidy_table).
# TidyBot's imported collision meshes overlap slightly between adjacent links, so enabling
# self-collision makes the robot vibrate at rest and swamps base commands -> force it off.
cfg["robots"][0]["self_collisions"] = False
gm.ENABLE_TRANSITION_RULES = False
print("building trash env...", flush=True)
env = og.Environment(configs=cfg)
env = DataCollectionWrapper(env=env, output_path=OUTPUT, only_successes=False)
robot = env.robots[0]; env.reset()
_uq = robot.get_joint_positions(); untuck_q = _uq.clone() if hasattr(_uq, "clone") else np.array(_uq).copy()
for _ in range(40): og.sim.step()
# hide ceilings/roof so the overhead chase camera can see the interior
from pxr import UsdGeom as _UG
for _obj in env.scene.objects:
    if (getattr(_obj, "category", "") or "").lower() in ("ceilings", "roof"):
        _pr = getattr(_obj, "prim", None)
        if _pr is not None:
            try: _UG.Imageable(_pr).MakeInvisible()
            except Exception: pass
for _ in range(3): og.sim.step()
can = env.scene.object_registry("name", CAN); trash = env.scene.object_registry("name", TRASH)
cpos = arr(can.get_position_orientation()[0]); tpos = arr(trash.get_position_orientation()[0])
# --- NO pre-staging: start from the sampler-validated BDDL spawn. ---
# Teleporting this holonomic robot (set_position_orientation) during a placement search left
# the articulation in a corrupted state (virtual base joints w/ meter-scale offsets, chassis
# sunk into the floor, base drives fighting) -> continuous oscillation + uncontrollable base.
# A/B probe proved the untouched BDDL spawn is healthy: idle maxjv=0.00, base drives at 0.575 m/s.
# The user simply drives from the spawn to the counter (~4.7 m) -- a richer source demo anyway.
robot_path = robot.prim_path
def _nonfloor_hits():
    hits = set()
    try:
        for c in robot.contact_list():
            for b in (c.body0, c.body1):
                if not b or robot_path in b: continue
                bl = b.lower()
                if "floor" in bl or "ground" in bl: continue
                hits.add(b.split("/")[-1])
    except Exception as ex:
        print("contact err", ex, flush=True)
    return hits
robot.keep_still()
bpos = arr(robot.get_position_orientation()[0])
teleop = WebTeleop(robot)
print("READY can=%s trash=%s base=%s" % (np.round(cpos, 2).tolist(), np.round(tpos, 2).tolist(), np.round(bpos, 2).tolist()), flush=True)

import omni.replicator.core as rep
from pxr import UsdGeom, Gf
stg = og.sim.stage
cam = UsdGeom.Camera.Define(stg, "/World/jc_cam"); cam.GetFocalLengthAttr().Set(16.0)
cam.GetHorizontalApertureAttr().Set(24.0); cam.GetClippingRangeAttr().Set((0.02, 200.0))
camxf = UsdGeom.Xformable(cam.GetPrim()).AddTransformOp()
def yaw_of(q):
    x, y, z, w = q; return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
# orbit camera around the robot; user adjusts az/el/dist live (keys 1-6). az is relative
# to the robot heading (0 = directly behind), so the view tracks the robot as it drives.
cam_state = {"az": 0.0, "el": 73.0, "dist": 3.45, "h": 0.55}
def set_cam():
    p, q = robot.get_position_orientation()
    bp = arr(p); yaw = yaw_of(arr(q))
    T = np.array([bp[0], bp[1], cam_state["h"]])
    azw = yaw + math.pi + math.radians(cam_state["az"])
    el = math.radians(max(8.0, min(86.0, cam_state["el"])))
    dist = max(1.2, min(7.0, cam_state["dist"]))
    eye = T + dist * np.array([math.cos(el) * math.cos(azw), math.cos(el) * math.sin(azw), math.sin(el)])
    up = np.array([0, 0, 1.]); f = T - eye; f /= np.linalg.norm(f)
    rr = np.cross(f, up); rr /= np.linalg.norm(rr); uu = np.cross(rr, f)
    M = np.eye(4); M[:3, 0] = rr; M[:3, 1] = uu; M[:3, 2] = -f; M[:3, 3] = eye
    camxf.Set(Gf.Matrix4d(*M.T.flatten().tolist()))
set_cam()
rp = rep.create.render_product("/World/jc_cam", (640, 384))
ann = rep.AnnotatorRegistry.get_annotator("rgb"); ann.attach([rp])

latest = [None]; fid = [0]; key_queue = collections.deque()
def process_key(k):
    k = k.lower(); B = BASE_LIN; A = BASE_ANG; P = POS_STEP; O = ORI_STEP
    base = {'w': [B,0,0], 's': [-B,0,0], 'a': [0,B,0], 'd': [0,-B,0], 'q': [0,0,A], 'e': [0,0,-A]}
    npos = {'i': [P,0,0], 'k': [-P,0,0], 'j': [0,P,0], 'l': [0,-P,0], 'u': [0,0,P], 'o': [0,0,-P]}
    nori = {'t': [0,O,0], 'g': [0,-O,0], 'f': [0,0,O], 'h': [0,0,-O], 'r': [O,0,0], 'y': [-O,0,0]}
    if k in base: teleop.base_cmd = th.tensor([float(x) for x in base[k]])
    elif k in npos: teleop.nudge(dpos=npos[k])
    elif k in nori: teleop.nudge(dori=nori[k])
    elif k == '1': cam_state["az"] -= 7
    elif k == '2': cam_state["az"] += 7
    elif k == '3': cam_state["el"] = max(8.0, cam_state["el"] - 4)
    elif k == '4': cam_state["el"] = min(86.0, cam_state["el"] + 4)
    elif k == '5': cam_state["dist"] = max(1.2, cam_state["dist"] - 0.3)
    elif k == '6': cam_state["dist"] = min(7.0, cam_state["dist"] + 0.3)
    elif k in (' ', 'spacebar', 'space'): teleop.gripper_closed = not teleop.gripper_closed
    elif k == 'c': teleop.done = True

HTML = """<!doctype html><html><head><title>TidyBot Trash Teleop</title><meta charset=utf-8>
<style>body{background:#1a1a1a;color:#ddd;font-family:monospace;text-align:center}
img{border:2px solid #444;max-width:98vw}kbd{background:#333;padding:1px 5px;border-radius:3px}</style></head>
<body><h3>TidyBot dispose-trash source demo &mdash; click here, then use keys</h3><img src="/stream"><br>
<p>Grasp the soda can, carry it to the trash can, drop it in. Base <kbd>W</kbd><kbd>S</kbd><kbd>A</kbd><kbd>D</kbd> <kbd>Q</kbd><kbd>E</kbd> | Arm <kbd>I</kbd><kbd>K</kbd><kbd>J</kbd><kbd>L</kbd><kbd>U</kbd><kbd>O</kbd> xyz, <kbd>T</kbd><kbd>G</kbd><kbd>F</kbd><kbd>H</kbd><kbd>R</kbd><kbd>Y</kbd> rot | <kbd>SPACE</kbd> grip | Cam <kbd>1</kbd><kbd>2</kbd> orbit <kbd>3</kbd><kbd>4</kbd> tilt <kbd>5</kbd><kbd>6</kbd> zoom | <kbd>C</kbd> save+exit</p>
<p id=s>ready</p><script>
var CONT="wsadqeijkluotgfhry123456", held={};
document.addEventListener('keydown',function(e){var k=e.key.toLowerCase();
 if(k===' '||k==='c'){if(!e.repeat)fetch('/key?k='+encodeURIComponent(e.key));}
 else if(CONT.indexOf(k)>=0){held[k]=1; if(!e.repeat)fetch('/key?k='+k);}
 document.getElementById('s').innerText='key: '+e.key;
 if([' ','ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(e.key))e.preventDefault();});
document.addEventListener('keyup',function(e){delete held[e.key.toLowerCase()];});
setInterval(function(){for(var k in held)fetch('/key?k='+k);},50);
</script></body></html>"""
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path.startswith("/key"):
            k = parse_qs(urlparse(self.path).query).get("k", [""])[0]
            if k: key_queue.append(k)
            self.send_response(204); self.end_headers()
        elif self.path == "/stream":
            self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store"); self.end_headers()
            last = -1
            try:
                while True:
                    if fid[0] != last and latest[0]:
                        last = fid[0]
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"); self.wfile.write(latest[0]); self.wfile.write(b"\r\n")
                    else:
                        time.sleep(0.004)
            except Exception: pass
        else:
            self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers(); self.wfile.write(HTML.encode())
# --- close-up probe: eye-level stills of the arm vs the counter + raw link coordinates, then exit ---
if os.environ.get("JC_CLOSEUP"):
    for _ in range(30): env.step(teleop.action())  # let controller settle the true teleop pose
    print("== ARM LINK WORLD POSITIONS ==", flush=True)
    for _ln, _lk in robot.links.items():
        if any(k in _ln.lower() for k in ("arm", "wrist", "forearm", "bracelet", "end_effector", "gripper", "finger", "tool")):
            print("LINK %-28s %s" % (_ln, np.round(arr(_lk.get_position_orientation()[0]), 3).tolist()), flush=True)
    ctr = np.array([bpos[0], bpos[1], 0.75])
    for i, azd in enumerate((0, 90, 200)):
        azr = math.radians(azd)
        eye = ctr + np.array([2.0 * math.cos(azr), 2.0 * math.sin(azr), 0.35])
        up = np.array([0, 0, 1.]); f = ctr - eye; f /= np.linalg.norm(f)
        rr = np.cross(f, up); rr /= np.linalg.norm(rr); uu = np.cross(rr, f)
        M = np.eye(4); M[:3, 0] = rr; M[:3, 1] = uu; M[:3, 2] = -f; M[:3, 3] = eye
        camxf.Set(Gf.Matrix4d(*M.T.flatten().tolist()))
        for _ in range(10): og.sim.render()
        d = ann.get_data()
        if d is not None:
            from PIL import Image as _Im
            _Im.fromarray(np.array(d)[:, :, :3].astype(np.uint8)).save("/root/rivermind-data/closeup_%d.jpg" % i)
            print("CLOSEUP_SAVED %d az=%d" % (i, azd), flush=True)
    print("CLOSEUP_DONE", flush=True)
    og.shutdown(); raise SystemExit

# --- joint probe: dump what PhysX actually has for the base joints, then a command test ---
if os.environ.get("JC_JOINTPROBE"):
    print("== ROBOT n_dof=%d ==" % robot.n_dof, flush=True)
    try: print("dof_names_ordered:", list(robot.dof_names_ordered), flush=True)
    except Exception as e: print("dofnames err", e, flush=True)
    bc = robot.controllers["base"]
    print("base_controller dof_idx:", bc.dof_idx.tolist(), "control_type:", bc.control_type, flush=True)
    print("base isaac_kd:", getattr(bc, "_isaac_kd", None), flush=True)
    for jn in ("base_footprint_x_joint", "base_footprint_y_joint", "base_footprint_rz_joint", "joint_1"):
        try:
            j = robot.joints[jn]
            print("JOINT %-26s dof_indices=%s driven=%s stiff=%s damp=%s max_effort=%s" % (
                jn, j.dof_indices.tolist() if hasattr(j.dof_indices, "tolist") else j.dof_indices,
                j.driven, float(j.stiffness), float(j.damping), float(j.max_effort)), flush=True)
        except Exception as e:
            print("JOINT %s ERR %s" % (jn, e), flush=True)
    # chassis ride height + UNFILTERED contacts (floor included!) + virtual joint limits
    try:
        chname = robot.base_footprint_link_name
        ch = robot.links[chname]
        chp = arr(ch.get_position_orientation()[0])
        print("CHASSIS link=%s world_z=%.4f (expect ~+0.05 caster lift)" % (chname, chp[2]), flush=True)
        try:
            print("CHASSIS aabb z: center=%.4f ext=%.4f -> bottom=%.4f" % (
                float(arr(ch.aabb_center)[2]), float(arr(ch.aabb_extent)[2]),
                float(arr(ch.aabb_center)[2] - arr(ch.aabb_extent)[2] / 2)), flush=True)
        except Exception as e: print("aabb err", e, flush=True)
    except Exception as e: print("chassis err", e, flush=True)
    og.sim.step()
    allc = {}
    for c in robot.contact_list():
        for b in (c.body0, c.body1):
            if b and robot.prim_path not in b:
                allc[b.split("/")[-1]] = allc.get(b.split("/")[-1], 0) + 1
    print("ALL_CONTACTS (unfiltered):", allc, flush=True)
    for jn in ("base_footprint_x_joint", "base_footprint_y_joint", "base_footprint_rz_joint"):
        try:
            j = robot.joints[jn]
            print("LIMITS %-26s lower=%s upper=%s" % (jn, j.lower_limit, j.upper_limit), flush=True)
        except Exception as e: print("LIMITS %s ERR %s" % (jn, e), flush=True)
    # command test: hold forward for 40 steps, watch base + dof velocities
    p0 = arr(robot.get_position_orientation()[0])
    for n in range(40):
        teleop.base_cmd = th.tensor([0.4, 0.0, 0.0])
        env.step(teleop.action())
        if n % 10 == 0:
            jp = arr(robot.get_joint_positions()); jvv = arr(robot.get_joint_velocities())
            bidx = bc.dof_idx.tolist()
            print("CMDTEST n=%d base_dq=%s base_dv=%s pos_delta=%.3f" % (
                n, [round(float(jp[i]), 3) for i in bidx], [round(float(jvv[i]), 3) for i in bidx],
                float(np.linalg.norm(arr(robot.get_position_orientation()[0])[:2] - p0[:2]))), flush=True)
    print("JOINTPROBE_DONE", flush=True)
    og.shutdown(); raise SystemExit

# --- scripted-drive diagnostic: inject key input like a user session, record video + state log ---
if os.environ.get("JC_DIAG_DRIVE"):
    def yaw_of_q(q):
        x, y, z, w = q; return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    # (phase_name, n_steps, key, key_every): mimic the 20Hz browser repeat (~every 2-3 sim steps)
    SCRIPT = [("idle0", 50, None, 0), ("fwd", 70, 'w', 2), ("idle1", 50, None, 0),
              ("rot", 50, 'q', 2), ("idle2", 40, None, 0), ("arm_x", 60, 'i', 3),
              ("arm_z", 40, 'u', 3), ("idle3", 80, None, 0)]
    frames = []; step_i = 0
    for phname, nst, key, kev in SCRIPT:
        for j in range(nst):
            if key and kev and j % kev == 0: process_key(key)
            cam_state["az"] = 25.0
            set_cam()
            env.step(teleop.action())
            data = ann.get_data()
            if data is not None:
                im = np.array(data)
                if im.ndim == 3 and im.shape[-1] >= 3: frames.append(im[:, :, :3].astype(np.uint8))
            if step_i % 10 == 0:
                p, q = robot.get_position_orientation()
                bp = arr(p); yw = yaw_of_q(arr(q))
                eef = arr(robot.get_relative_eef_pose(teleop.arm)[0])
                jv = arr(robot.get_joint_velocities()); mv = float(np.abs(jv).max())
                hits = sorted(_nonfloor_hits())[:3]
                print("DRV %-6s n=%03d base=[%.2f,%.2f] yaw=%.2f eef=[%.2f,%.2f,%.2f] maxjv=%.2f hits=%s" % (
                    phname, step_i, bp[0], bp[1], yw, eef[0], eef[1], eef[2], mv, hits), flush=True)
            step_i += 1
    out = "/root/rivermind-data/trash_drive.mp4"
    try:
        import imageio; imageio.mimsave(out, frames[::2], fps=15, macro_block_size=1)
    except Exception as ex:
        print("mp4 failed", ex, flush=True)
    print("DIAG_DRIVE_SAVED %s frames=%d" % (out, len(frames)), flush=True)
    og.shutdown(); raise SystemExit

# --- diagnostic video mode: record the sim from t=0 while orbiting the camera, then exit ---
if os.environ.get("JC_DIAG_VIDEO"):
    frames = []
    NF = 360
    print("DIAG_VIDEO start (%d frames)..." % NF, flush=True)
    for n in range(NF):
        cam_state["az"] = (n * 360.0 / NF) - 180.0  # full orbit relative to robot heading
        set_cam()
        env.step(teleop.action())
        data = ann.get_data()
        if data is not None:
            im = np.array(data)
            if im.ndim == 3 and im.shape[-1] >= 3:
                frames.append(im[:, :, :3].astype(np.uint8))
        if n in (20, 60, 200):
            print("DIAG_HITS n=%d %s" % (n, sorted(_nonfloor_hits())[:8]), flush=True)
    out = "/root/rivermind-data/trash_diag.mp4"
    try:
        import imageio
        imageio.mimsave(out, frames[::2], fps=15, macro_block_size=1)
    except Exception as ex:
        print("mp4 failed (%s), writing gif" % ex, flush=True)
        out = "/root/rivermind-data/trash_diag.gif"
        from PIL import Image as _Im
        _fr = [_Im.fromarray(f) for f in frames[::4]]
        _fr[0].save(out, save_all=True, append_images=_fr[1:], duration=120, loop=0)
    print("DIAG_VIDEO_SAVED %s frames=%d" % (out, len(frames)), flush=True)
    og.shutdown(); raise SystemExit

srv = ThreadingHTTPServer(("0.0.0.0", PORT), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
print("HTTP_SERVER_UP port %d" % PORT, flush=True)

n = 0
while not teleop.done:
    while key_queue:
        try: process_key(key_queue.popleft())
        except IndexError: break
        except Exception as ex: print("key err", ex, flush=True)
    set_cam()
    env.step(teleop.action()); n += 1
    data = ann.get_data()
    if data is not None:
        im = np.array(data)
        if im.ndim == 3 and im.shape[-1] >= 3:
            latest[0] = enc(im[:, :, :3].astype(np.uint8)); fid[0] += 1
    if n == 4: print("FIRST_FRAMES_RENDERED bytes=%s" % (len(latest[0]) if latest[0] else 0), flush=True)
    if n in (40, 100, 180) and latest[0]:
        open("/root/rivermind-data/trash_frame.jpg", "wb").write(latest[0]); print("SAVED_FRAME n=%d" % n, flush=True)
    if n in (20, 60):
        try:
            for c in robot.contact_list():
                b0 = "/".join((c.body0 or "?").split("/")[-2:]); b1 = "/".join((c.body1 or "?").split("/")[-2:])
                if not any(k in (b0 + b1).lower() for k in ("floor", "ground")):
                    print("LIVE_CONTACT n=%d %s <-> %s" % (n, b0, b1), flush=True)
            print("LIVE_HITS n=%d %s" % (n, sorted(_nonfloor_hits())[:8]), flush=True)
        except Exception as ex: print("lc err", ex, flush=True)
    if n % 300 == 0:
        try: print("progress n=%d success=%s" % (n, env.task.success), flush=True)
        except Exception: pass
print("saving demo...", flush=True); env.save_data()
try: print("FINAL_SUCCESS=%s" % env.task.success, flush=True)
except Exception: pass
import h5py
with h5py.File(OUTPUT, "r+") as f:
    d = sorted(f["data"].keys())
    if "mask" not in f: f.create_group("mask")
    if "use" in f["mask"]: del f["mask"]["use"]
    f["mask"].create_dataset("use", data=np.array([d[-1].encode()]))
print("SAVED %s" % OUTPUT, flush=True); srv.shutdown(); og.shutdown()
