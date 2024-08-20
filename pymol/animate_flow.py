import pymol.cmd as cmd
import numpy as np

cmd.set("ray_trace_fog", 0)
cmd.set("ray_shadows", 0)
cmd.set("depth_cue", 0)
cmd.bg_color("white")
cmd.set("hash_max", 300)

cmd.set("antialias", 1)
cmd.set("surface_quality", 0)
cmd.set("cartoon_transparency", 0.25)
cmd.set("light_count", 1)

cmd.set("ray_trace_mode", 1)
cmd.set("ray_opaque_background", "off")

cmd.set_color("pale_purple", [220, 208, 255])
cmd.set_color("warm_beige", [210, 180, 140])
cmd.set_color("soft_coral_red", [240, 128, 128])
cmd.set_color("natural_blue", [135, 206, 235])

cmd.set("sphere_scale", 0.25, "all")
cmd.set("valence", "on")


def load_structures(folder_path, time_steps):

    # state 2: full complex
    cmd.load(f"{folder_path}/4ffv.pdb", "4ffv", state=2)

    # state 3: full WT state
    cmd.load(f"{folder_path}/4ffv.pdb", "4ffv", state=3)
    cmd.load(f"{folder_path}/WT.pdb", "WT", state=3)

    # state 3: full WT state
    cmd.load(f"{folder_path}/WT.pdb", "WT_no_cdr", state=4)

    cmd.load(f"{folder_path}/full_complex.pdb", "full_complex")
    util.color_chains("(4ffv and elem C)", _self=cmd)
    cmd.remove("4ffv and not (chain A or chain H or chain L)")
    cmd.remove("solvent")

    # Loading the trajectory files as states
    traj_start = 5
    for i in range(traj_start, time_steps + traj_start):
        cmd.load(f"{folder_path}/WT.pdb", "WT_no_cdr", state=i)
        cmd.load(f"{folder_path}/DesAb_{i-traj_start}.pdb", "DesAb_animation", state=i)

    # ADD A FINAL CDR STATE
    final_state = time_steps + traj_start
    cmd.load(f"{folder_path}/WT.pdb", "WT_no_cdr", state=final_state)
    cmd.load(
        f"{folder_path}/DesAb_{final_state - traj_start}.pdb",
        "DesAb_final_state",
        state=final_state,
    )

    cmd.load(
        f"{folder_path}/DesAb_{final_state - traj_start}.pdb",
        "DesAb_final_state",
        state=final_state + 1,
    )

    cmd.align("4ffv", "full_complex")
    cmd.hide("everything", "all")

    cmd.show_as("cartoon", "4ffv")
    cmd.show("spheres", "WT")
    cmd.show("sticks", "WT")
    cmd.show("spheres", "WT_no_cdr")
    cmd.show("sticks", "WT_no_cdr")
    cmd.show("spheres", "DesAb_animation")
    cmd.show("spheres", "DesAb_final_state")
    cmd.show("sticks", "DesAb_final_state")

    # color the structures
    cmd.color("warm_beige", "4ffv and chain A")
    cmd.color("soft_coral_red", "4ffv and chain H")
    cmd.color("natural_blue", "4ffv and chain L")

    # state 1
    cmd.create("4ffv_no_cdr", "4ffv", 2, 1)

    # color the structures
    cmd.color("warm_beige", "WT")
    cmd.color("warm_beige", "WT_no_cdr")

    # color the cdr in pink
    cmd.color("pale_purple", "DesAb_animation")
    cmd.color("pale_purple", "DesAb_final_state")

    cmd.mset(
        f"2 x100 3 x100 4 x100 {traj_start}-{time_steps+traj_start} {final_state} x100 {final_state+1} x100"
    )

    cmd.orient()
    cmd.mplay()


# \Users\10331\OneDrive\Documents\Cam_MPhil\data\DesAb
folder_path = "/Users/10331/OneDrive/Documents/Cam_MPhil/data/desabs"
time_steps = 500

load_structures(folder_path, time_steps)
