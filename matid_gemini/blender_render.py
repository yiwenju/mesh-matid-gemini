"""Run inside Blender: load a mesh, normalize it, and render clay views.

    blender -b --factory-startup --python blender_render.py -- \
        --mesh model.glb --out DIR --cameras DIR/cameras.json

For a .blend, pass it as the file to open instead of --mesh:

    blender -b model.blend --python blender_render.py -- --out DIR --cameras ...

Frames. Blender world is Z-up. Everything written to disk is in the Y-up frame
(p_y_up = (x, z, -y) of the Blender point), then normalized so the bounding-box
center is the origin and the farthest vertex is at radius 1. cameras.json is in
that normalized frame (see cameras.py); this script maps it back to Blender
world to render. Importers already convert OBJ/glTF/PLY/STL/FBX Y-up files to
Blender's Z-up, so a Y-up input mesh keeps its orientation.

Writes mesh.npz (verts, faces, face_mat, mat_names, center, scale) and one
RGBA png per camera (transparent background).
"""

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

# Blender world -> Y-up frame, (x, y, z) -> (x, z, -y). A proper rotation.
B2O = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
NO_MATERIAL = "<none>"


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=Path, help="mesh file to import (omit for a .blend opened on the command line)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cameras", type=Path, required=True)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument(
        "--shading",
        choices=["clay", "textured"],
        default="clay",
        help="clay: one neutral gray material on everything (geometry only); "
        "textured: the file's own materials",
    )
    return p.parse_args(argv)


def import_mesh(path: Path):
    for obj in list(bpy.context.scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    ext = path.suffix.lower()
    if ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif ext in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=str(path))
    elif ext == ".stl":
        bpy.ops.wm.stl_import(filepath=str(path))
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    else:
        raise ValueError(f"unsupported mesh format {ext}; use obj, glb, gltf, ply, stl, fbx, or blend")


def collect_mesh():
    """Merge every renderable mesh instance into world-space triangles."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    verts, faces, face_mat, mat_names = [], [], [], []
    mat_index = {}
    offset = 0
    for inst in depsgraph.object_instances:
        obj = inst.object
        if obj.type != "MESH" or obj.hide_render:
            continue
        mesh = obj.to_mesh()
        if len(mesh.polygons) == 0:
            obj.to_mesh_clear()
            continue
        mesh.calc_loop_triangles()
        m = np.array(inst.matrix_world, dtype=np.float64)
        v = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", v)
        v = v.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
        tri = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
        mesh.loop_triangles.foreach_get("vertices", tri)
        tri_poly = np.empty(len(mesh.loop_triangles), dtype=np.int64)
        mesh.loop_triangles.foreach_get("polygon_index", tri_poly)
        poly_slot = np.empty(len(mesh.polygons), dtype=np.int64)
        mesh.polygons.foreach_get("material_index", poly_slot)
        slot_names = [s.material.name if s.material else NO_MATERIAL for s in obj.material_slots]
        slot_names = slot_names or [NO_MATERIAL]
        slot_ids = []
        for name in slot_names:
            if name not in mat_index:
                mat_index[name] = len(mat_names)
                mat_names.append(name)
            slot_ids.append(mat_index[name])
        slot_ids = np.array(slot_ids)
        slot = np.clip(poly_slot[tri_poly], 0, len(slot_ids) - 1)
        verts.append(v)
        faces.append(tri.reshape(-1, 3) + offset)
        face_mat.append(slot_ids[slot])
        offset += len(v)
        obj.to_mesh_clear()
    if not faces:
        raise RuntimeError("no renderable mesh in scene")
    return np.concatenate(verts), np.concatenate(faces), np.concatenate(face_mat), mat_names


def normalize(v_blender):
    v = v_blender @ B2O.T
    center = (v.min(0) + v.max(0)) / 2
    scale = 1.0 / np.linalg.norm(v - center, axis=1).max()
    return (v - center) * scale, center, scale


def enable_gpu():
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = kind
            prefs.get_devices()
        except Exception:
            continue
        gpus = [d for d in prefs.devices if d.type == kind]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == kind
            bpy.context.scene.cycles.device = "GPU"
            print(f"render device: {kind} x{len(gpus)}")
            return
    print("render device: CPU (no GPU found)")


def setup_scene(samples, res, shading):
    scene = bpy.context.scene
    for obj in list(scene.objects):
        if obj.type in {"LIGHT", "CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    enable_gpu()
    world = bpy.data.worlds.new("flat")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    scene.world = world
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 2.0
    sun.rotation_euler = (0.9, 0.2, 0.6)
    scene.collection.objects.link(sun)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    scene.collection.objects.link(cam)
    scene.camera = cam
    if shading == "clay":
        clay = bpy.data.materials.new("clay")
        clay.use_nodes = True
        bsdf = clay.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.55, 0.55, 0.55, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.8
        bsdf.inputs["Metallic"].default_value = 0.0
        bpy.context.view_layer.material_override = clay
    return scene, cam


def place_camera(cam, view, center, scale):
    """Set a Blender camera from an OpenCV-convention c2w in the normalized frame."""
    c2w = np.array(view["c2w"], dtype=np.float64)
    # Undo normalization: only the translation carries scale, rotation is unchanged.
    R_obj = c2w[:3, :3]
    t_obj = c2w[:3, 3] / scale + center
    R_bl = B2O.T @ R_obj
    t_bl = B2O.T @ t_obj
    # OpenCV camera looks +Z with +Y down; Blender looks -Z with +Y up.
    R_bl = R_bl @ np.diag([1.0, -1.0, -1.0])
    mat = np.eye(4)
    mat[:3, :3] = R_bl
    mat[:3, 3] = t_bl
    cam.matrix_world = Matrix(mat.tolist())
    K = np.array(view["K"], dtype=np.float64)
    cam.data.sensor_fit = "HORIZONTAL"
    cam.data.sensor_width = 36.0
    cam.data.lens = float(K[0, 0] * 36.0 / view["width"])
    cam.data.clip_start = 0.01 / scale
    cam.data.clip_end = 100.0 / scale


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cameras = json.loads(args.cameras.read_text())
    if args.mesh is not None:
        import_mesh(args.mesh)

    v_bl, faces, face_mat, mat_names = collect_mesh()
    verts, center, scale = normalize(v_bl)
    np.savez(
        args.out / "mesh.npz",
        verts=verts.astype(np.float32),
        faces=faces.astype(np.int32),
        face_mat=face_mat.astype(np.int32),
        mat_names=np.array(mat_names),
        center=center,
        scale=np.array(scale),
    )
    print(f"mesh: {len(verts)} verts, {len(faces)} tris, {len(mat_names)} materials")

    scene, cam = setup_scene(args.samples, cameras["views"][0]["width"], args.shading)
    for view in cameras["views"]:
        place_camera(cam, view, center, scale)
        scene.render.filepath = str(args.out / view["image_rgba"])
        bpy.ops.render.render(write_still=True)
        print("rendered", view["image_rgba"])


main()
