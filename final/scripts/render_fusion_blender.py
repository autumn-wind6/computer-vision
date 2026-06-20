#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Render an orbit video for a fused PLY scene in Blender.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration", type=float, default=24.0)
    parser.add_argument("--resolution-x", type=int, default=1280)
    parser.add_argument("--resolution-y", type=int, default=720)
    parser.add_argument("--samples", type=int, default=64)
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_ply(path: Path) -> list[bpy.types.Object]:
    path_str = str(path)
    errors: list[str] = []
    for name, op in (
        ("import_mesh.ply", lambda: bpy.ops.import_mesh.ply(filepath=path_str)),
        ("wm.ply_import", lambda: bpy.ops.wm.ply_import(filepath=path_str)),
    ):
        try:
            op()
            break
        except (AttributeError, RuntimeError) as exc:
            errors.append(f"{name}: {exc}")
    else:
        raise RuntimeError("No working PLY importer: " + "; ".join(errors))
    objects = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not objects:
        raise RuntimeError(f"No mesh objects imported from {path}")
    return objects


def is_point_cloud(obj: bpy.types.Object) -> bool:
    mesh = obj.data
    return len(mesh.polygons) == 0 and len(mesh.vertices) > 0


def add_point_cloud_geometry_nodes(obj: bpy.types.Object, point_radius: float) -> None:
    """Turn bare vertices into instanced micro-spheres so Cycles can render them."""
    modifier = obj.modifiers.new(name="PointCloudVis", type="NODES")
    node_group = bpy.data.node_groups.new(name="PointCloudVis", type="GeometryNodeTree")
    modifier.node_group = node_group

    interface = node_group.interface
    interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    nodes = node_group.nodes
    links = node_group.links
    input_node = nodes.new("NodeGroupInput")
    output_node = nodes.new("NodeGroupOutput")
    mesh_to_points = nodes.new("GeometryNodeMeshToPoints")
    mesh_to_points.mode = "VERTICES"
    ico = nodes.new("GeometryNodeMeshIcoSphere")
    ico.inputs["Radius"].default_value = point_radius
    ico.inputs["Subdivisions"].default_value = 1
    instance = nodes.new("GeometryNodeInstanceOnPoints")

    links.new(input_node.outputs["Geometry"], mesh_to_points.inputs["Mesh"])
    links.new(mesh_to_points.outputs["Points"], instance.inputs["Points"])
    links.new(ico.outputs["Mesh"], instance.inputs["Instance"])
    links.new(instance.outputs["Instances"], output_node.inputs["Geometry"])


def prepare_point_cloud_objects(objects: list[bpy.types.Object], scene_radius: float) -> None:
    point_radius = max(scene_radius * 0.004, 0.002)
    for obj in objects:
        if is_point_cloud(obj):
            add_point_cloud_geometry_nodes(obj, point_radius)


def scene_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, float]:
    mins = Vector((math.inf, math.inf, math.inf))
    maxs = Vector((-math.inf, -math.inf, -math.inf))
    for obj in objects:
        for corner in obj.bound_box:
            point = obj.matrix_world @ Vector(corner)
            mins.x = min(mins.x, point.x)
            mins.y = min(mins.y, point.y)
            mins.z = min(mins.z, point.z)
            maxs.x = max(maxs.x, point.x)
            maxs.y = max(maxs.y, point.y)
            maxs.z = max(maxs.z, point.z)
    center = (mins + maxs) * 0.5
    radius = max((maxs - mins).length * 0.5, 1.0)
    return center, radius


def add_camera_orbit(center: Vector, radius: float, fps: int, duration: float) -> int:
    frame_end = max(2, int(round(fps * duration)))
    empty = bpy.data.objects.new("scene_center", None)
    empty.location = center
    bpy.context.collection.objects.link(empty)

    camera_data = bpy.data.cameras.new("OrbitCamera")
    camera_data.lens = 28
    camera_data.dof.use_dof = True
    camera_data.dof.focus_object = empty
    camera_data.dof.aperture_fstop = 8
    camera = bpy.data.objects.new("OrbitCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera

    track = camera.constraints.new(type="TRACK_TO")
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    track.target = empty

    keyframes = [1, frame_end // 4, frame_end // 2, (3 * frame_end) // 4, frame_end]
    distance = radius * 2.8
    height = radius * 0.9
    for frame in keyframes:
        theta = 2.0 * math.pi * (frame - 1) / max(frame_end - 1, 1)
        camera.location = center + Vector((math.cos(theta) * distance, height, math.sin(theta) * distance))
        camera.keyframe_insert(data_path="location", frame=frame)

    if camera.animation_data and camera.animation_data.action:
        for fcurve in camera.animation_data.action.fcurves:
            for point in fcurve.keyframe_points:
                point.interpolation = "LINEAR"
    return frame_end


def setup_lighting(center: Vector, radius: float) -> None:
    bpy.ops.object.light_add(type="AREA", location=center + Vector((radius * 1.5, radius * 2.5, radius * 1.5)))
    key = bpy.context.object
    key.name = "KeyLight"
    key.data.energy = 800
    key.data.size = max(radius * 2.0, 2.0)

    bpy.ops.object.light_add(type="POINT", location=center + Vector((-radius * 1.2, radius * 1.1, -radius * 1.0)))
    fill = bpy.context.object
    fill.name = "FillLight"
    fill.data.energy = 150


def configure_cycles_gpu(scene: bpy.types.Scene, samples: int) -> None:
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    prefs = bpy.context.preferences
    cycles_prefs = prefs.addons["cycles"].preferences
    cycles_prefs.compute_device_type = "CUDA"
    cycles_prefs.get_devices()
    for device in cycles_prefs.devices:
        device.use = device.type == "CUDA"
    scene.cycles.device = "GPU"


def setup_render(args: argparse.Namespace, frame_end: int) -> None:
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frame_end
    scene.render.fps = args.fps
    scene.render.resolution_x = args.resolution_x
    scene.render.resolution_y = args.resolution_y
    scene.render.filepath = str(Path(args.output))
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.world = scene.world or bpy.data.worlds.new("World")
    scene.world.color = (0.03, 0.03, 0.035)

    use_cycles = os.environ.get("BLENDER_USE_CYCLES", "auto")
    headless = not os.environ.get("DISPLAY")
    if use_cycles == "1" or (use_cycles == "auto" and headless):
        configure_cycles_gpu(scene, args.samples)
    elif hasattr(scene, "eevee"):
        scene.render.engine = (
            "BLENDER_EEVEE_NEXT"
            if "BLENDER_EEVEE_NEXT"
            in [item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items]
            else "BLENDER_EEVEE"
        )
    else:
        configure_cycles_gpu(scene, args.samples)


def main() -> int:
    args = parse_args()
    source = Path(args.input)
    if not source.exists():
        raise FileNotFoundError(source)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    clear_scene()
    objects = import_ply(source)
    center, radius = scene_bounds(objects)
    prepare_point_cloud_objects(objects, radius)
    setup_lighting(center, radius)
    frame_end = add_camera_orbit(center, radius, args.fps, args.duration)
    setup_render(args, frame_end)
    bpy.ops.render.render(animation=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
