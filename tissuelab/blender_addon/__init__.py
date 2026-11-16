"""TissueLab add-on for Blender: load a patient's body wall (`surgvid body-wall`) and
drag it with handles, the tissue responding live by its measured layers and mechanics.

Sidebar (N) → TissueLab:
1. Install solver (once): SciPy into Blender's user modules (matching Blender's NumPy).
2. Load body wall: a body_wall_h*.npz.
3. Grab tissue: click and hold a surface, drag in the view plane, release to let go.
4. Alternatively add a persistent handle at the cursor, enable Live, and use G / R.
5. Reset: back to the CT shape, handles removed.
"""

bl_info = {
    "name": "TissueLab",
    "author": "dylanrussellmd",
    "version": (0, 2, 3),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > TissueLab",
    "description": "Patient body wall from CT, simulated live by tissue layer mechanics",
    "category": "Physics",
}

import os
import subprocess
import sys
import time

try:
    import bpy
    from bpy.props import BoolProperty, FloatProperty, IntProperty, StringProperty
    from bpy_extras.io_utils import ImportHelper
except ImportError:                        # Imported outside Blender (tests of the engine)
    bpy = None

COLLECTION = "TissueLab Body Wall"
SURFACE_COLOURS = {"skin": (0.80, 0.56, 0.47, 1.0), "peritoneum": (0.86, 0.62, 0.60, 1.0),
                   "pleura": (0.82, 0.66, 0.66, 1.0)}
TISSUE_COLOURS = {"skin": (0.80, 0.56, 0.47, 1.0), "subcutaneous_fat": (0.95, 0.80, 0.35, 1.0),
                  "muscle": (0.55, 0.10, 0.10, 1.0), "deep_fat": (0.98, 0.88, 0.55, 1.0),
                  "bone": (0.93, 0.90, 0.82, 1.0), "cartilage": (0.70, 0.82, 0.95, 1.0)}

_live_operators = {}
_grab_operator = None

_sessions: dict = {}                       # npz path -> engine.Session


def _modules_dir() -> str:
    return bpy.utils.user_resource("SCRIPTS", path="modules", create=True)


def scipy_available() -> bool:
    try:
        import scipy.sparse.linalg  # noqa: F401
        return True
    except ImportError:
        return False


def install_scipy(target: str) -> str:
    """pip-install a prebuilt SciPy wheel into `target`, without dependencies, so Blender's own
    NumPy is used. Blender's bundled Python with NumPy 1.x gets SciPy 1.13.1; any other
    (NumPy 2, newer Pythons such as a Linux distribution's Python 3.14) the newest SciPy, which
    ships wheels for current Pythons. Never builds from source (that needs a Fortran compiler
    and OpenBLAS)."""
    if bpy is not None and not bpy.app.online_access:
        raise RuntimeError("Enable Allow Online Access in Blender preferences to download SciPy")
    import numpy
    spec = "scipy==1.13.1" if numpy.__version__.startswith("1.") else "scipy"
    subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], check=False, capture_output=True)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--only-binary=:all:", "--upgrade",
                        "--target", target, spec], capture_output=True, text=True)
    if r.returncode != 0:
        tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-6:])
        raise RuntimeError(f"pip could not install a prebuilt {spec} for Python "
                           f"{sys.version.split()[0]} / NumPy {numpy.__version__}:\n{tail}\n{SCIPY_FALLBACK}")
    if target not in sys.path:
        sys.path.append(target)
    import importlib
    importlib.invalidate_caches()
    if not scipy_available():
        raise RuntimeError(f"{spec} installed but does not import with NumPy {numpy.__version__}. "
                           f"{SCIPY_FALLBACK}")
    return spec


SCIPY_FALLBACK = ("Alternatives: install your Linux distribution's SciPy package (e.g. `sudo apt install "
                  "python3-scipy`, `sudo dnf install python3-scipy`, `sudo pacman -S python-scipy`) when Blender "
                  "runs on the system Python, or use the blender.org build of Blender.")


def _session(scene):
    return _sessions.get(scene.tissuelab_wall_path)


def _engine():
    from . import engine
    return engine


# ---------------------------------------------------------------------------- scene objects

def _collection():
    col = bpy.data.collections.get(COLLECTION)
    if col is None:
        col = bpy.data.collections.new(COLLECTION)
        bpy.context.scene.collection.children.link(col)
    return col


def _material(name, colour, roughness=0.5):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = colour
        bsdf.inputs["Roughness"].default_value = roughness
    m.diffuse_color = colour
    return m


def _mesh_object(name, verts, faces, materials, face_material=None, smooth=True):
    import numpy as np
    me = bpy.data.meshes.get(name)
    if me is not None:
        ob = bpy.data.objects.get(name)
        if ob is not None:
            bpy.data.objects.remove(ob)
        bpy.data.meshes.remove(me)
    me = bpy.data.meshes.new(name)
    me.from_pydata(np.asarray(verts).tolist(), [], np.asarray(faces).tolist())
    for m in materials:
        me.materials.append(m)
    if face_material is not None:
        me.polygons.foreach_set("material_index", np.asarray(face_material, np.int32))
    me.polygons.foreach_set("use_smooth", np.full(len(me.polygons), smooth))
    me.update()
    ob = bpy.data.objects.new(name, me)
    _collection().objects.link(ob)
    return ob


def _set_coords(ob, pts):
    import numpy as np
    ob.data.vertices.foreach_set("co", np.ascontiguousarray(pts, np.float32).ravel())
    ob.data.update()


def build_objects(sess):
    """Render surfaces (skin, peritoneum, pleura) and the cut faces at the scan's ends (an
    object sharing the wall's vertex array; unused vertices are harmless)."""
    for name, s in sess.surfaces.items():
        mat = _material(f"TissueLab_{name.title()}", SURFACE_COLOURS.get(name, (0.8, 0.6, 0.5, 1.0)),
                        0.45 if name == "skin" else 0.25)
        ob = _mesh_object(f"BW_{name.title()}", sess.surface_positions(name), s["faces"], [mat])
        ob["tissuelab_surface"] = name
    tris, tissue = sess.scan_edge()
    mats = [_material(f"TissueLab_Cut_{t}", TISSUE_COLOURS.get(t, (0.7, 0.7, 0.7, 1.0)), 0.6) for t in sess.tissue_names]
    ob = _mesh_object("BW_Scan_Edge", sess.x, tris, mats, face_material=tissue, smooth=False)
    ob["tissuelab_scan_edge"] = True
    build_tether_object(sess)


def _tether_visibility(self, context):
    ob = bpy.data.objects.get("BW_Tethers")
    if ob is not None:
        if context.scene.tissuelab_show_tethers:
            sess = _session(context.scene)
            if sess is not None:
                _set_coords(ob, sess.tether_positions().reshape(-1, 3))
        ob.hide_set(not context.scene.tissuelab_show_tethers)


def build_tether_object(sess):
    """Inspection lines only; never included in rendering or used as physical geometry."""
    import numpy as np
    old = bpy.data.objects.get("BW_Tethers")
    if old is not None:
        data = old.data
        bpy.data.objects.remove(old, do_unlink=True)
        bpy.data.meshes.remove(data)
    pts = sess.tether_positions().reshape(-1, 3)
    if not len(pts):
        return
    me = bpy.data.meshes.new("BW_Tethers")
    me.from_pydata(pts.tolist(), np.arange(len(pts)).reshape(-1, 2).tolist(), [])
    ob = bpy.data.objects.new("BW_Tethers", me)
    _collection().objects.link(ob)
    ob["tissuelab_tethers"] = True
    ob.hide_render = True
    ob.show_in_front = True
    ob.hide_select = True
    ob.color = (1.0, 0.5, 0.05, 1.0)
    ob.hide_set(not bpy.context.scene.tissuelab_show_tethers)


def update_objects(sess):
    for ob in _collection().objects:
        if "tissuelab_surface" in ob:
            _set_coords(ob, sess.surface_positions(ob["tissuelab_surface"]))
        elif "tissuelab_tethers" in ob and not ob.hide_get():
            _set_coords(ob, sess.tether_positions().reshape(-1, 3))
        elif "tissuelab_scan_edge" in ob:
            _set_coords(ob, sess.x)


def _enclosing_objects(context, sess):
    """Other visible meshes whose bounds contain the wall (Blender's default 2 m cube does)."""
    import numpy as np
    from mathutils import Vector
    lo, hi = sess.x.min(0), sess.x.max(0)
    out = []
    for ob in context.scene.objects:
        if ob.type != "MESH" or ob.name.startswith("BW_") or not ob.visible_get():
            continue
        bb = np.array([ob.matrix_world @ Vector(c) for c in ob.bound_box])
        if (bb.min(0) <= lo).all() and (bb.max(0) >= hi).all():
            out.append(ob.name)
    return out


def _frame_wall(context):
    """Select the wall's objects and zoom every 3D view to them."""
    for ob in context.selected_objects:
        ob.select_set(False)
    objs = [ob for ob in _collection().objects if ob.type == "MESH" and not ob.get("tissuelab_tethers")]
    for ob in objs:
        ob.select_set(True)
    if objs:
        context.view_layer.objects.active = objs[0]
    for area in (context.screen.areas if context.screen else []):
        if area.type != "VIEW_3D":
            continue
        region = next((r for r in area.regions if r.type == "WINDOW"), None)
        if region is None:
            continue
        with context.temp_override(area=area, region=region):
            bpy.ops.view3d.view_selected()


def handle_matrices(scene):
    import numpy as np
    return {ob.name: np.array(ob.matrix_world) for ob in scene.objects if ob.get("tissuelab_handle")}


# ---------------------------------------------------------------------------- operators

if bpy is not None:
    class TISSUELAB_OT_install_solver(bpy.types.Operator):
        bl_idname = "tissuelab.install_solver"
        bl_label = "Install solver (SciPy)"
        bl_description = "Download SciPy into Blender's user modules folder (needs internet, once)"

        def execute(self, context):
            try:
                spec = install_scipy(_modules_dir())
            except Exception as e:                     # noqa: BLE001
                self.report({"ERROR"}, f"SciPy install failed: {e}")
                return {"CANCELLED"}
            self.report({"INFO"}, f"Installed {spec}")
            return {"FINISHED"}

    class TISSUELAB_OT_load_wall(bpy.types.Operator, ImportHelper):
        bl_idname = "tissuelab.load_wall"
        bl_label = "Load body wall"
        bl_description = "Load a body wall built by `surgvid body-wall` (body_wall_h*.npz)"
        filename_ext = ".npz"
        filter_glob: StringProperty(default="*.npz", options={"HIDDEN"})

        def execute(self, context):
            if not scipy_available():
                self.report({"ERROR"}, "Install the solver first (SciPy)")
                return {"CANCELLED"}
            sc = context.scene
            sess = _engine().Session(self.filepath, handle_stiffness=sc.tissuelab_handle_stiffness,
                                     skin_stretch_limit=sc.tissuelab_skin_limit, max_stretch=sc.tissuelab_max_stretch,
                                     skin_stiffness=sc.tissuelab_skin_stiffness if sc.tissuelab_skin_stiffening else 0.0,
                                     tethering=sc.tissuelab_tethering, tether_modulus=sc.tissuelab_tether_modulus,
                                     tether_slack_fraction=sc.tissuelab_tether_slack)
            _sessions[self.filepath] = sess
            context.scene.tissuelab_wall_path = self.filepath
            build_objects(sess)
            hiding = _enclosing_objects(context, sess)
            _frame_wall(context)
            info = sess.info
            msg = (f"Body wall {info.get('case')}: {len(sess.tets)} elements, {len(sess.x)} vertices, "
                   f"h {info.get('h_mm')} mm (real size, metres)")
            if hiding:
                self.report({"WARNING"}, msg + f". It is inside {', '.join(hiding)}: hide or delete "
                                               f"{'it' if len(hiding) == 1 else 'them'} to see the body")
            else:
                self.report({"INFO"}, msg)
            return {"FINISHED"}

    class TISSUELAB_OT_add_handle(bpy.types.Operator):
        bl_idname = "tissuelab.add_handle"
        bl_label = "Add handle at cursor"
        bl_description = "An empty at the 3D cursor that holds the tissue within the handle radius"

        def execute(self, context):
            sess = _session(context.scene)
            if sess is None:
                self.report({"ERROR"}, "Load a body wall first")
                return {"CANCELLED"}
            r = context.scene.tissuelab_handle_radius
            ob = bpy.data.objects.new(f"BW_Handle_{len(sess.handles) + 1:02d}", None)
            ob.empty_display_type = "SPHERE"
            ob.empty_display_size = r
            ob.location = context.scene.cursor.location
            ob["tissuelab_handle"] = True
            _collection().objects.link(ob)
            context.view_layer.update()
            import numpy as np
            n = sess.add_handle(ob.name, np.array(ob.location), r, np.array(ob.matrix_world))
            if n == 0:
                bpy.data.objects.remove(ob)
                self.report({"WARNING"}, "No free tissue within the radius (move the cursor onto the wall)")
                return {"CANCELLED"}
            for o in context.selected_objects:
                o.select_set(False)
            ob.select_set(True)
            context.view_layer.objects.active = ob
            self.report({"INFO"}, f"{ob.name}: holds {n} vertices")
            return {"FINISHED"}

    class TISSUELAB_OT_reset(bpy.types.Operator):
        bl_idname = "tissuelab.reset"
        bl_label = "Reset"
        bl_description = "Back to the CT shape; handles removed"

        def execute(self, context):
            sess = _session(context.scene)
            context.scene.tissuelab_live = False
            for ob in list(_collection().objects):
                if ob.get("tissuelab_handle"):
                    bpy.data.objects.remove(ob)
            if sess is not None:
                sess.reset()
                sess.display_tether_stats = sess.tether_stats()
                update_objects(sess)
            return {"FINISHED"}

    class TISSUELAB_OT_live(bpy.types.Operator):
        """One timer per scene; rate-limit TIMER events and sleep at equilibrium."""
        bl_idname = "tissuelab.live"
        bl_label = "Live"
        _timer = None

        def finish(self, context):
            if self._timer is not None:
                context.window_manager.event_timer_remove(self._timer)
                self._timer = None
            if _live_operators.get(self._key) is self:
                _live_operators.pop(self._key, None)
            return {"FINISHED"}

        def modal(self, context, event):
            import numpy as np
            if self._timer is None:
                return {"FINISHED"}
            scene = self._scene
            if not getattr(scene, "tissuelab_live", False) or _session(scene) is None:
                return self.finish(context)
            # A grab owns Esc while it is armed; releasing the grab must not stop relaxation.
            if event.type == "ESC" and _grab_operator is None:
                scene.tissuelab_live = False
                return self.finish(context)
            # Blender's RNA Event has no timer identity. Other tools can emit TIMER
            # events too; use a monotonic deadline instead of accessing Event.timer.
            if event.type != "TIMER":
                return {"PASS_THROUGH"}
            if time.perf_counter() < self._next_tick:
                return {"PASS_THROUGH"}
            try:
                sess = _session(scene)
                matrices = handle_matrices(scene)
                changed = (sess is not self._last_session or sess.solver is None
                           or matrices.keys() != self._last.keys()
                           or any(not np.array_equal(m, self._last[k]) for k, m in matrices.items()))
                if changed:
                    self._quiet = 0
                if self._quiet >= 3:
                    return {"PASS_THROUGH"}
                start = time.perf_counter()
                # During a drag prioritize input. Additional iterations settle on later ticks.
                iterations = 1 if changed or (_grab_operator and _grab_operator._handle) else scene.tissuelab_iters
                move = sess.step(matrices, iterations)
                if move > 1e-7:
                    update_objects(sess)
                self._quiet = self._quiet + 1 if move < 1e-7 else 0
                self._last, self._last_session = matrices, sess
                if time.perf_counter() >= self._next_stats or self._quiet >= 3:
                    sess.display_tether_stats = sess.tether_stats()
                    self._next_stats = time.perf_counter() + .2
                elapsed = time.perf_counter() - start
                self._next_tick = max(start + 1 / 30, time.perf_counter() + 1 / 60)
                scene.tissuelab_status = (("Tissue limit reached" if sess.limit_reached else f"max move {move * 1000:.2f} mm")
                                          + f" | update {elapsed * 1000:.0f} ms")
                if context.area:
                    context.area.tag_redraw()
            except Exception as exc:
                scene.tissuelab_live = False
                self.report({"ERROR"}, f"Tissue update stopped: {exc}")
                return self.finish(context)
            return {"PASS_THROUGH"}

        def execute(self, context):
            if _session(context.scene) is None:
                self.report({"ERROR"}, "Load a body wall first")
                return {"CANCELLED"}
            self._key = context.scene.as_pointer()
            context.scene.tissuelab_live = True
            if self._key in _live_operators:
                return {"FINISHED"}
            self._scene = context.scene
            self._last, self._last_session = {}, None
            self._quiet, self._next_tick, self._next_stats = 0, 0., 0.
            self._timer = context.window_manager.event_timer_add(1 / 30, window=context.window)
            _live_operators[self._key] = self
            context.window_manager.modal_handler_add(self)
            return {"RUNNING_MODAL"}

    class TISSUELAB_OT_grab(bpy.types.Operator):
        bl_idname = "tissuelab.grab"
        bl_label = "Grab tissue"
        bl_description = "Click and hold tissue to pull in the view plane; release to let go; Esc exits"
        _handle = None

        @classmethod
        def poll(cls, context):
            return context.area is not None and context.area.type == "VIEW_3D" and _session(context.scene) is not None

        def release(self):
            if self._handle is not None:
                try:
                    self._session.remove_handles([self._handle.name])
                    bpy.data.objects.remove(self._handle, do_unlink=True)
                except ReferenceError:  # Reset or scene teardown already removed the empty.
                    pass
                self._handle = None

        def finish(self, context):
            global _grab_operator
            self.release()
            self._area.header_text_set(None)
            context.window.cursor_modal_restore()
            if _grab_operator is self:
                _grab_operator = None
            return {"FINISHED"}

        def invoke(self, context, event):
            global _grab_operator
            if _grab_operator is not None:
                return {"CANCELLED"}
            self._area = context.area
            self._region = next(r for r in context.area.regions if r.type == "WINDOW")
            self._rv3d = context.space_data.region_3d
            self._session = _session(context.scene)
            self._scene = context.scene
            self._handle = None
            _grab_operator = self
            self._area.header_text_set("TissueLab: hold left mouse to pull; release to let go; orbit with middle mouse; Esc exits")
            context.window.cursor_modal_set("CROSSHAIR")
            bpy.ops.tissuelab.live()
            context.window_manager.modal_handler_add(self)
            return {"RUNNING_MODAL"}

        def modal(self, context, event):
            if _grab_operator is not self:
                return {"FINISHED"}
            from bpy_extras import view3d_utils
            from mathutils import Vector
            import numpy as np
            if (_session(self._scene) is not self._session or not self._scene.tissuelab_live
                    or event.type in {"ESC", "RIGHTMOUSE", "WINDOW_DEACTIVATE"}):
                return self.finish(context)
            coord = (event.mouse_x - self._region.x, event.mouse_y - self._region.y)
            inside = 0 <= coord[0] < self._region.width and 0 <= coord[1] < self._region.height
            try:
                if event.type == "LEFTMOUSE" and event.value == "PRESS" and inside and self._handle is None:
                    origin = view3d_utils.region_2d_to_origin_3d(self._region, self._rv3d, coord)
                    direction = view3d_utils.region_2d_to_vector_3d(self._region, self._rv3d, coord)
                    hit, point, normal, face, ob, matrix = self._scene.ray_cast(context.evaluated_depsgraph_get(), origin, direction)
                    if not hit or not ob.original.get("tissuelab_surface"):
                        return {"RUNNING_MODAL"}
                    # Wall coordinates are in world metres. Do not silently grab a displaced display mesh.
                    if not np.allclose(np.array(ob.matrix_world), np.eye(4)):
                        self.report({"WARNING"}, "Reload the wall before grabbing a transformed surface")
                        return {"RUNNING_MODAL"}
                    self._point = point.copy()
                    self._normal = self._rv3d.view_rotation @ Vector((0, 0, 1))
                    handle = bpy.data.objects.new("BW_Direct_Grab", None)
                    handle.empty_display_type = "SPHERE"
                    handle.empty_display_size = self._scene.tissuelab_handle_radius
                    handle.location = point
                    handle["tissuelab_handle"] = True
                    _collection().objects.link(handle)
                    self._handle = handle
                    context.view_layer.update()
                    count = self._session.add_handle(handle.name, np.array(point), self._scene.tissuelab_handle_radius,
                                                     np.array(handle.matrix_world))
                    if not count:
                        bpy.data.objects.remove(handle, do_unlink=True)
                        self._handle = None
                        self.report({"WARNING"}, "No free tissue here; try a larger handle radius")
                        return {"RUNNING_MODAL"}
                    self._handle = handle
                    return {"RUNNING_MODAL"}
                if event.type == "LEFTMOUSE" and event.value == "RELEASE" and self._handle is not None:
                    self.release()
                    return {"RUNNING_MODAL"}
                if event.type == "MOUSEMOVE" and self._handle is not None:
                    origin = view3d_utils.region_2d_to_origin_3d(self._region, self._rv3d, coord)
                    direction = view3d_utils.region_2d_to_vector_3d(self._region, self._rv3d, coord)
                    denom = direction.dot(self._normal)
                    if abs(denom) > 1e-6:
                        self._handle.location = origin + direction * ((self._point - origin).dot(self._normal) / denom)
                    return {"RUNNING_MODAL"}
                if self._handle is not None and event.type in {"MIDDLEMOUSE", "WHEELUPMOUSE", "WHEELDOWNMOUSE"}:
                    return {"RUNNING_MODAL"}
            except Exception as exc:
                self.report({"ERROR"}, f"Grab stopped: {exc}")
                return self.finish(context)
            return {"PASS_THROUGH"}

    class TISSUELAB_PT_panel(bpy.types.Panel):
        bl_label = "TissueLab"
        bl_space_type = "VIEW_3D"
        bl_region_type = "UI"
        bl_category = "TissueLab"

        def draw(self, context):
            sc = context.scene
            col = self.layout.column()
            if not scipy_available():
                col.label(text="Solver not installed", icon="ERROR")
                col.operator("tissuelab.install_solver", icon="IMPORT")
                return
            col.prop(sc, "tissuelab_handle_stiffness")
            col.prop(sc, "tissuelab_max_stretch")
            col.label(text="Hard limit is experimental; reload to apply")
            col.prop(sc, "tissuelab_skin_stiffening")
            if sc.tissuelab_skin_stiffening:
                col.prop(sc, "tissuelab_skin_limit")
                col.prop(sc, "tissuelab_skin_stiffness")
                col.label(text="Experimental; not patient calibrated")
            col.prop(sc, "tissuelab_tethering")
            if sc.tissuelab_tethering:
                col.prop(sc, "tissuelab_tether_modulus")
                col.prop(sc, "tissuelab_tether_slack")
            col.label(text="Settings apply when loading a wall")
            col.operator("tissuelab.load_wall", icon="FILE_FOLDER")
            sess = _session(sc)
            if sess is None:
                col.label(text="No body wall loaded")
                return
            col.label(text=f"{sess.info.get('case')}: {len(sess.tets)} elements")
            if sess.tether_file_present:
                report = sess.info.get("tethers", {})
                coverage = report.get("covered_area_m2", 0) / max(report.get("skin_area_m2", 0), 1e-30)
                col.label(text=f"Tethers: {len(sess.tether_tets)} ({coverage:.0%} skin coverage)")
                col.label(text="Tethering active" if sess.tethering and sess.tether_modulus > 0 and len(sess.tether_tets) else "Tethering inactive")
                lengths = report.get("rest_length_mm")
                if lengths:
                    col.label(text=f"Rest lengths: {lengths['min']:.1f}–{lengths['max']:.1f} mm")
                excluded = report.get("rejected_area_m2", {})
                if excluded:
                    reason = max(excluded, key=excluded.get)
                    col.label(text="Largest uncovered cause: " + reason.replace("_", " "))
                if not hasattr(sess, "display_tether_stats"):
                    sess.display_tether_stats = sess.tether_stats()
                ts = sess.display_tether_stats
                col.label(text=f"Taut: {ts['taut']} / {ts['connections']}")
                col.label(text=f"Peak tether tension: {ts['max_tension_n']:.3f} N (estimated)")
                if sess.limit_reached:
                    col.label(text="Hard stop active; tether response may be masked", icon="INFO")
                col.prop(sc, "tissuelab_show_tethers")
            elif sc.tissuelab_tethering:
                col.label(text="Rebuild body wall to generate tethers", icon="INFO")
            col.separator()
            col.prop(sc, "tissuelab_handle_radius")
            col.operator("tissuelab.grab", icon="HAND")
            col.label(text="Hold left mouse to pull; release to let go")
            col.operator("tissuelab.add_handle", icon="EMPTY_AXIS")
            col.label(text=f"Handles: {len(sess.handles)}")
            for h in sess.handles:
                col.label(text=f"{h.name}: {h.force_n:.2f} N (estimated)")
            col.separator()
            col.prop(sc, "tissuelab_iters")
            if sc.tissuelab_live:
                col.prop(sc, "tissuelab_live", text="Live (on)", toggle=True, icon="PAUSE")
            else:
                col.operator("tissuelab.live", text="Live", icon="PLAY")
            col.label(text=sc.tissuelab_status)
            col.operator("tissuelab.reset", icon="LOOP_BACK")

    CLASSES = (TISSUELAB_OT_install_solver, TISSUELAB_OT_load_wall, TISSUELAB_OT_add_handle, TISSUELAB_OT_reset,
               TISSUELAB_OT_live, TISSUELAB_OT_grab, TISSUELAB_PT_panel)


def register():
    md = _modules_dir()
    if os.path.isdir(md) and md not in sys.path:
        sys.path.append(md)
    S = bpy.types.Scene
    S.tissuelab_handle_stiffness = FloatProperty(name="Grip spring (N/m)", default=1000, min=1, max=100000,
        description="Total spring stiffness per handle, independent of vertex count; modelling assumption")
    S.tissuelab_tethering = BoolProperty(name="Experimental skin tethering", default=False)
    S.tissuelab_tether_modulus = FloatProperty(name="Tether strength (Pa)", default=10000, min=0,
        description="Assumed effective modulus; spring stiffness = strength × skin patch area / length")
    S.tissuelab_tether_slack = FloatProperty(name="Tether slack fraction", default=0.1, min=0, max=1,
        description="0.1 allows 10% length increase before tension; experimental, not calibrated")
    S.tissuelab_show_tethers = BoolProperty(name="Show tether connections", default=False, update=_tether_visibility)
    S.tissuelab_max_stretch = FloatProperty(name="Hard tissue stretch limit", default=1.5, min=1.01, max=3.0,
        description="Maximum principal stretch from the CT rest shape; 1.5 means 50% elongation. "
                    "Experimental guard, not calibrated failure. Stops further stretch without tearing")
    S.tissuelab_skin_stiffening = BoolProperty(name="Experimental skin stiffening", default=False)
    S.tissuelab_skin_limit = FloatProperty(name="Skin stretch onset", default=1.2, min=1.0, max=3.0,
        description="Principal stretch at stiffening onset; 1.2 means 20% elongation; uncalibrated")
    S.tissuelab_skin_stiffness = FloatProperty(name="Extra skin stiffness (Pa)", default=1000000, min=0,
        description="Experimental membrane penalty, not a measured patient modulus")
    S.tissuelab_wall_path = StringProperty(name="Body wall file")
    S.tissuelab_handle_radius = FloatProperty(name="Handle radius", default=0.03, min=0.005, max=0.15,
                                            unit="LENGTH")
    S.tissuelab_iters = IntProperty(name="Iterations per update", default=1, min=1, max=50)
    S.tissuelab_live = BoolProperty(name="Live", default=False, options={"SKIP_SAVE"})
    S.tissuelab_status = StringProperty(name="Status", default="")
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    global _grab_operator
    if _grab_operator is not None:
        _grab_operator.finish(bpy.context)
    for op in list(_live_operators.values()):
        op._scene.tissuelab_live = False
        op.finish(bpy.context)
    _sessions.clear()
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
    S = bpy.types.Scene
    for p in ("tissuelab_wall_path", "tissuelab_handle_radius", "tissuelab_iters", "tissuelab_live", "tissuelab_status",
              "tissuelab_handle_stiffness", "tissuelab_skin_stiffening", "tissuelab_skin_limit", "tissuelab_skin_stiffness", "tissuelab_max_stretch",
              "tissuelab_tethering", "tissuelab_tether_modulus", "tissuelab_tether_slack", "tissuelab_show_tethers"):
        delattr(S, p)
