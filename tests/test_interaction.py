"""Modal scheduling/lifecycle checks using a small Blender API double (no viewport claims)."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS, ModuleType
from unittest.mock import Mock
import sys

import numpy as np
import pytest


@pytest.fixture
def ui(monkeypatch):
    bpy = ModuleType('bpy')
    bpy.types = NS(Operator=type('Operator', (), {'report': lambda *args: None}), Panel=type('Panel', (), {}))
    props = ModuleType('bpy.props')
    for name in ['BoolProperty', 'FloatProperty', 'IntProperty', 'StringProperty']:
        setattr(props, name, lambda **kw: None)
    extras = ModuleType('bpy_extras')
    io = ModuleType('bpy_extras.io_utils')
    io.ImportHelper = type('ImportHelper', (), {})
    math = ModuleType('mathutils'); math.Vector = np.asarray
    view = NS(region_2d_to_origin_3d=lambda r, v, p: np.array([p[0]*.001, p[1]*.001, 1.]),
              region_2d_to_vector_3d=lambda *args: np.array([0., 0., -1.]))
    extras.view3d_utils = view
    for name, mod in [('bpy', bpy), ('bpy.props', props), ('bpy_extras', extras),
                      ('bpy_extras.io_utils', io), ('mathutils', math)]:
        monkeypatch.setitem(sys.modules, name, mod)
    path = Path(__file__).parents[1]/'tissuelab/blender_addon/__init__.py'
    spec = importlib.util.spec_from_file_location('interaction_test_addon', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_accepts_rna_events_without_timer_and_rate_limits_updates(ui):
    clock = [100.]
    ui.time = NS(perf_counter=lambda: clock[0])
    scene = NS(as_pointer=lambda: 1, tissuelab_live=False, tissuelab_iters=3)
    sess = NS(solver=object(), step=Mock(return_value=0.), limit_reached=False, tether_stats=lambda: {})
    ui._session = lambda sc: sess
    matrix = np.eye(4)
    ui.handle_matrices = lambda sc: {'grip': matrix.copy()}
    ui.update_objects = Mock()
    wm = NS(event_timer_add=Mock(return_value=object()), modal_handler_add=Mock(), event_timer_remove=Mock())
    ctx = NS(scene=scene, window_manager=wm, window=object(), area=None)
    op = ui.TISSUELAB_OT_live()
    assert op.execute(ctx) == {'RUNNING_MODAL'}
    assert ui.TISSUELAB_OT_live().execute(ctx) == {'FINISHED'}
    assert wm.event_timer_add.call_count == 1
    op.modal(ctx, NS(type='MOUSEMOVE'))
    assert sess.step.call_count == 0
    op.modal(ctx, NS(type='TIMER'))
    assert sess.step.call_count == 1
    for _ in range(10):
        op.modal(ctx, NS(type='TIMER'))
    assert sess.step.call_count == 1  # Other tools' timers cannot trigger a burst.
    clock[0] += .01
    op.modal(ctx, NS(type='TIMER'))
    assert sess.step.call_count == 1
    for _ in range(8):
        clock[0] += .04
        op.modal(ctx, NS(type='TIMER'))
    assert sess.step.call_count == 3
    assert sess.step.call_args_list[0].args[1] == 1
    matrix[0, 3] = .01
    op._next_tick = 0
    op.modal(ctx, NS(type='TIMER'))
    assert sess.step.call_count == 4
    scene.tissuelab_live = False
    assert op.modal(ctx, NS(type='TIMER')) == {'FINISHED'}
    wm.event_timer_remove.assert_called_once()
    assert not ui._live_operators


def test_direct_drag_updates_target_and_release_removes_only_its_handle(ui):
    handle = NS(name='direct', location=np.zeros(3))
    sess = NS(remove_handles=Mock())
    scene = NS(tissuelab_live=True)
    ui._session = lambda sc: sess
    ui.bpy.data = NS(objects=NS(remove=Mock()))
    op = ui.TISSUELAB_OT_grab()
    op._scene, op._session, op._handle = scene, sess, handle
    op._region = NS(x=100, y=200, width=500, height=500)
    op._rv3d = object()
    op._point, op._normal = np.zeros(3), np.array([0., 0., 1.])
    op._area = NS(header_text_set=Mock())
    ui._grab_operator = op
    ctx = NS(window=NS(cursor_modal_restore=Mock()))
    event = NS(type='MOUSEMOVE', value='NOTHING', mouse_x=150, mouse_y=220)
    assert op.modal(ctx, event) == {'RUNNING_MODAL'}
    assert np.allclose(handle.location, [.05, .02, 0])
    event.type, event.value = 'LEFTMOUSE', 'RELEASE'
    assert op.modal(ctx, event) == {'RUNNING_MODAL'}
    sess.remove_handles.assert_called_once_with(['direct'])
    assert op._handle is None
    event.type = 'ESC'
    assert op.modal(ctx, event) == {'FINISHED'}
    assert scene.tissuelab_live  # Let the released tissue relax.
    assert ui._grab_operator is None
    ctx.window.cursor_modal_restore.assert_called_once()


def test_direct_grab_hits_surface_and_starts_at_clicked_point(ui):
    point = np.array([.02, .03, .04])
    handle = NS(name='direct', matrix_world=np.eye(4))
    class Empty(dict):
        name = 'direct'
        matrix_world = np.eye(4)
    handle = Empty()
    sess = NS(add_handle=Mock(return_value=3), remove_handles=Mock())
    scene = NS(tissuelab_live=True, tissuelab_handle_radius=.03,
               ray_cast=Mock(return_value=(True, point, None, 0,
                            NS(original={'tissuelab_surface': 'skin'}, matrix_world=np.eye(4)), None)))
    ui._session = lambda sc: sess
    ui.bpy.data = NS(objects=NS(new=Mock(return_value=handle), remove=Mock()))
    ui._collection = lambda: NS(objects=NS(link=Mock()))
    op = ui.TISSUELAB_OT_grab()
    op._scene, op._session, op._handle = scene, sess, None
    op._region = NS(x=0, y=0, width=500, height=500)
    op._rv3d = NS(view_rotation=np.eye(3))
    ui._grab_operator = op
    ctx = NS(evaluated_depsgraph_get=lambda: None, view_layer=NS(update=Mock()))
    event = NS(type='LEFTMOUSE', value='PRESS', mouse_x=20, mouse_y=30)
    assert op.modal(ctx, event) == {'RUNNING_MODAL'}
    assert op._handle is handle
    assert np.array_equal(handle.location, point)
    assert np.array_equal(sess.add_handle.call_args.args[1], point)
    event.type = 'MIDDLEMOUSE'
    assert op.modal(ctx, event) == {'RUNNING_MODAL'}  # No view changes during a pull.
    op.release()
    assert op.modal(ctx, event) == {'PASS_THROUGH'}  # Orbit between pulls.


def test_reload_applies_current_settings_and_stops_interaction(ui):
    sc = NS(tissuelab_wall_path='/wall.npz', tissuelab_live=True, as_pointer=lambda: 1,
            tissuelab_handle_stiffness=1000., tissuelab_skin_limit=1.25,
            tissuelab_max_stretch=1.7, tissuelab_skin_stiffness=500.,
            tissuelab_skin_stiffening=True, tissuelab_tethering=True,
            tissuelab_tether_modulus=2000., tissuelab_tether_slack=.2)
    ctx = NS(scene=sc)
    sess = NS(surfaces={}, scan_edge=Mock())
    factory = Mock(return_value=sess)
    ui._engine = lambda: NS(Session=factory)
    ui.bpy.path = NS(abspath=lambda path: path)
    ui._clear_wall_objects, ui.build_objects, ui._frame_wall = Mock(), Mock(), Mock()
    live, grab = NS(finish=Mock()), NS(finish=Mock())
    ui._live_operators[1], ui._grab_operator = live, grab
    ui._sessions['/wall.npz'] = object()
    assert ui.TISSUELAB_OT_reload_wall().execute(ctx) == {'FINISHED'}
    assert factory.call_args.kwargs['max_stretch'] == 1.7
    assert factory.call_args.kwargs['tether_slack_fraction'] == .2
    assert ui._sessions['/wall.npz'] is sess
    assert not sc.tissuelab_live
    live.finish.assert_called_once_with(ctx)
    grab.finish.assert_called_once_with(ctx)
    ui._clear_wall_objects.assert_called_once_with(sc)
    ui._frame_wall.assert_not_called()
    # Failed parsing leaves the successfully loaded scene untouched.
    factory.side_effect = FileNotFoundError('missing')
    assert ui.TISSUELAB_OT_reload_wall().execute(ctx) == {'CANCELLED'}
    assert ui._sessions['/wall.npz'] is sess
    assert ui._clear_wall_objects.call_count == 1


def test_cleanup_removes_tagged_duplicates_but_preserves_unrelated_objects(ui):
    class Object(dict):
        type = 'MESH'
        def __init__(self, name, props):
            super().__init__(props)
            self.name = name
            self.data = NS(users=0)
    original = Object('BW_Skin', {'tissuelab_surface': 'skin'})
    duplicate = Object('BW_Skin.001', {'tissuelab_surface': 'skin'})
    legacy = Object('renamed', {'surgvid_surface': 'skin'})
    unrelated = Object('BW_Pleura', {})
    scene = NS(objects=[original, duplicate, legacy, unrelated])
    remove = Mock(side_effect=lambda ob, **kw: scene.objects.remove(ob))
    meshes = Mock()
    ui.bpy.data = NS(objects=NS(remove=remove), meshes=NS(remove=meshes))
    ui._clear_wall_objects(scene)
    assert scene.objects == [unrelated]
    assert remove.call_count == meshes.call_count == 3
    assert all(c.kwargs == {'do_unlink': True} for c in remove.call_args_list)
