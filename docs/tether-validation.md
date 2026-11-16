# Tether quality and response

Tethers remain effective skin-to-muscle-interface connections. They are not segmented
retinacula or Scarpa's fascia. Strength and slack are modelling assumptions, not calibrated
patient measurements. Version 0.2.1 adds diagnostics without changing the constitutive law.

The tension-only law is `F = (E A / L0) max(L - L0 (1 + slack), 0)` along the connection.
Both endpoints move with their containing tetrahedra. Tension is zero during compression
and within the slack range; it increases continuously after slack is taken up. Sideways
motion develops resistance only as it increases connection length. This is not a complete
model of shear coupling within subcutaneous tissue.

The panel reports the number carrying tension and the largest individual tether tension.
That value is not total grip force or a tissue failure threshold. The hard stop is a
separate global constraint; a stop warning means it may be masking the tether response.

## Repeatable Blender check

1. Save the scene. Load the same wall with stiffening off and tethering off. Place a handle
   at a reproducible skin location and record its position/radius and the grip spring.
2. Move it outward in small increments, recording handle displacement and estimated grip
   force. Stop before the hard-limit indication appears.
3. Reload with tethering on, keeping other settings and the handle placement identical.
   Repeat. Watch the taut count and peak tension as slack is taken up.
4. Repeat with a sideways pull. Connections may remain slack longer than for an outward
   pull; resistance depends on length, not on handle distance alone.
5. Return the handle through the same positions, then delete it and allow relaxation.
   This elastic model should unload; it has no intentional damage or permanent set.
6. Compare several skin locations using the connection display. Do not infer support in
   uncovered areas or increase stiffness merely to hide missing connections.

New SurgVid wall builds record excluded skin area by reason, length min/median/max, and
whether original voxel labels were checked. Coverage is area-weighted. Older files still
work but need rebuilding to acquire these geometry diagnostics. Unsupported routes remain
omitted; no connections are added to fill coverage gaps.

Automated tests cover axial/sideways forces, continuous slack onset, equal-and-opposite
endpoint forces, load/unload equilibrium, density independence, moving deep endpoints,
malformed embeddings, and diagnostic reset/disabled behavior. SurgVid integration tests
compare enabled/disabled wall response before the hard stop and reject cavity crossings.
These synthetic checks establish implementation behavior, not clinical validity.

## Scarpa's fascia decision

No source CT or segmentation volume was available in the development workspace for this
pass. Therefore visibility, continuity and thickness of Scarpa's fascia have not been
assessed. Before adding a patient-specific layer, review the source CT in multiple planes
and establish a traceable segmentation with documented uncertainty. Do not manufacture a
visible fascia sheet at an arbitrary depth. If it is not identifiable, retain the effective
tether model. A separate membrane would need independent in-plane mechanics and attachments
above/below it, with care to avoid counting the same resistance twice.
