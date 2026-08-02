import trimesh

m = trimesh.load("mycobot_280/assets/m1.stl")
m.apply_translation(-m.centroid)      # center geometry at origin
m.apply_scale(0.001)                  # mm -> m
m2 = m.simplify_quadric_decimation(face_count=180000)
m2.export("mycobot_280/assets/m1_dec.stl")
print("new extents:", m2.extents, " bounds:", m2.bounds)