# Offline PBR material assets

Powered by [Poly Haven](https://polyhaven.com). These original 1K JPEG texture
maps are vendored for reproducible, offline Gazebo rendering. There is no
runtime download or API dependency. No website preview renders are included.

Poly Haven distributes its asset files under
[CC0 1.0](https://polyhaven.com/license), permitting reuse and redistribution.
Attribution is not required for the assets; the original authors are credited
here and in `manifest.json` nevertheless. The manifest records each original
download URL, API-provided MD5 and byte count, and a locally verified SHA-256.

| Scene material | Original asset | Author(s) | Repeat size |
| --- | --- | --- | --- |
| `wood` | [Oak Wood Planks](https://polyhaven.com/a/oak_wood_planks) | Dimitrios Savva | 1.2 m |
| `floor` | [Wood Floor](https://polyhaven.com/a/wood_floor) | Dimitrios Savva | 1.7 m |
| `plaster` | [White Plaster 02](https://polyhaven.com/a/white_plaster_02) | Rob Tuytel | 1.0 m |
| `fabric` | [Denim Fabric 03](https://polyhaven.com/a/denim_fabric_03) | colormass (photography), Rico Cilliers (processing) | approximately 0.271 m |

Repeat sizes come from the physical dimensions published by the provider API,
converted from millimetres to metres. The slightly rectangular fabric scan is
1024 × 1022 pixels; its scalar repeat size rounds the measured dimensions.
All other maps are 1024 × 1024. All downloaded files retain their original
bytes, so their hashes can be checked against the provider records.

## Renderer integration

The manifest's `materials` mapping contains `albedo`, `normal`, and `roughness`
records. Each record's `path` is relative to this directory. Resolve it to an
absolute local resource path when generating the world. Use albedo as sRGB and
normal/roughness as linear data. Normal maps use OpenGL tangent-space (+Y),
not DirectX. Roughness is a separate grayscale texture, not a packed ORM map.
These are nonmetallic surfaces (metalness 0).

These maps affect visual appearance only. No displacement or collision
geometry is provided; the scene builder remains responsible for geometry,
UV tiling, contact/friction properties, and traversability. Fabric may be used
for upholstered furniture or a visual rug, but is not a measured carpet
friction model. Materials are natural surface patterns, not artificial
visual localization markers or poster templates.
