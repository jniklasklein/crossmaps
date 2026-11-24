"""Deterministic furnished office geometry, independent of the rover model.

Dimensions are in metres. Visual meshes carry metre-scaled UVs and smooth
bevels; simple static collision shapes keep contact simulation inexpensive.
No fiducials, diagnostic posters, downloaded executable assets or world poses
are exposed to perception. The room is a synthetic scene, not a scanned twin.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import xml.etree.ElementTree as ET


def _node(parent, tag, value=None, **attributes):
    child = ET.SubElement(parent, tag, attributes)
    if value is not None:
        child.text = str(value)
    return child


def _numbers(values):
    return " ".join(f"{float(v):.8g}" for v in values)


def _pose(parent, xyz, rpy=(0., 0., 0.)):
    _node(parent, "pose", _numbers((*xyz, *rpy)))


def _unit(vector):
    length = math.sqrt(sum(v*v for v in vector))
    return tuple(v / length for v in vector) if length else (0., 0., 1.)


def _collada(path, positions, normals, uvs, triangles):
    """Write portable COLLADA with explicit vertex normals and texture UVs."""
    root = ET.Element("COLLADA", version="1.4.1", xmlns="http://www.collada.org/2005/11/COLLADASchema")
    asset = _node(root, "asset")
    _node(asset, "created", "2026-11-23T00:00:00Z")
    _node(asset, "modified", "2026-11-23T00:00:00Z")
    _node(asset, "unit", meter="1", name="metre")
    _node(asset, "up_axis", "Z_UP")
    mesh = _node(_node(_node(root, "library_geometries"), "geometry", id="shape"), "mesh")
    for name, values, components in (("positions", positions, "XYZ"), ("normals", normals, "XYZ"), ("uv", uvs, "ST")):
        source = _node(mesh, "source", id=name)
        _node(source, "float_array", _numbers(v for row in values for v in row), id=name+"-array", count=str(len(values)*len(components)))
        access = _node(_node(source, "technique_common"), "accessor", source="#"+name+"-array", count=str(len(values)), stride=str(len(components)))
        for component in components:
            _node(access, "param", name=component, type="float")
    _node(_node(mesh, "vertices", id="vertices"), "input", semantic="POSITION", source="#positions")
    faces = _node(mesh, "triangles", count=str(len(triangles)))
    for offset, semantic, source in ((0, "VERTEX", "vertices"), (1, "NORMAL", "normals"), (2, "TEXCOORD", "uv")):
        attributes = dict(semantic=semantic, source="#"+source, offset=str(offset))
        if semantic == "TEXCOORD":
            attributes["set"] = "0"
        _node(faces, "input", **attributes)
    _node(faces, "p", " ".join(str(index) for face in triangles for vertex in face for index in (vertex, vertex, vertex)))
    scene = _node(_node(root, "library_visual_scenes"), "visual_scene", id="scene")
    _node(_node(scene, "node", id="mesh"), "instance_geometry", url="#shape")
    _node(_node(root, "scene"), "instance_visual_scene", url="#scene")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def rounded_box_mesh(path, size, radius, scale=1.):
    """Round a subdivided cube using the exact rounded-box surface normal."""
    half = tuple(v/2 for v in size)
    radius = min(max(0., radius), min(half)*.8)
    positions, normals, uvs, triangles = [], [], [], []
    # Outward normal, horizontal UV axis, vertical UV axis. U cross V = N.
    faces = ((0, 1, 1, 1, 2, 1), (0, -1, 1, -1, 2, 1),
             (1, 1, 0, -1, 2, 1), (1, -1, 0, 1, 2, 1),
             (2, 1, 0, 1, 1, 1), (2, -1, 0, 1, 1, -1))
    for axis, sign, u_axis, u_sign, v_axis, v_sign in faces:
        def coordinates(h):
            if not radius:
                return [-h, h]
            return [-h, -h+radius*.35, -h+radius, h-radius, h-radius*.35, h]
        us, vs = coordinates(half[u_axis]), coordinates(half[v_axis])
        first = len(positions)
        for v in vs:
            for u in us:
                raw = [0., 0., 0.]
                raw[axis], raw[u_axis], raw[v_axis] = sign*half[axis], u_sign*u, v_sign*v
                inside = [max(-half[i]+radius, min(half[i]-radius, raw[i])) for i in range(3)]
                normal = _unit([raw[i]-inside[i] for i in range(3)]) if radius else tuple(sign if i == axis else 0. for i in range(3))
                positions.append(tuple(inside[i]+radius*normal[i] for i in range(3)) if radius else tuple(raw))
                normals.append(normal)
                uvs.append(((u+half[u_axis])/scale, (v+half[v_axis])/scale))
        for j in range(len(vs)-1):
            for i in range(len(us)-1):
                a = first+j*len(us)+i
                b, c, d = a+1, a+len(us)+1, a+len(us)
                triangles.extend(((a, b, c), (a, c, d)))
    _collada(path, positions, normals, uvs, triangles)
    return len(triangles)


def lathe_mesh(path, profile, segments=32, scale=1.):
    """Revolve an ordered (radius,z) profile; supports actual open vessel rims."""
    positions, normals, uvs, triangles = [], [], [], []
    distance = 0.
    for row, (radius, z) in enumerate(profile):
        if row:
            distance += math.hypot(radius-profile[row-1][0], z-profile[row-1][1])
        previous, following = profile[max(0, row-1)], profile[min(len(profile)-1, row+1)]
        dr, dz = following[0]-previous[0], following[1]-previous[1]
        for i in range(segments+1):
            angle = 2*math.pi*i/segments
            c, s = math.cos(angle), math.sin(angle)
            positions.append((radius*c, radius*s, z))
            normals.append(_unit((dz*c, dz*s, -dr)))
            uvs.append((2*math.pi*radius*i/segments/scale, distance/scale))
        if row:
            for i in range(segments):
                a = (row-1)*(segments+1)+i
                b, c, d = a+1, a+segments+2, a+segments+1
                triangles.extend(((a, b, c), (a, c, d)))
    _collada(path, positions, normals, uvs, triangles)
    return len(triangles)


class Room:
    def __init__(self, world, assets, materials):
        self.world, self.assets, self.materials = world, Path(assets), materials
        self.assets.mkdir(parents=True, exist_ok=True)
        self.meshes, self.visual_count, self.triangle_count = {}, 0, 0

    def model(self, name, xyz=(0., 0., 0.), yaw=0.):
        model = _node(self.world, "model", name=name)
        _node(model, "static", "true")
        _pose(model, xyz, (0., 0., yaw))
        return _node(model, "link", name="body")

    def material(self, visual, color, name=None, roughness=.65, metallic=0.):
        mat = _node(visual, "material")
        _node(mat, "ambient", _numbers(color))
        _node(mat, "diffuse", _numbers(color))
        _node(mat, "specular", ".12 .12 .12 1")
        pbr = _node(_node(mat, "pbr"), "metal")
        _node(pbr, "roughness", roughness)
        _node(pbr, "metalness", metallic)
        source = self.materials.get(name, {})
        for field, tag in (("albedo", "albedo_map"), ("normal", "normal_map"), ("roughness", "roughness_map")):
            if source.get(field):
                attributes = {"type": "tangent"} if field == "normal" else {}
                _node(pbr, tag, Path(source[field]).resolve().as_uri(), **attributes)

    def mesh(self, kind, parameters, scale):
        key = json.dumps([kind, parameters, scale], sort_keys=True)
        if key not in self.meshes:
            path = self.assets / (kind+"_"+hashlib.sha256(key.encode()).hexdigest()[:12]+".dae")
            if kind == "bevel":
                triangles = rounded_box_mesh(path, parameters[0], parameters[1], scale)
            else:
                triangles = lathe_mesh(path, parameters, scale=scale)
            self.meshes[key] = (path, triangles)
        return self.meshes[key]

    def visual(self, link, name, xyz, shape, dimensions, color, mat=None, rpy=(0., 0., 0.), radius=0., roughness=.65, metallic=0.):
        visual = _node(link, "visual", name=name)
        _pose(visual, xyz, rpy)
        geometry = _node(visual, "geometry")
        if shape in ("box", "lathe"):
            scale = float(self.materials.get(mat, {}).get("scale_m", 1.))
            path, triangles = self.mesh("bevel" if shape == "box" else "lathe", (dimensions, radius) if shape == "box" else dimensions, scale)
            _node(_node(geometry, "mesh"), "uri", path.as_uri())
            self.triangle_count += triangles
        else:
            primitive = _node(geometry, shape)
            _node(primitive, "radius", dimensions[0])
            if shape == "cylinder":
                _node(primitive, "length", dimensions[1])
            self.triangle_count += 128 if shape == "cylinder" else 256
        _node(visual, "cast_shadows", "true")
        self.material(visual, color, mat, roughness, metallic)
        self.visual_count += 1
        return visual

    def box(self, link, name, xyz, size, color=(1., 1., 1., 1.), mat=None, radius=.004, rpy=(0., 0., 0.), collide=True, **finish):
        result = self.visual(link, name, xyz, "box", size, color, mat, rpy, radius, **finish)
        if collide:
            collision = _node(link, "collision", name=name)
            _pose(collision, xyz, rpy)
            _node(_node(_node(collision, "geometry"), "box"), "size", _numbers(size))
        return result

    def cylinder(self, link, name, xyz, radius, length, color, rpy=(0., 0., 0.), collide=True, **finish):
        result = self.visual(link, name, xyz, "cylinder", (radius, length), color, rpy=rpy, **finish)
        if collide:
            collision = _node(link, "collision", name=name)
            _pose(collision, xyz, rpy)
            primitive = _node(_node(collision, "geometry"), "cylinder")
            _node(primitive, "radius", radius)
            _node(primitive, "length", length)
        return result

    def beam(self, link, name, a, b, radius, color, **finish):
        delta = [b[i]-a[i] for i in range(3)]
        length = math.sqrt(sum(v*v for v in delta))
        direction = _unit(delta)
        rpy = (0., math.acos(max(-1., min(1., direction[2]))), math.atan2(direction[1], direction[0]))
        return self.cylinder(link, name, tuple((a[i]+b[i])/2 for i in range(3)), radius, length, color, rpy=rpy, **finish)

    def vessel(self, link, name, xyz, profile, color, mat=None, roughness=.65):
        return self.visual(link, name, xyz, "lathe", profile, color, mat, roughness=roughness)


def _book(room, link, name, xyz, size, color, yaw=0., standing=True):
    # Paper inset, two covers and a curved-looking separately bevelled spine.
    x, y, z = xyz
    depth, width, height = size
    room.box(link, name+"_pages", (x, y, z), (depth-.008, width-.009, height-.008), (.82, .79, .69, 1), radius=0, collide=False)
    for side in (-1, 1):
        room.box(link, name+"_cover"+str(side), (x, y+side*(width-.003)/2, z), (depth, .003, height), color, radius=0, collide=False)
    room.box(link, name+"_spine", (x-depth/2, y, z), (.007, width, height), color, radius=.002, collide=False)
    for stripe_z in (-height*.32, height*.33):
        room.box(link, name+"_binding"+str(stripe_z), (x-depth/2-.004, y, z+stripe_z), (.001, width*.8, .003), (.65, .58, .4, 1), radius=0, collide=False)


def build_room(world, assets: Path, materials: dict):
    """Populate architecture, lighting and detailed static furniture.

    Material names: floor, wood, plaster, fabric; each accepts albedo, normal,
    roughness paths and a physical scale_m. Missing maps use plain materials.
    """
    r = Room(world, assets, materials)
    metal, rubber = (.13, .15, .16, 1), (.035, .041, .044, 1)
    white = (.81, .82, .80, 1)
    scene = _node(world, "scene")
    _node(scene, "ambient", ".48 .48 .46 1")
    _node(scene, "background", ".65 .72 .78 1")
    _node(scene, "shadows", "true")
    light = _node(world, "light", name="window_daylight", type="directional")
    _pose(light, (2., 3.8, 3.))
    _node(light, "direction", ".25 -.7 -.85")
    _node(light, "diffuse", ".88 .91 1 1")
    _node(light, "specular", ".3 .3 .3 1")
    _node(light, "cast_shadows", "true")
    for index, x in enumerate((.5, 3.8)):
        light = _node(world, "light", name=f"ceiling_fill_{index}", type="point")
        _pose(light, (x, 0., 2.85))
        _node(light, "diffuse", ".65 .61 .53 1")
        _node(light, "specular", ".2 .2 .2 1")
        _node(light, "cast_shadows", "false")
        attenuation = _node(light, "attenuation")
        for key, value in (("range", 12), ("constant", .7), ("linear", .08), ("quadratic", .015)):
            _node(attenuation, key, value)

    architecture = r.model("room_architecture")
    r.box(architecture, "oak_floor", (1.5, 0., -.05), (8., 8., .1), mat="floor", radius=0)
    for name, xyz, size in (("back_wall", (5.5, 0., 1.5), (.15, 8., 3.)),
                            ("front_wall", (-2.5, 0., 1.5), (.15, 8., 3.)),
                            ("left_wall", (1.5, 4., 1.5), (8., .15, 3.)),
                            ("right_wall", (1.5, -4., 1.5), (8., .15, 3.))):
        r.box(architecture, name, xyz, size, mat="plaster", radius=0)
    for name, xyz, size in (("back_skirting", (5.409, 0., .065), (.035, 8., .13)),
                            ("front_skirting", (-2.409, 0., .065), (.035, 8., .13)),
                            ("left_skirting", (1.5, 3.909, .065), (8., .035, .13)),
                            ("right_skirting", (1.5, -3.909, .065), (8., .035, .13))):
        r.box(architecture, name, xyz, size, white, radius=.003)
    # Closed recessed door, jambs and hardware: no unexplained hole behind it.
    door = r.model("oak_door", (5.39, -2.35, 0.))
    r.box(door, "door_leaf", (0., 0., 1.025), (.045, .92, 2.05), mat="wood", radius=.004)
    for side in (-1, 1):
        r.box(door, "jamb"+str(side), (-.02, side*.5, 1.06), (.065, .065, 2.12), white, radius=.003)
    r.box(door, "head", (-.02, 0., 2.10), (.065, 1.065, .065), white, radius=.003)
    r.cylinder(door, "handle_rosette", (-.042, .32, .98), .027, .015, (.5,.52,.54,1), rpy=(0.,math.pi/2,0.), metallic=.85, roughness=.25)
    r.beam(door, "lever", (-.065,.32,.98), (-.065,.22,.98), .009, (.5,.52,.54,1), metallic=.85, roughness=.25)
    # Daylight window with frame, glazing, a sill and partially open blinds.
    window = r.model("side_window", (2.6, 3.91, 1.65))
    r.box(window, "glazing", (0., 0., 0.), (2.2,.018,1.25), (.55,.68,.73,1), radius=0, roughness=.15)
    for x in (-1.13, 0., 1.13):
        r.box(window, "vertical_frame"+str(x), (x,-.018,0.), (.055,.07,1.35), white, radius=.003)
    for z in (-.65,.65):
        r.box(window, "horizontal_frame"+str(z), (0.,-.018,z), (2.3,.07,.055), white, radius=.003)
    r.box(window, "sill", (0.,-.13,-.69), (2.42,.32,.055), mat="wood", radius=.006)
    for i in range(8):
        r.box(window, f"blind_slat_{i}", (0.,-.08,.61-i*.055), (2.16,.035,.025), (.70,.69,.63,1), rpy=(.35,0.,0.), radius=.001, collide=False)
    radiator = r.model("radiator", (3.1,3.77,.4))
    for i in range(15):
        r.box(radiator, f"fin_{i}", (-.56+i*.08,0.,0.), (.055,.13,.55), (.73,.73,.70,1), radius=.012)
    r.beam(radiator, "pipe", (-.69,.0,-.32), (.69,0.,-.32), .015, white)
    for index, y in enumerate((-1.0, 1.8)):
        socket = r.model(f"wall_power_{index}", (5.408,y,.27))
        r.box(socket,"plate",(0.,0.,0.),(.022,.12,.08),white,radius=.006)
        for dy in (-.03,.03):
            r.cylinder(socket,"socket"+str(dy),(-.014,dy,0.),.020,.002,(.15,.15,.14,1),rpy=(0.,math.pi/2,0.),collide=False)

    # Wooden desk, bevelled edges and under-desk steel supports.
    desk = r.model("workbench", (3.15, 1.1, 0.), yaw=-.035)
    r.box(desk,"desktop",(0.,0.,.755),(1.35,1.5,.055),mat="wood",radius=.012)
    for x in (-.54,.54):
        for y in (-.61,.61):
            r.box(desk,f"leg_{x}_{y}",(x,y,.365),(.045,.045,.73),metal,radius=.004,metallic=.7)
            r.box(desk,f"foot_{x}_{y}",(x,y,.015),(.055,.055,.03),rubber,radius=.005)
    for x in (-.54,.54):
        r.box(desk,"crossbar"+str(x),(x,0.,.64),(.035,1.22,.04),metal,radius=.003,metallic=.7)
    r.box(desk,"cable_tray",(.43,0.,.65),(.16,1.1,.035),metal,radius=.005)
    # Monitor faces -X, so it is naturally viewed from the rovers.
    r.box(desk,"monitor_base",(.27,.1,.798),(.25,.30,.025),metal,radius=.018)
    r.box(desk,"monitor_neck",(.31,.1,.91),(.035,.05,.23),metal,radius=.004)
    r.box(desk,"monitor_body",(.31,.1,1.13),(.035,.58,.34),rubber,radius=.01)
    screen = r.box(desk,"screen",(.290,.1,1.13),(.002,.548,.303),(.06,.105,.13,1),radius=0,collide=False,roughness=.2)
    _node(screen.find("material"),"emissive",".025 .05 .065 1")
    r.box(desk,"keyboard",(-.20,.1,.797),(.16,.45,.021),(.16,.17,.17,1),radius=.008)
    for row in range(4):
        for key in range(10):
            r.box(desk,f"key_{row}_{key}",(-.255+row*.035,-.09+key*.041,.811),(.025,.030,.006),(.31,.33,.33,1),radius=0,collide=False)
    r.box(desk,"mouse",(-.18,-.29,.81),(.105,.06,.038),rubber,radius=.018,collide=False)
    for i in range(3):
        r.box(desk,"notebook"+str(i),(-.12,.48,.797+i*.020),(.25,.18,.018),[(.24,.31,.29,1),(.61,.47,.29,1),(.36,.18,.16,1)][i],radius=.003)
    r.vessel(desk,"coffee_cup",(-.35,-.49,.783),[(.035,0.),(.041,.007),(.043,.09),(.041,.095),(.035,.095),(.035,.01),(.0,.01)],(.73,.72,.65,1),roughness=.3)
    r.cylinder(desk,"coffee",(-.35,-.49,.862),.034,.001,(.07,.035,.018,1),collide=False)
    for i in range(8):
        a,b=2*math.pi*i/8,2*math.pi*(i+1)/8
        r.beam(desk,f"cup_handle_{i}",(-.35,-.538+.026*math.cos(a),.835+.026*math.sin(a)),(-.35,-.538+.026*math.cos(b),.835+.026*math.sin(b)),.006,(.73,.72,.65,1),collide=False,roughness=.3)

    chair = r.model("office_chair", (2.25,-.60,0.), yaw=.15)
    for i in range(5):
        a=2*math.pi*i/5
        end=(.27*math.cos(a),.27*math.sin(a),.075)
        r.beam(chair,f"spoke_{i}",(0.,0.,.13),end,.022,metal,metallic=.6)
        r.cylinder(chair,f"caster_{i}",(end[0],end[1],.035),.035,.035,rubber,rpy=(math.pi/2,0.,a),roughness=.8)
    r.cylinder(chair,"gas_lift",(0.,0.,.25),.025,.31,(.35,.38,.39,1),metallic=.8,roughness=.25)
    r.box(chair,"seat_shell",(0.,0.,.43),(.46,.46,.045),rubber,radius=.02)
    r.box(chair,"seat_cushion",(-.015,0.,.473),(.46,.47,.065),mat="fabric",radius=.025)
    r.box(chair,"back_support",(.20,0.,.62),(.045,.09,.35),metal,radius=.012)
    r.box(chair,"back_shell",(.23,0.,.745),(.075,.445,.43),rubber,rpy=(0.,-.10,0.),radius=.035)
    r.box(chair,"back_cushion",(.183,0.,.745),(.055,.43,.405),mat="fabric",rpy=(0.,-.10,0.),radius=.022)
    for side in (-1,1):
        r.beam(chair,"arm_support"+str(side),(.08,side*.235,.43),(.10,side*.255,.63),.015,metal)
        r.box(chair,"armrest"+str(side),(-.015,side*.255,.64),(.28,.055,.035),rubber,radius=.016)

    shelf=r.model("oak_bookshelf",(4.9,.2,0.))
    r.box(shelf,"back",(.18,0.,.85),(.022,1.65,1.7),mat="wood",radius=.003)
    for y in (-.81,.81):
        r.box(shelf,"side"+str(y),(0.,y,.85),(.40,.035,1.7),mat="wood",radius=.006)
    for z in (.10,.48,.87,1.26,1.68):
        r.box(shelf,"shelf"+str(z),(0.,0.,z),(.40,1.65,.025),mat="wood",radius=.005)
    palette=[(.20,.28,.28,1),(.56,.24,.18,1),(.62,.52,.33,1),(.21,.29,.42,1),(.56,.56,.48,1),(.31,.18,.15,1)]
    rng=random.Random(2026)
    for row,z in enumerate((.115,.495,.885)):
        y=-.70
        for i in range(11):
            width=rng.uniform(.035,.085)
            height=rng.uniform(.20,.31)
            depth=rng.uniform(.22,.31)
            _book(r,shelf,f"book_{row}_{i}",(-.035,y+width/2,z+height/2),(depth,width,height),palette[rng.randrange(len(palette))])
            y+=width+rng.uniform(.008,.035)
    for i in range(2):
        r.box(shelf,"archive_box"+str(i),(-.015,-.40+i*.71,1.425),(.29,.52,.28),(.44,.45,.40,1),mat="fabric",radius=.008)
        r.box(shelf,"archive_handle"+str(i),(-.164,-.40+i*.71,1.43),(.005,.12,.025),rubber,radius=.005,collide=False)

    # Ordinary unlabelled packing boxes. Real folds/tape replace printed targets.
    for index,xyz,size in ((0,(1.85,.83,.19),(.48,.42,.38)),(1,(3.55,-1.17,.265),(.53,.57,.53))):
        box=r.model(f"cardboard_carton_{index}",xyz,yaw=(-.12 if index==0 else .08))
        brown=(.49,.35,.20,1)
        r.box(box,"carton",(0.,0.,0.),size,brown,radius=.007)
        r.box(box,"top_seam",(0.,0.,size[2]/2+.001),(size[0]-.01,.004,.001),(.24,.18,.12,1),radius=0,collide=False)
        r.box(box,"packing_tape",(0.,0.,size[2]/2+.002),(size[0]+.002,.06,.0015),(.62,.47,.29,1),radius=0,collide=False,roughness=.4)
        for side in (-1,1):
            r.box(box,"tape_drop"+str(side),(side*(size[0]/2+.001),0.,size[2]*.31),(.0015,.06,size[2]*.37),(.62,.47,.29,1),radius=0,collide=False,roughness=.4)
            r.box(box,"vertical_fold"+str(side),(-size[0]/2-.001,side*(size[1]/2-.016),0.),(.001,.002,size[2]-.015),(.36,.25,.15,1),radius=0,collide=False)

    # Tapered open bins with visible inner walls, rolled lips and lids nearby.
    bins=r.model("recycling_bins",(4.0,-1.75,0.))
    for i,color in enumerate(((.48,.53,.49,1),(.13,.16,.17,1))):
        profile=[(.14,0.),(.145,.025),(.19,.47),(.198,.48),(.198,.49),(.18,.49),(.17,.46),(.13,.035),(.0,.035)]
        r.vessel(bins,"bin"+str(i),(0.,-i*.47,0.),profile,color,roughness=.55)
        r.cylinder(bins,"collision_body"+str(i),(0.,-i*.47,.23),.15,.46,color)
        # The simplified cylinder stays inside the hollow visual surface.
        bins.remove(bins.find(f"visual[@name='collision_body{i}']"))
        r.visual_count-=1
        r.triangle_count-=128

    plant_path = Path(__file__).parent / "object_assets/potted_plant_02/potted_plant_02_1k.gltf"
    if not plant_path.is_file():
        raise FileNotFoundError(f"Vendored plant asset missing: {plant_path}")
    plant=r.model("potted_rubber_plant",(3.05,-2.10,0.))
    visual=_node(plant,"visual",name="photographic_plant")
    _pose(visual,(0.,0.,0.),(math.pi/2,0.,0.))
    _node(_node(_node(visual,"geometry"),"mesh"),"uri",plant_path.resolve().as_uri())
    _node(visual,"cast_shadows","true")
    # No SDF material override: retain the glTF's photographic PBR materials
    # and alpha-masked foliage. Collision follows the pot, not leaf triangles.
    collision=_node(plant,"collision",name="pot")
    _pose(collision,(0.,0.,.168))
    primitive=_node(_node(collision,"geometry"),"cylinder")
    _node(primitive,"radius",.237)
    _node(primitive,"length",.336)
    r.visual_count+=1
    r.triangle_count+=69806
    # A folded textile throw and a low stool add non-planar close-range detail.
    stool=r.model("upholstered_stool",(2.0,-1.55,0.),yaw=-.2)
    r.box(stool,"cushion",(0.,0.,.34),(.43,.37,.08),mat="fabric",radius=.035)
    for x in (-.15,.15):
        for y in (-.12,.12):
            r.beam(stool,f"leg_{x}_{y}",(x*1.25,y*1.25,.015),(x,y,.31),.022,(.5,.39,.27,1))
    return {"style":"furnished office with photographic PBR materials; no posters or feature targets",
            "visuals":r.visual_count,"estimated_visual_triangles":r.triangle_count,
            "mesh_files":len(r.meshes),"collision_policy":"static primitive approximations; detailed visuals",
            "materials":sorted(materials),"imported_objects":["Poly Haven potted_plant_02"],
            "limitations":"Synthetic arrangement; fixed furniture and approximate contact meshes, opaque simplified glazing."}
