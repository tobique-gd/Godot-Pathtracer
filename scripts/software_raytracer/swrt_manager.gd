@tool
extends Node
class_name RaytracingManager

var display_rect: TextureRect

@export var max_samples: int = 1024

@export var raytracing_enabled: bool = true:
	set(value):
		raytracing_enabled = value
		for mesh in get_mesh_instances():
			mesh.visible = not value

var shader_file := "uid://d180kqd22g5uw"
@export var image_save_name : String = ""

@export_tool_button("Update Scene", "Callable") var update_btn: Callable = _initialize_raytracing_pipeline
@export_tool_button("Recompile Shader", "Callable") var recompile_btn: Callable = recompile_shader


var material_lookup := {}
var material_count := 0
var material_data := PackedByteArray()
var light_data := PackedByteArray()
var unique_materials: Array[StandardMaterial3D] = []

var albedo_textures: Array[Texture2D] = []
var normal_textures: Array[Texture2D] = []
var orm_textures: Array[Texture2D] = []

const TEXTURE_ARRAY_SIZE := Vector2i(1024, 1024)


var rd: RenderingDevice
var shader: RID
var pipeline: RID

var triangle_buffer: RID
var normal_buffer: RID
var uv_buffer: RID
var material_buffer: RID
var bvh_buffer: RID
var light_buffer: RID

var output_image: RID
var output_texture: Texture2DRD
var accumulation_image: RID
var output_size := Vector2i.ZERO

var albedo_texture_array: RID
var normal_texture_array: RID
var orm_texture_array: RID
var texture_sampler: RID

var camera: Camera3D
var previous_camera_transform: Transform3D
var frame := 0


func _ready() -> void:
	display_rect = get_child(0)
	rd = RenderingServer.get_rendering_device()
	if rd == null:
		push_error("No RenderingDevice available (need Forward+/Mobile renderer).")
		return

	recompile_shader()
	create_output_texture(get_render_resolution())
	_initialize_raytracing_pipeline()

	if not RenderingServer.frame_pre_draw.is_connected(_on_frame_pre_draw):
		RenderingServer.frame_pre_draw.connect(_on_frame_pre_draw)

	raytracing_enabled = true

func get_render_resolution() -> Vector2i:
	if Engine.is_editor_hint():
		var vp := EditorInterface.get_editor_viewport_3d(0)
		if vp:
			return vp.size
	else:
		var vp := get_viewport()
		if vp:
			return vp.get_visible_rect().size

	return Vector2i.ZERO

func _exit_tree() -> void:
	if RenderingServer.frame_pre_draw.is_connected(_on_frame_pre_draw):
		RenderingServer.frame_pre_draw.disconnect(_on_frame_pre_draw)
	_free_rids()


func _free_rids() -> void:
	if rd == null:
		return
	if triangle_buffer.is_valid(): rd.free_rid(triangle_buffer)
	if normal_buffer.is_valid(): rd.free_rid(normal_buffer)
	if uv_buffer.is_valid(): rd.free_rid(uv_buffer)
	if material_buffer.is_valid(): rd.free_rid(material_buffer)
	if bvh_buffer.is_valid(): rd.free_rid(bvh_buffer)
	if light_buffer.is_valid(): rd.free_rid(light_buffer)
	if output_image.is_valid(): rd.free_rid(output_image)
	if accumulation_image.is_valid(): rd.free_rid(accumulation_image)
	if albedo_texture_array.is_valid(): rd.free_rid(albedo_texture_array)
	if normal_texture_array.is_valid(): rd.free_rid(normal_texture_array)
	if orm_texture_array.is_valid(): rd.free_rid(orm_texture_array)
	if texture_sampler.is_valid(): rd.free_rid(texture_sampler)
	if pipeline.is_valid(): rd.free_rid(pipeline)
	if shader.is_valid(): rd.free_rid(shader)


func create_output_texture(size: Vector2i) -> void:
	if size.x <= 0 or size.y <= 0:
		return

	if output_image.is_valid():
		rd.free_rid(output_image)
	if accumulation_image.is_valid():
		rd.free_rid(accumulation_image)

	var format := RDTextureFormat.new()
	format.width = size.x
	format.height = size.y
	format.format = RenderingDevice.DATA_FORMAT_R16G16B16A16_SFLOAT
	format.usage_bits = (
		RenderingDevice.TEXTURE_USAGE_STORAGE_BIT |
		RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT |
		RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT |
		RenderingDevice.TEXTURE_USAGE_CAN_COPY_TO_BIT |
		RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT |
		RenderingDevice.TEXTURE_USAGE_CPU_READ_BIT
	)

	var view := RDTextureView.new()
	output_image = rd.texture_create(format, view)
	accumulation_image = rd.texture_create(format, view)

	output_size = size
	frame = 0

	if output_texture == null:
		output_texture = Texture2DRD.new()
	output_texture.texture_rd_rid = output_image

	if display_rect:
		print("setting")
		display_rect.texture = output_texture
	print("setting no")


func recompile_shader() -> void:
	if rd == null:
		rd = RenderingServer.get_rendering_device()
	if rd == null:
		push_error("No rendering device found.")
		return

	if pipeline.is_valid(): rd.free_rid(pipeline)
	if shader.is_valid(): rd.free_rid(shader)

	var file: RDShaderFile = load(shader_file)
	if file == null:
		push_error("Failed to load shader file: %s" % shader_file)
		return

	var source := file.get_spirv()
	shader = rd.shader_create_from_spirv(source)

	if not shader.is_valid():
		push_error("Failed creating shader from SPIR-V.")
		return

	pipeline = rd.compute_pipeline_create(shader)
	frame = 0


func _initialize_raytracing_pipeline() -> void:
	if rd == null:
		return

	reset_scene_data()

	var triangles := gather_triangles()
	var bvh := build_blas(triangles)
	build_light_buffer(bvh)

	upload_scene_data(
		get_bvh_triangles(bvh),
		get_bvh_normals(bvh),
		get_bvh_uvs(bvh),
		material_data,
		get_bvh_nodes(bvh),
		light_data
	)

	upload_texture_arrays()

	setup_camera()
	reset_accumulation()


func reset_scene_data() -> void:
	material_lookup.clear()
	material_count = 0
	material_data.clear()
	light_data.clear()
	unique_materials.clear()
	albedo_textures.clear()
	normal_textures.clear()
	orm_textures.clear()


func build_light_buffer(bvh: BLAS) -> void:
	var total_area := 0.0
	for i in range(bvh.get_triangles().size()):
		var tri = bvh.get_triangles()[i]
		var material = get_material_from_index(tri.material_index)

		if material != null and material.emission_enabled and material.emission_energy_multiplier > 0 and material.emission != Color.BLACK:
			var area = tri.edge1.cross(tri.edge2).length() * 0.5
			total_area += area
			append_light(i, area, total_area)


func get_material_from_index(index: int):
	if index >= unique_materials.size():
		return null
	return unique_materials[index]


func srgb_to_linear(c: float) -> float:
	if c <= 0.04045:
		return c / 12.92
	return pow((c + 0.055) / 1.055, 2.4)


func append_light(index: int, area: float, total_area: float) -> void:
	var s := StreamPeerBuffer.new()
	s.big_endian = false
	s.put_u32(index)
	s.put_float(area)
	light_data.append_array(s.data_array)


func upload_scene_data(
		triangles: PackedFloat32Array,
		normals: PackedByteArray,
		uvs: PackedFloat32Array,
		materials: PackedByteArray,
		bvh: PackedByteArray,
		lights: PackedByteArray) -> void:

	if triangle_buffer.is_valid(): rd.free_rid(triangle_buffer)
	if normal_buffer.is_valid(): rd.free_rid(normal_buffer)
	if uv_buffer.is_valid(): rd.free_rid(uv_buffer)
	if material_buffer.is_valid(): rd.free_rid(material_buffer)
	if bvh_buffer.is_valid(): rd.free_rid(bvh_buffer)
	if light_buffer.is_valid(): rd.free_rid(light_buffer)

	if triangles.size() > 0:
		triangle_buffer = rd.storage_buffer_create(triangles.size() * 4, triangles.to_byte_array())
	if normals.size() > 0:
		normal_buffer = rd.storage_buffer_create(normals.size(), normals)
	if uvs.size() > 0:
		uv_buffer = rd.storage_buffer_create(uvs.size() * 4, uvs.to_byte_array())
	if materials.size() > 0:
		material_buffer = rd.storage_buffer_create(materials.size(), materials)
	if bvh.size() > 0:
		bvh_buffer = rd.storage_buffer_create(bvh.size(), bvh)
	if lights.size() > 0:
		light_buffer = rd.storage_buffer_create(lights.size(), lights)

	reset_accumulation()


func get_bvh_triangles(bvh: BLAS):
	return bvh.get_bvh_triangle_structure()

func get_bvh_normals(bvh: BLAS):
	return bvh.get_bvh_normal_structure()

func get_bvh_uvs(bvh: BLAS):
	return bvh.get_bvh_uv_structure()

func get_bvh_nodes(bvh: BLAS):
	return bvh.get_bvh_gpu_structure()


func setup_camera() -> void:
	var cam: Camera3D
	if Engine.is_editor_hint():
		cam = EditorInterface.get_editor_viewport_3d(0).get_camera_3d()
	else:
		cam = get_viewport().get_camera_3d()

	if cam == null:
		print("No Camera")
		return

	camera = cam


func gather_triangles() -> Array[BVHTriangle]:
	var result: Array[BVHTriangle] = []
	for mesh_instance in get_mesh_instances():
		extract_mesh_triangles(mesh_instance, result)
	return result


func extract_mesh_triangles(mesh_instance: MeshInstance3D, output: Array[BVHTriangle]) -> void:
	var mesh := mesh_instance.mesh
	var transform := mesh_instance.global_transform

	for surface in mesh.get_surface_count():
		var arrays = mesh.surface_get_arrays(surface)

		var vertices: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
		var normals: PackedVector3Array = arrays[Mesh.ARRAY_NORMAL]
		var indices: PackedInt32Array = arrays[Mesh.ARRAY_INDEX]
		var texcoords: PackedVector2Array = arrays[Mesh.ARRAY_TEX_UV]

		var material_index := get_material_index(mesh_instance, surface)

		for i in range(0, indices.size(), 3):
			var i0 = indices[i]
			var i1 = indices[i + 1]
			var i2 = indices[i + 2]

			var v0 = transform * vertices[i0]
			var v1 = transform * vertices[i1]
			var v2 = transform * vertices[i2]

			var tri := BVHTriangle.new()
			tri.v0 = v0
			tri.v1 = v1
			tri.v2 = v2
			tri.edge1 = v1 - v0
			tri.edge2 = v2 - v0
			tri.n0 = (transform.basis * normals[i0]).normalized()
			tri.n1 = (transform.basis * normals[i1]).normalized()
			tri.n2 = (transform.basis * normals[i2]).normalized()
		
			if texcoords.size() > 0:
				tri.uv0 = texcoords[i0]
				tri.uv1 = texcoords[i1]
				tri.uv2 = texcoords[i2]
				
			else:
				tri.uv0 = Vector2.ZERO
				tri.uv1 = Vector2.ZERO
				tri.uv2 = Vector2.ZERO

			tri.material_index = material_index

			tri.aabb_min = v0.min(v1).min(v2)
			tri.aabb_max = v0.max(v1).max(v2)
			tri.centroid_x = (tri.aabb_min.x + tri.aabb_max.x) * 0.5
			tri.centroid_y = (tri.aabb_min.y + tri.aabb_max.y) * 0.5
			tri.centroid_z = (tri.aabb_min.z + tri.aabb_max.z) * 0.5

			output.append(tri)


func get_material_index(mesh_instance: MeshInstance3D, surface: int) -> int:
	var material = mesh_instance.mesh.surface_get_material(surface)
	if material == null:
		material = mesh_instance.get_active_material(surface)
	if material == null:
		return 0

	if material_lookup.has(material):
		return material_lookup[material]

	var index := material_count
	material_lookup[material] = index
	unique_materials.append(material)

	if material is StandardMaterial3D:
		append_material(material_data, material)
	else:
		append_material(material_data, StandardMaterial3D.new())

	material_count += 1
	return index


func build_blas(triangles: Array[BVHTriangle]) -> BLAS:
	return BLAS.new(triangles)


func get_or_add_texture_index(texture: Texture2D, list: Array[Texture2D]) -> int:
	if texture == null:
		return -1

	var idx := list.find(texture)
	if idx != -1:
		return idx

	list.append(texture)
	return list.size() - 1


func append_material(bytes: PackedByteArray, material: StandardMaterial3D) -> void:
	var s := StreamPeerBuffer.new()
	s.big_endian = false

	s.put_float(srgb_to_linear(material.albedo_color.r))
	s.put_float(srgb_to_linear(material.albedo_color.g))
	s.put_float(srgb_to_linear(material.albedo_color.b))
	s.put_float(1.0)

	s.put_float(srgb_to_linear(1.0))
	s.put_float(srgb_to_linear(1.0))
	s.put_float(srgb_to_linear(1.0))
	s.put_float(material.metallic_specular)

	s.put_float(material.emission.r)
	s.put_float(material.emission.g)
	s.put_float(material.emission.b)
	s.put_float(material.emission_energy_multiplier)

	s.put_float(material.roughness)
	s.put_float(material.metallic)
	s.put_float(1.0 - material.albedo_color.a)
	s.put_float(1.33)

	var albedo_id := get_or_add_texture_index(material.albedo_texture, albedo_textures)
	var normal_id := get_or_add_texture_index(material.normal_texture, normal_textures)
	var orm_id := get_or_add_texture_index(material.roughness_texture, orm_textures)

	s.put_32(albedo_id)
	s.put_32(normal_id)
	s.put_32(orm_id)
	s.put_float(0.0)

	bytes.append_array(s.data_array)


func upload_texture_arrays() -> void:
	if not texture_sampler.is_valid():
		var sampler_state := RDSamplerState.new()
		sampler_state.mag_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
		sampler_state.min_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
		sampler_state.mip_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
		sampler_state.repeat_u = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
		sampler_state.repeat_v = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
		texture_sampler = rd.sampler_create(sampler_state)

	if albedo_texture_array.is_valid(): rd.free_rid(albedo_texture_array)
	if normal_texture_array.is_valid(): rd.free_rid(normal_texture_array)
	if orm_texture_array.is_valid(): rd.free_rid(orm_texture_array)

	albedo_texture_array = build_texture_array(albedo_textures, true)
	normal_texture_array = build_texture_array(normal_textures, false)
	orm_texture_array = build_texture_array(orm_textures, false)


func build_texture_array(textures: Array[Texture2D], is_srgb: bool) -> RID:
	var layer_count = max(textures.size(), 1)

	var format := RDTextureFormat.new()
	format.width = TEXTURE_ARRAY_SIZE.x
	format.height = TEXTURE_ARRAY_SIZE.y
	format.array_layers = layer_count
	format.texture_type = RenderingDevice.TEXTURE_TYPE_2D_ARRAY
	format.format = RenderingDevice.DATA_FORMAT_R8G8B8A8_SRGB if is_srgb else RenderingDevice.DATA_FORMAT_R8G8B8A8_UNORM
	format.usage_bits = (
		RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT |
		RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT |
		RenderingDevice.TEXTURE_USAGE_CAN_COPY_TO_BIT
	)

	var view := RDTextureView.new()
	var layers: Array[PackedByteArray] = []

	if textures.is_empty():
		var blank := Image.create(TEXTURE_ARRAY_SIZE.x, TEXTURE_ARRAY_SIZE.y, false, Image.FORMAT_RGBA8)
		blank.fill(Color(1.0, 1.0, 1.0, 1.0))
		layers.append(blank.get_data())
	else:
		for tex in textures:
			layers.append(prepare_layer_image(tex))

	var combined := PackedByteArray()
	for layer in layers:
		combined.append_array(layer)

	
	return rd.texture_create(format, view, layers)


func prepare_layer_image(tex: Texture2D) -> PackedByteArray:
	var img := tex.get_image()

	if img == null:
		img = Image.create(TEXTURE_ARRAY_SIZE.x, TEXTURE_ARRAY_SIZE.y, false, Image.FORMAT_RGBA8)
		img.fill(Color(1.0, 1.0, 1.0, 1.0))
		return img.get_data()

	if img.is_compressed():
		img.decompress()

	if img.get_format() != Image.FORMAT_RGBA8:
		img.convert(Image.FORMAT_RGBA8)

	if img.get_size() != TEXTURE_ARRAY_SIZE:
		img.resize(TEXTURE_ARRAY_SIZE.x, TEXTURE_ARRAY_SIZE.y, Image.INTERPOLATE_LANCZOS)

	if img.has_mipmaps():
		img.clear_mipmaps()
	
	return img.get_data()


func get_mesh_instances() -> Array[MeshInstance3D]:
	var result: Array[MeshInstance3D] = []
	
	var root_node = get_parent()
	if root_node:
		collect_meshes(root_node, result)
	return result


func collect_meshes(node: Node, result: Array[MeshInstance3D]) -> void:
	if node is MeshInstance3D and node.mesh:
		result.append(node)
	for child in node.get_children():
		collect_meshes(child, result)


func reset_accumulation() -> void:
	frame = 0


func _on_frame_pre_draw() -> void:
	if not raytracing_enabled:
		return
	if rd == null or not pipeline.is_valid() or camera == null:
		return
	if not output_image.is_valid() or not accumulation_image.is_valid():
		return
	if not triangle_buffer.is_valid() or not bvh_buffer.is_valid() or not uv_buffer.is_valid():
		return
	if not albedo_texture_array.is_valid() or not normal_texture_array.is_valid() or not orm_texture_array.is_valid():
		return

	var current_size := get_render_resolution()

	if current_size != output_size and current_size.x > 0 and current_size.y > 0:
		create_output_texture(current_size)

	if frame >= max_samples:
		var render = display_rect.texture.get_image()
		render.save_png("res://" + image_save_name + ".png")
		return

	if not camera.global_transform.is_equal_approx(previous_camera_transform):
		reset_accumulation()
		previous_camera_transform = camera.global_transform

	var groups_x := int((output_size.x + 31) / 32)
	var groups_y := int((output_size.y + 31) / 32)
	var push_constants := get_camera_params(output_size)

	var u_image := RDUniform.new()
	u_image.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_image.binding = 0
	u_image.add_id(output_image)

	var u_triangles := RDUniform.new()
	u_triangles.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_triangles.binding = 1
	u_triangles.add_id(triangle_buffer)

	var u_normals := RDUniform.new()
	u_normals.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_normals.binding = 2
	u_normals.add_id(normal_buffer)

	var u_materials := RDUniform.new()
	u_materials.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_materials.binding = 3
	u_materials.add_id(material_buffer)

	var u_bvh := RDUniform.new()
	u_bvh.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_bvh.binding = 4
	u_bvh.add_id(bvh_buffer)

	var u_lights := RDUniform.new()
	u_lights.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_lights.binding = 5
	u_lights.add_id(light_buffer)

	var u_accum := RDUniform.new()
	u_accum.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_accum.binding = 6
	u_accum.add_id(accumulation_image)

	var u_uvs := RDUniform.new()
	u_uvs.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_uvs.binding = 7
	u_uvs.add_id(uv_buffer)

	var uniform_set := UniformSetCacheRD.get_cache(
		shader,
		0,
		[u_image, u_triangles, u_normals, u_materials, u_bvh, u_lights, u_accum, u_uvs]
	)

	var u_albedo_tex := RDUniform.new()
	u_albedo_tex.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_albedo_tex.binding = 0
	u_albedo_tex.add_id(texture_sampler)
	u_albedo_tex.add_id(albedo_texture_array)

	var u_normal_tex := RDUniform.new()
	u_normal_tex.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_normal_tex.binding = 1
	u_normal_tex.add_id(texture_sampler)
	u_normal_tex.add_id(normal_texture_array)

	var u_orm_tex := RDUniform.new()
	u_orm_tex.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_orm_tex.binding = 2
	u_orm_tex.add_id(texture_sampler)
	u_orm_tex.add_id(orm_texture_array)

	var texture_uniform_set := UniformSetCacheRD.get_cache(
		shader,
		2,
		[u_albedo_tex, u_normal_tex, u_orm_tex]
	)

	var compute_list := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, pipeline)
	rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
	rd.compute_list_bind_uniform_set(compute_list, texture_uniform_set, 2)
	rd.compute_list_set_push_constant(
		compute_list,
		push_constants.to_byte_array(),
		push_constants.size() * 4
	)
	rd.compute_list_dispatch(compute_list, groups_x, groups_y, 1)
	rd.compute_list_end()

	frame += 1
	print("Sample: ", frame)


func get_camera_params(raster_size: Vector2i) -> PackedFloat32Array:
	if camera == null:
		return PackedFloat32Array()

	var near := camera.near
	var plane_height := near * tan(deg_to_rad(camera.fov * 0.5)) * 2.0
	var plane_width := plane_height * (float(raster_size.x) / float(raster_size.y))
	var view_params := Vector3(plane_width, plane_height, near)
	var t := camera.global_transform

	var pc := PackedFloat32Array()
	pc.push_back(float(raster_size.x))
	pc.push_back(float(raster_size.y))
	pc.push_back(0.0)
	pc.push_back(0.0)

	pc.push_back(view_params.x)
	pc.push_back(view_params.y)
	pc.push_back(view_params.z)
	pc.push_back(float(frame))

	pc.append_array([
		t.basis.x.x, t.basis.x.y, t.basis.x.z, 0.0,
		t.basis.y.x, t.basis.y.y, t.basis.y.z, 0.0,
		t.basis.z.x, t.basis.z.y, t.basis.z.z, 0.0,
		t.origin.x,  t.origin.y,  t.origin.z,  1.0,
	])

	return pc
