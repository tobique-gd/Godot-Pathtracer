@tool
extends CompositorEffect
class_name SoftwareRaytracing

var shader_file := "res://assets/shaders/raytracer.glsl"

var rd: RenderingDevice = RenderingServer.get_rendering_device()
var shader: RID
var pipeline: RID
var camera: Camera3D

var triangle_buffer: RID
var normal_buffer: RID
var material_buffer: RID
var bvh_buffer: RID
var light_buffer: RID

var frame := 0
var previous_camera_transform: Transform3D
var accumulation_image: RID
var accumulation_size := Vector2i.ZERO
var MAX_SAMPLES := 1024

func _init() -> void:
	recompile_shader()


func create_accumulation_texture(size: Vector2i) -> void:
	if accumulation_image.is_valid():
		rd.free_rid(accumulation_image)

	var format := RDTextureFormat.new()
	format.width = size.x
	format.height = size.y
	format.format = RenderingDevice.DATA_FORMAT_R16G16B16A16_SFLOAT
	format.usage_bits = (
		RenderingDevice.TEXTURE_USAGE_STORAGE_BIT |
		RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT |
		RenderingDevice.TEXTURE_USAGE_CAN_COPY_TO_BIT
	)

	accumulation_image = rd.texture_create(format, RDTextureView.new())
	accumulation_size = size
	
	reset_accumulation()


func reset_accumulation() -> void:
	frame = 0
	if accumulation_image.is_valid():
		rd.texture_clear(
			accumulation_image,
			Color.BLACK,
			0, 1, 0, 1
		)


func upload_scene(
	triangles: PackedFloat32Array,
	normals: PackedByteArray,
	materials: PackedByteArray,
	bvh: PackedByteArray,
	lights: PackedByteArray
) -> void:
	if triangle_buffer.is_valid(): rd.free_rid(triangle_buffer)
	if normal_buffer.is_valid(): rd.free_rid(normal_buffer)
	if material_buffer.is_valid(): rd.free_rid(material_buffer)
	if bvh_buffer.is_valid(): rd.free_rid(bvh_buffer)
	if light_buffer.is_valid(): rd.free_rid(light_buffer)

	if triangles.size() > 0:
		triangle_buffer = rd.storage_buffer_create(triangles.size() * 4, triangles.to_byte_array())
	if normals.size() > 0:
		normal_buffer = rd.storage_buffer_create(normals.size(), normals)
	if materials.size() > 0:
		material_buffer = rd.storage_buffer_create(materials.size(), materials)
	if bvh.size() > 0:
		bvh_buffer = rd.storage_buffer_create(bvh.size(), bvh)
	if lights.size() > 0:
		light_buffer = rd.storage_buffer_create(lights.size(), lights)

	reset_accumulation()


func recompile_shader() -> void:

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

	if !shader.is_valid():
		push_error("Failed creating shader from SPIR-V.")
		return

	pipeline = rd.compute_pipeline_create(shader)


func _render_callback(callback_type: int, render_data: RenderData) -> void:
	if callback_type != EFFECT_CALLBACK_TYPE_POST_TRANSPARENT:
		return

	if !pipeline.is_valid() or not camera:
		return

	var buffers := render_data.get_render_scene_buffers() as RenderSceneBuffersRD
	if buffers == null:
		return

	var size := buffers.get_internal_size()
	if size.x == 0 or size.y == 0:
		return

	if size != accumulation_size:
		create_accumulation_texture(size)

	# Reset frame accumulation if camera moved
	if not camera.global_transform.is_equal_approx(previous_camera_transform):
		reset_accumulation()
		previous_camera_transform = camera.global_transform

	var groups_x := int((size.x + 7) / 8)
	var groups_y := int((size.y + 7) / 8)
	var push_constants := get_camera_params(size)

	for view in range(buffers.get_view_count()):
		var image := buffers.get_color_layer(view)

		# Build uniform array
		var u_image := RDUniform.new()
		u_image.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
		u_image.binding = 0
		u_image.add_id(image)

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

		var uniform_set := UniformSetCacheRD.get_cache(
			shader,
			0,
			[u_image, u_triangles, u_normals, u_materials, u_bvh, u_lights, u_accum]
		)

		var compute_list := rd.compute_list_begin()
		rd.compute_list_bind_compute_pipeline(compute_list, pipeline)
		rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
		rd.compute_list_set_push_constant(
			compute_list,
			push_constants.to_byte_array(),
			push_constants.size() * 4
		)
		rd.compute_list_dispatch(compute_list, groups_x, groups_y, 1)
		rd.compute_list_end()

	frame += 1
	
	if frame >= MAX_SAMPLES:
		pass


func get_camera_params(raster_size: Vector2i) -> PackedFloat32Array:
	if not camera:
		return PackedFloat32Array()

	var near := camera.near
	var plane_height := near * tan(deg_to_rad(camera.fov * 0.5)) * 2.0
	var plane_width = plane_height * camera.get_viewport().size.aspect()
	var view_params := Vector3(plane_width, plane_height, near)
	var t := camera.global_transform

	var pc := PackedFloat32Array()
	pc.push_back(float(raster_size.x))
	pc.push_back(float(raster_size.y))
	pc.push_back(0.0) # Padding
	pc.push_back(0.0) # Padding

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

func _notification(what: int) -> void:
	if what == NOTIFICATION_PREDELETE:
		if triangle_buffer.is_valid(): rd.free_rid(triangle_buffer)
		if normal_buffer.is_valid(): rd.free_rid(normal_buffer)
		if material_buffer.is_valid(): rd.free_rid(material_buffer)
		if bvh_buffer.is_valid(): rd.free_rid(bvh_buffer)
		if light_buffer.is_valid(): rd.free_rid(light_buffer)
		if accumulation_image.is_valid(): rd.free_rid(accumulation_image)
		if shader.is_valid(): rd.free_rid(shader)
		if pipeline.is_valid(): rd.free_rid(pipeline)
