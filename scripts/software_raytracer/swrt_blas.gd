extends Node
class_name BLAS

const SAH_BINS = 8
const MAX_LEAF_SIZE = 4

var triangle_buffer: Array[BVHTriangle] = []
var gpu_triangle_buffer := PackedFloat32Array()
var all_nodes: Array[BVHNode] = []

func surface_area(min: Vector3, max: Vector3) -> float:
	var e = max - min
	return 2.0 * (e.x * e.y + e.x * e.z + e.y * e.z)

func get_bvh_uv_structure() -> PackedFloat32Array:
	var gpu_uv_buffer := PackedFloat32Array()

	for tri in triangle_buffer:
		gpu_uv_buffer.append(tri.uv0.x)
		gpu_uv_buffer.append(tri.uv0.y)
		gpu_uv_buffer.append(tri.uv1.x)
		gpu_uv_buffer.append(tri.uv1.y)

		gpu_uv_buffer.append(tri.uv2.x)
		gpu_uv_buffer.append(tri.uv2.y)
		gpu_uv_buffer.append(0.0)
		gpu_uv_buffer.append(0.0)



	return gpu_uv_buffer

func get_centroid_component(tri: BVHTriangle, axis: int) -> float:
	match axis:
		0:
			return tri.centroid_x
		1:
			return tri.centroid_y
		_:
			return tri.centroid_z

func _init(triangles: Array[BVHTriangle]):

	triangle_buffer = triangles.duplicate()

	var root := BVHNode.new()
	root.triangle_offset = 0
	root.triangle_count = triangle_buffer.size()
	
	calculate_bounds(root)

	all_nodes.append(root)
	
	split(0, 0)



func get_triangles() -> Array[BVHTriangle]:
	return triangle_buffer

func calculate_bounds(node: BVHNode):
	node.aabb_min = Vector3.INF
	node.aabb_max = -Vector3.INF

	for i in range(node.triangle_offset, node.triangle_offset + node.triangle_count):
		var tri = triangle_buffer[i]
		node.aabb_min = node.aabb_min.min(tri.aabb_min)
		node.aabb_max = node.aabb_max.max(tri.aabb_max)

func get_bvh_triangle_structure() -> PackedFloat32Array:
	gpu_triangle_buffer.clear()

	for tri in triangle_buffer:
		gpu_triangle_buffer.append(tri.v0.x)
		gpu_triangle_buffer.append(tri.v0.y)
		gpu_triangle_buffer.append(tri.v0.z)
		gpu_triangle_buffer.append(float(tri.material_index))

		gpu_triangle_buffer.append(tri.edge1.x)
		gpu_triangle_buffer.append(tri.edge1.y)
		gpu_triangle_buffer.append(tri.edge1.z)
		gpu_triangle_buffer.append(0.0)

		gpu_triangle_buffer.append(tri.edge2.x)
		gpu_triangle_buffer.append(tri.edge2.y)
		gpu_triangle_buffer.append(tri.edge2.z)
		gpu_triangle_buffer.append(0.0)

	return gpu_triangle_buffer

func get_bvh_normal_structure() -> PackedByteArray:
	var bytes := PackedByteArray()

	for tri in triangle_buffer:
		var s := StreamPeerBuffer.new()
		s.big_endian = false

		s.put_half(tri.n0.x)
		s.put_half(tri.n0.y)
		s.put_half(tri.n0.z)
		s.put_u16(0)

		s.put_half(tri.n1.x)
		s.put_half(tri.n1.y)
		s.put_half(tri.n1.z)
		s.put_u16(0)

		s.put_half(tri.n2.x)
		s.put_half(tri.n2.y)
		s.put_half(tri.n2.z)
		s.put_u16(0)

		bytes.append_array(s.data_array)

	return bytes

func float_to_half_bytes(f: float) -> PackedByteArray:
	var s := StreamPeerBuffer.new()
	s.big_endian = false
	s.put_half(f)
	return s.data_array

func get_bvh_structure() -> Array[BVHNode]:
	return all_nodes

func get_bvh_gpu_structure() -> PackedByteArray:
	var bytes := PackedByteArray()

	for node in all_nodes:
		var s := StreamPeerBuffer.new()
		s.big_endian = false

		s.put_float(node.aabb_min.x)
		s.put_float(node.aabb_min.y)
		s.put_float(node.aabb_min.z)
		s.put_float(0.0)

		s.put_float(node.aabb_max.x)
		s.put_float(node.aabb_max.y)
		s.put_float(node.aabb_max.z)
		s.put_float(0.0)

		s.put_u32(node.child_index)
		s.put_u32(node.triangle_offset)
		s.put_u32(node.triangle_count)
		s.put_u32(0)

		bytes.append_array(s.data_array)

	return bytes

func split(node_index:int, depth:=0):

	var parent = all_nodes[node_index]

	if parent.triangle_count <= MAX_LEAF_SIZE:
		return


	var best_axis = -1
	var best_split = -1
	var best_cost = INF


	var parent_area = surface_area(
		parent.aabb_min,
		parent.aabb_max
	)


	for axis in range(3):

		var centroid_min = INF
		var centroid_max = -INF


		for i in range(
			parent.triangle_offset,
			parent.triangle_offset + parent.triangle_count
		):

			var c = get_centroid_component(triangle_buffer[i], axis)

			centroid_min = min(centroid_min,c)
			centroid_max = max(centroid_max,c)


		if centroid_max == centroid_min:
			continue


		var bins = []

		for i in range(SAH_BINS):
			bins.append({
				"count":0,
				"min":Vector3.INF,
				"max":-Vector3.INF
			})


		var scale = SAH_BINS / (centroid_max-centroid_min)


		for i in range(
			parent.triangle_offset,
			parent.triangle_offset + parent.triangle_count
		):

			var tri = triangle_buffer[i]

			var id = int(
				(get_centroid_component(tri, axis)-centroid_min)*scale
			)

			id = clamp(id,0,SAH_BINS-1)


			bins[id].count += 1
			bins[id].min = bins[id].min.min(tri.aabb_min)
			bins[id].max = bins[id].max.max(tri.aabb_max)



		var left_count=[]
		var right_count=[]

		var left_min=[]
		var left_max=[]

		var right_min=[]
		var right_max=[]


		var c=0
		var mn=Vector3.INF
		var mx=-Vector3.INF


		for i in range(SAH_BINS):

			c += bins[i].count
			mn = mn.min(bins[i].min)
			mx = mx.max(bins[i].max)

			left_count.append(c)
			left_min.append(mn)
			left_max.append(mx)



		c=0
		mn=Vector3.INF
		mx=-Vector3.INF


		for i in range(SAH_BINS-1,-1,-1):

			c += bins[i].count
			mn = mn.min(bins[i].min)
			mx = mx.max(bins[i].max)

			right_count.insert(0,c)
			right_min.insert(0,mn)
			right_max.insert(0,mx)



		for i in range(SAH_BINS-1):

			if left_count[i]==0 or right_count[i+1]==0:
				continue


			var cost = (
				left_count[i] *
				surface_area(left_min[i],left_max[i])
				+
				right_count[i+1] *
				surface_area(right_min[i+1],right_max[i+1])
			) / parent_area


			if cost < best_cost:
				best_cost=cost
				best_axis=axis
				best_split=i



	if best_axis==-1:
		return



	# partition triangles

	var centroid_min=INF
	var centroid_max=-INF


	for i in range(
		parent.triangle_offset,
		parent.triangle_offset+parent.triangle_count
	):

		var c = get_centroid_component(triangle_buffer[i], best_axis)

		centroid_min=min(centroid_min,c)
		centroid_max=max(centroid_max,c)



	var split_pos = lerp(
		centroid_min,
		centroid_max,
		float(best_split+1)/SAH_BINS
	)


	var left=parent.triangle_offset
	var right=parent.triangle_offset+parent.triangle_count-1


	while left<=right:

		while left<=right and get_centroid_component(triangle_buffer[left], best_axis) < split_pos:
			left+=1

		while left<=right and get_centroid_component(triangle_buffer[right], best_axis) >= split_pos:
			right-=1


		if left<right:
			var tmp=triangle_buffer[left]
			triangle_buffer[left]=triangle_buffer[right]
			triangle_buffer[right]=tmp

			left+=1
			right-=1



	var mid=left

	var left_size=mid-parent.triangle_offset
	var right_size=parent.triangle_count-left_size


	if left_size==0 or right_size==0:
		return



	parent.child_index=all_nodes.size()


	var a=BVHNode.new()
	a.triangle_offset=parent.triangle_offset
	a.triangle_count=left_size


	var b=BVHNode.new()
	b.triangle_offset=mid
	b.triangle_count=right_size


	calculate_bounds(a)
	calculate_bounds(b)


	all_nodes.append(a)
	all_nodes.append(b)


	split(parent.child_index,depth+1)
	split(parent.child_index+1,depth+1)
