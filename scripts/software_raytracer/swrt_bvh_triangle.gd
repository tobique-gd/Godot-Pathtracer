extends RefCounted
class_name BVHTriangle


var v0: Vector3
var v1: Vector3
var v2: Vector3

var edge1: Vector3
var edge2: Vector3

var n0: Vector3
var n1: Vector3
var n2: Vector3

var material_index := 0

var aabb_min: Vector3
var aabb_max: Vector3
var centroid_x: float
var centroid_y: float
var centroid_z: float

var uv0 : Vector2
var uv1 : Vector2
var uv2 : Vector2

func center() -> Vector3:
	return (aabb_min + aabb_max) * 0.5
