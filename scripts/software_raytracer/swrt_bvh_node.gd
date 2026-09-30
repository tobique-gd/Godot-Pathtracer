extends Node
class_name BVHNode


var aabb_min := Vector3.INF
var aabb_max := -Vector3.INF

var child_index := 0xFFFFFFFF
var triangle_offset := 0
var triangle_count := 0
