# Godot Pathtracer

This project is a compute shader implementation of a simple pathtracer written in glsl. It was made as a learning project to get more familiar with raytracing and compute shaders. 
It supports all meshes which get a BVH built on the cpu, basic material settings, emmisive surfaces as lights and textures, only if all texture sizes match.

## Features

The pathtracer parses the scene at start, gets all of the mesh instances in the scene and builds a BVH from the triangle pool. It binds materials and their properties.
Currently the pathtracer is very simple, so it doesnt support different sized textures, light objects and more... 

All material properties are derived from the StandardMaterial3D properties.

List of supported material properties:
  Albedo (Albedo.rgb) - Textures are supported
  Emmision (Emmision.rgb - color, Emmision.a - strength)
  Roughness - Textures are supported
  Metallic - Textures are supported
  Specular
  Transmittance (1.0 - Albedo.a)
  IOR (Set to a fixed value, 1.33, can be changed in code)

  *Roughness, Metallic and Transmittance textures are placed into one texture for better memory packing.

## Showcase

<img width="1918" height="1079" alt="car_render_0" src="https://github.com/user-attachments/assets/405184df-dc86-449d-a94c-2100f0c33067" />

<img width="1280" height="720" alt="dragon_render_0" src="https://github.com/user-attachments/assets/227e5e00-1ec5-4c9b-ac94-36345924dfe0" />

<img width="1280" height="720" alt="car_render_2" src="https://github.com/user-attachments/assets/59f5adb9-e73d-44ba-afcb-2d8aede7775d" />

<img width="1280" height="720" alt="bunny" src="https://github.com/user-attachments/assets/3aa3c3b7-3fe7-4b54-82ac-2f3a43a8116b" />
<img width="1280" height="720" alt="ball_render_0" src="https://github.com/user-attachments/assets/93407a94-ade8-415e-952e-71e1ae8d4771" />

