# Renderppl Blender Render Tool
free-view rendering static renderppl dataset, lighting applied.

# Renderppl Static Dataset
You can download the [RenderPeople](https://renderpeople.com/free-3d-people/) one free model [here](https://renderpeople.com/sample/free/rp_dennis_posed_004_OBJ.zip), and test the rendering code with this model.

* input dataset folder structure

```
├── renderppl_sample/
	├── <subject1_OBJ>
	├── <subject2_OBJ>
	│	├── {subject2}_100k.obj
	│	└── tex/{subject2}_dif_2k.obj
	└── ...      
```

* output dataset folder structure

```
├── free_view_renderppl/
	├── camera
		├── <subject1>
		├── <subject2>
		│	├── cam_K.txt
		│	├── cam_RT_001.txt
		│	└── ...
		└── ... 
	├── depth
		├── <subject1>
		├── <subject2>
		│	├── depth_001.exr
		│	├── depth_002.exr
		│	└── ...
		└── ... 
	├── image
		├── <subject1>
		├── <subject2>
		│	├── color_001.png
		│	├── color_002.png
		│	└── ...
		└── ... 
	├── normal
		├── <subject1>
		├── <subject2>
		│	├── normal_001.png
		│	├── normal_002.png
		│	└── ...
		└── ... 
	├── geometry
		├── <subject1>
		├── <subject2>
		│	├── <subject2>.mtl
		│	└── <subject2>.obj
		└── ... 
	├── data
		├── all.txt
		├── train.txt
		└── val.txt
```

# Dependences
* blender 2.93.4
```
cd ./renderppl-dataset
wget https://mirror.clarkson.edu/blender/release/Blender2.93/blender-2.93.4-linux-x64.tar.xz -O ./bin/blender-2.93.4-linux-x64.tar.xz
tar -xf ./bin/blender-2.93.4-linux-x64.tar.xz -C ./bin/
```

# Usage
* run free-view blender render
```
$ ./bin/blender-2.93.4-linux-x64/blender ./bin/for_free_view.blend --background --python ./render/run.py
```

# Outputs
* camera KRT (computer vision camera coordinates)
	- x is horizontal
	- y is down
	- right-handed: positive z look-at direction

|- camera K (3x3)
```
7.111110839843750000e+02 0.000000000000000000e+00 2.560000000000000000e+02
0.000000000000000000e+00 7.111110839843750000e+02 2.560000000000000000e+02
0.000000000000000000e+00 0.000000000000000000e+00 1.000000000000000000e+00
```

|- camera RT (3x4)
	
```
-1.757705658674240112e-01 -5.411332182347905473e-08 -9.844312071800231934e-01 6.565072059631347656e+00
-5.411332182347905473e-08 -1.000000000000000000e+00 7.727678053015551995e-09 9.211421203613281250e+01
-9.844312071800231934e-01 1.215344695992826018e-07 1.757705062627792358e-01 3.042410888671875000e+02
```
* color images (transparent background: RGBA)
<p float="left">
  <img src="https://github.com/Junyingw/renderppl-dataset/blob/master/examples/aaron_image.png" width="384" height="256">
  <img src="https://github.com/Junyingw/renderppl-dataset/blob/master/examples/julia_image.png" width="384" height="256">
</p>

* normal maps (exr: 3 channels; object view)
<p float="left">
  <img src="https://github.com/Junyingw/renderppl-dataset/blob/master/examples/aaron_normal.png" width="384" height="256">
  <img src="https://github.com/Junyingw/renderppl-dataset/blob/master/examples/julia_normal.png" width="384" height="256">
</p>

* depth maps (exr: 3 channels)
<p float="left">
  <img src="https://github.com/Junyingw/renderppl-dataset/blob/master/examples/aaron_depth.png" width="384" height="256">
  <img src="https://github.com/Junyingw/renderppl-dataset/blob/master/examples/julia_depth.png" width="384" height="256">
</p>

* mesh with texture
<p float="left">
  <img src="https://github.com/Junyingw/renderppl-dataset/blob/master/examples/aaron_mesh.png" width="384" height="256">
  <img src="https://github.com/Junyingw/renderppl-dataset/blob/master/examples/julia_mesh.png" width="384" height="256">
</p>

