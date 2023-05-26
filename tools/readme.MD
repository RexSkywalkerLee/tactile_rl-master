# Useful Object Creation Tools for Robotics.

In the RwS project, I created some very simple tools for rapid prototyping.

### [Object Creation] Draw Polygons
Use *polygon_drawer.py* file to draw some polygons. Leftclick to draw each point. Press ENTER to finish the drawing of each polygon.
After drawing several polygons, a $POLYGON.txt file will be generated.

### [Object Creation] Polygons to Meshes
Use *polygon_to_obj.py* file to convert any $POLYGON.txt file to some meshes, i.e., $OBJ_i.obj.

### [Preprocessing] Mesh Scaling
We can use the *object rescale.py* to rescale the .obj mehses to an appropriate size.

In our RwS project, the default mesh scale is 2m * 2m * 2m. In the urdf file, we will rescale each mesh to 0.03, resulting in a 0.6cm^3 scale.



### [Preprocessing] Mesh Convex Decomposition

Generate the convex decomposition of several .objs. Though isaacgym has vhacd file, but we find that it does not work well on the server.

To install vhacd, we can directly download the compiled vhacd binaries from the internet.
#### Usage: Step by step
1. Put all the meshes in the assets/urdf/objects/meshes/$FOLDER. 
2. Then run the *object_gen_batch.py* to decompose all the .obj files in the assets/urdf/objects/meshes/$FOLDER. 
3. The generated urdf files will be in the assets/urdf/objects/. Each urdf file will combine all the decomposed .obj files.
4. A metadata file *asset_dict.txt* will also be produced. We will need this in the simulator file to index the objects.





