## Reconstruction & Mesh Generation Parameters

| Parameter name        | Parameter value   | Description                                                                 |
|-----------------------|-------------------|-----------------------------------------------------------------------------|
| **ANTIALIAS_ITERATIONS** | 30                | Number of iterations used for surface smoothing, improving mesh regularity. |
| **ISO_SPACING**          | [1, 1, 1] mm      | Isotropic voxel spacing used for interpolation to ensure spatial uniformity. |
| **PAD_SIZE**             | 5 voxels          | Number of voxels added as padding to prevent boundary artifacts.            |
| **PAD_VALUE**            | 0                 | Padding intensity value; zero-padding to avoid artificial edge structures.  |
| **ISO_VALUE**            | 0.5               | Isosurface extraction threshold for faithful surface reconstruction.         |
| **ICP_ITERATIONS**       | 200               | Number of iterations for ICP to standardize orientation and position.       |


## Shape Modeling Parameters

| Parameter name            | Parameter value                         | Description                                                                 |
|---------------------------|------------------------------------------|-----------------------------------------------------------------------------|
| **number_of_particles**   | - 128 for RF and TA<br>- 256 for VL     | Number of particles per shape; defines the granularity of surface sampling and shape variability. |
| **ending_regularization** | 1                                        | Final level of regularization, allowing finer particle adjustment as optimization proceeds. |
| **starting_regularization** | 100                                    | Initial level of regularization to prevent convergence to local minima. |
| **initial_relative_weighting** | 0.1                                 | Initial weighting value for the correspondence–regularization trade-off at early stages. |
| **mesh_mode**             | True                                     | Enables mesh-based optimization, accounting for the 3D surface geometry. |
