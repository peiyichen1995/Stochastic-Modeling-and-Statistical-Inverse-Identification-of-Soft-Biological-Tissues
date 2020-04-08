import meshio
msh = meshio.read("media_flatboundaries.msh")
for cell in msh.cells:
    if cell.type == "triangle":
        triangle_cells = cell.data
    elif  cell.type == "tetra":
        tetra_cells = cell.data

for key in msh.cell_data_dict["gmsh:physical"].keys():
    if key == "triangle":
        triangle_data = msh.cell_data_dict["gmsh:physical"][key]
    elif key == "tetra":
        tetra_data = msh.cell_data_dict["gmsh:physical"][key]

tetra_mesh = meshio.Mesh(points=msh.points, cells={"tetra": tetra_cells})
triangle_mesh = meshio.Mesh(points=msh.points,
                           cells=[("triangle", triangle_cells)],
                           cell_data={"face_id":[triangle_data]})

meshio.write("mesh.xdmf", tetra_mesh)

meshio.write("mf.xdmf", triangle_mesh)

# import meshio
# from dolfin import Mesh, XDMFFile, File, MeshValueCollection, cpp, Measure,\
#                    DirichletBC, FunctionSpace, Constant, TrialFunction, \
#                    TestFunction, dot, grad, dx, Function, solve
#
# msh = meshio.read("media_flatboundaries.msh")
# meshio.write("mesh.xdmf",
#              meshio.Mesh(points=msh.points,
#                          cells={"tetra10": msh.cells["tetra10"]}))
#
# meshio.write("mf.xdmf",
#              meshio.Mesh(points=msh.points,
#                          cells={"triangle6": msh.cells["triangle6"]},
#                          cell_data={"triangle6": {"name_to_read": msh.cell_data["triangle6"]["gmsh:physical"]}}))
