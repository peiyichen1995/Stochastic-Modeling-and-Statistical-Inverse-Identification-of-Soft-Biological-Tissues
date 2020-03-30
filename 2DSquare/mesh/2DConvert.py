import meshio
msh = meshio.read("square.msh")
for cell in msh.cells:
    if cell.type == "line":
        line_cells = cell.data
    elif  cell.type == "triangle":
        triangle_cells = cell.data

for key in msh.cell_data_dict["gmsh:physical"].keys():
    if key == "line":
        line_data = msh.cell_data_dict["gmsh:physical"][key]
    elif key == "triangle":
        triangle_data = msh.cell_data_dict["gmsh:physical"][key]

triangle_mesh = meshio.Mesh(points=msh.points, cells={"triangle": triangle_cells})
line_mesh = meshio.Mesh(points=msh.points,
                           cells=[("line", line_cells)],
                           cell_data={"face_id":[line_data]})

meshio.write("mesh.xdmf", triangle_mesh)

meshio.write("mf.xdmf", line_mesh)

# import meshio
# from dolfin import Mesh, XDMFFile, File, MeshValueCollection, cpp, Measure,\
#                    DirichletBC, FunctionSpace, Constant, TrialFunction, \
#                    TestFunction, dot, grad, dx, Function, solve
#
# msh = meshio.read("media_flatboundaries.msh")
# meshio.write("mesh.xdmf",
#              meshio.Mesh(points=msh.points,
#                          cells={"triangle10": msh.cells["triangle10"]}))
#
# meshio.write("mf.xdmf",
#              meshio.Mesh(points=msh.points,
#                          cells={"line6": msh.cells["line6"]},
#                          cell_data={"line6": {"name_to_read": msh.cell_data["line6"]["gmsh:physical"]}}))
