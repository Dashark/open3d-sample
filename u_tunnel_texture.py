import vtk

# --------------------------
# 1. 创建并配置纹理读取器
# --------------------------
texture_reader = vtk.vtkJPEGReader()
texture_path = "texture.jpg"# 请替换纹理文件路径
texture_reader.SetFileName(texture_path)
texture_reader.Update()

texture = vtk.vtkTexture()
texture.SetInputConnection(texture_reader.GetOutputPort())
texture.InterpolateOn() # 平滑纹理效果

# --------------------------
# 2. 读取 PLY 模型
# --------------------------
ply_reader = vtk.vtkPLYReader()
ply_reader.SetFileName("true_support.ply")# 替换为实际存在的模型文件路径
ply_reader.Update()

# 获取模型数据
polydata = ply_reader.GetOutput()

# --------------------------
# 3. 检查纹理坐标，如果没有，则生成默认纹理坐标
# --------------------------
if not polydata.GetPointData().GetTCoords():
    print("模型没有纹理坐标，尝试生成默认的平面纹理坐标...")
textureMapper = vtk.vtkTextureMapToPlane()
textureMapper.SetInputData(polydata)
textureMapper.Update()
polydata = textureMapper.GetOutput()

# --------------------------
# 4. 创建映射器、演员并应用纹理
# --------------------------
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(polydata)

actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.SetTexture(texture)

# --------------------------
# 5. 设置渲染器、渲染窗口和交互器
# --------------------------
renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(0.1, 0.2, 0.3)

renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(renderWindow)

# --------------------------
# 6. 开始渲染
# --------------------------
renderWindow.Render()
interactor.Start()
