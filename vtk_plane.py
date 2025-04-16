import vtk

# 创建纹理图像读取器
reader = vtk.vtkJPEGReader()
reader.SetFileName("texture.jpg")

# 创建纹理对象
texture = vtk.vtkTexture()
texture.SetInputConnection(reader.GetOutputPort())
texture.InterpolateOn()

# 创建一个平面作为贴图对象
plane = vtk.vtkCubeSource()
#plane.SetXResolution(1)
#plane.SetYResolution(1)

# 创建映射器和演员
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(plane.GetOutputPort())

actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.SetTexture(texture)

# 创建渲染器、渲染窗口和交互器
renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

# 添加演员到渲染器
renderer.AddActor(actor)
renderer.SetBackground(0.1, 0.2, 0.3)

# 渲染和启动交互器
renderWindow.Render()
renderWindowInteractor.Start()