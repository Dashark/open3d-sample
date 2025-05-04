import vtk

# 创建一个球体源
sphere = vtk.vtkSphereSource()
sphere.SetRadius(1.0)
sphere.SetThetaResolution(30)
sphere.SetPhiResolution(30)
sphere.Update()

# 创建映射器和演员
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(sphere.GetOutputPort())

actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # 设置球体颜色为红色

# 创建一个平行光
light = vtk.vtkLight()
light.SetLightTypeToCameraLight()  # 设置为场景光
light.SetPosition(10, 10, 10)  # 设置光源位置
light.SetDiffuseColor(1.0, 1.0, 1.0)  # 设置漫反射颜色
light.SetSpecularColor(10, 1.0, 1.0)  # 设置镜面反射颜色
light.SetIntensity(1.0)  # 设置光照强度

# 创建渲染器、渲染窗口和交互器
renderer = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

# 将演员和光源添加到渲染器中
renderer.AddActor(actor)
renderer.AddLight(light)
renderer.SetBackground(0.1, 0.2, 0.3)

# 开始交互
render_window.Render()
interactor.Start()
    