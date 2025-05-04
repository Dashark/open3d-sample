import vtk


def main():
    # 创建一个球体源作为三维曲面
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(1.0)
    sphere.SetThetaResolution(30)
    sphere.SetPhiResolution(30)
    sphere.Update()

    # 读取纹理图像
    reader = vtk.vtkJPEGReader()
    reader.SetFileName("texture.jpg")  # 请确保该图像文件存在

    texture = vtk.vtkTexture()
    texture.SetInputConnection(reader.GetOutputPort())
    texture.InterpolateOn()  # 启用纹理插值使纹理更平滑

    # 创建纹理坐标
    texture_map = vtk.vtkTextureMapToSphere()
    texture_map.SetInputConnection(sphere.GetOutputPort())
    texture_map.PreventSeamOn()  # 防止在球体背面出现接缝

    # 创建映射器和演员
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(texture_map.GetOutputPort())  # 使用带有纹理坐标的数据

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.SetTexture(texture)

    # 创建渲染器、渲染窗口和交互器
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)  # 设置窗口大小
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # 将演员添加到渲染器中
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.3)
    renderer.ResetCamera()  # 重置相机以确保对象可见

    # 坐标系
    # axes = vtk.vtkAxesActor()
    # renderer.AddActor(axes)

    # 开始交互
    render_window.Render()
    interactor.Start()


if __name__ == "__main__":
    main()