import vtk


def main():
    # 创建一个球体源
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

    # 定义平移距离
    translation_distance = 0.5

    # 创建裁剪平面，用于分割球体
    plane1 = vtk.vtkPlane()
    # 沿着 x 轴正方向平移
    plane1.SetOrigin(translation_distance, 0, 0)
    plane1.SetNormal(1, 0, 0)

    plane2 = vtk.vtkPlane()
    # 沿着 x 轴负方向平移
    plane2.SetOrigin(-translation_distance, 0, 0)
    plane2.SetNormal(-1, 0, 0)

    clipper1 = vtk.vtkClipPolyData()
    clipper1.SetInputConnection(sphere.GetOutputPort())
    clipper1.SetClipFunction(plane1)
    clipper1.Update()

    clipper2 = vtk.vtkClipPolyData()
    clipper2.SetInputConnection(sphere.GetOutputPort())
    clipper2.SetClipFunction(plane2)
    clipper2.Update()

    # 定义裁剪结果的平移向量
    translation_vector1 = [1.5, 0, 0]
    translation_vector2 = [-1.5, 0, 0]

    # 创建变换对象和平移过滤器
    transform1 = vtk.vtkTransform()
    transform1.Translate(translation_vector1)

    transform2 = vtk.vtkTransform()
    transform2.Translate(translation_vector2)

    transform_filter1 = vtk.vtkTransformPolyDataFilter()
    transform_filter1.SetInputConnection(clipper1.GetOutputPort())
    transform_filter1.SetTransform(transform1)
    transform_filter1.Update()

    transform_filter2 = vtk.vtkTransformPolyDataFilter()
    transform_filter2.SetInputConnection(clipper2.GetOutputPort())
    transform_filter2.SetTransform(transform2)
    transform_filter2.Update()

    # 创建映射器和演员
    mapper1 = vtk.vtkPolyDataMapper()
    mapper1.SetInputConnection(transform_filter1.GetOutputPort())

    actor1 = vtk.vtkActor()
    actor1.SetMapper(mapper1)
    actor1.GetProperty().SetColor(1.0, 0.0, 0.0)

    mapper2 = vtk.vtkPolyDataMapper()
    mapper2.SetInputConnection(transform_filter2.GetOutputPort())

    actor2 = vtk.vtkActor()
    actor2.SetMapper(mapper2)
    actor2.GetProperty().SetColor(0.0, 0.0, 1.0)

    # 创建渲染器、渲染窗口和交互器
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # 将演员添加到渲染器中
    renderer.AddActor(actor1)
    renderer.AddActor(actor2)
    renderer.SetBackground(0.1, 0.2, 0.3)

    # 开始交互
    render_window.Render()
    interactor.Start()


if __name__ == "__main__":
    main()
    