import vtkmodules.all as vtk

class NiiObject:
    def __init__(self):
        self.file = None
        self.reader = None
        self.extent = ()
        self.labels = []
        self.image_mapper = None
        self.scalar_range = None


class NiiLabel:
    def __init__(self, color, opacity, smoothness):
        self.actor = None
        self.property = None
        self.smoother = None
        self.color = color
        self.opacity = opacity
        self.smoothness = smoothness

def create_mask_extractor(mask):
    """
    Given the output from mask (vtkNIFTIImageReader) extract it into 3D using
    vtkDiscreteMarchingCubes algorithm (https://www.vtk.org/doc/release/5.0/html/a01331.html).
    This algorithm is specialized for reading segmented volume labels.
    :param mask: a vtkNIFTIImageReader volume containing the mask
    :return: the extracted volume from vtkDiscreteMarchingCubes
    """
    mask_extractor = vtk.vtkDiscreteMarchingCubes()
    mask_extractor.SetInputConnection(mask.reader.GetOutputPort())
    return mask_extractor


def create_polygon_reducer(extractor):
    """
    Reduces the number of polygons (triangles) in the volume. This is used to speed up rendering.
    (https://www.vtk.org/doc/nightly/html/classvtkDecimatePro.html)
    :param extractor: an extractor (vtkPolyDataAlgorithm), will be either vtkFlyingEdges3D or vtkDiscreteMarchingCubes
    :return: the decimated volume
    """
    reducer = vtk.vtkDecimatePro()
    reducer.SetInputConnection(extractor.GetOutputPort())
    reducer.SetTargetReduction(0.5)  # magic number
    reducer.PreserveTopologyOn()
    return reducer

def create_smoother(reducer, smoothness):
    """
    Reorients some points in the volume to smooth the render edges.
    (https://www.vtk.org/doc/nightly/html/classvtkSmoothPolyDataFilter.html)
    :param reducer:
    :param smoothness:
    :return:
    """
    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputConnection(reducer.GetOutputPort())
    smoother.SetNumberOfIterations(smoothness)
    return smoother


def create_normals(smoother):
    """
    The filter can reorder polygons to insure consistent orientation across polygon neighbors. Sharp edges can be split
    and points duplicated with separate normals to give crisp (rendered) surface definition.
    (https://www.vtk.org/doc/nightly/html/classvtkPolyDataNormals.html)
    :param smoother:
    :return:
    """
    brain_normals = vtk.vtkPolyDataNormals()
    brain_normals.SetInputConnection(smoother.GetOutputPort())
    brain_normals.SetFeatureAngle(60.0)  #
    return brain_normals

def create_mapper(stripper):
    brain_mapper = vtk.vtkPolyDataMapper()
    brain_mapper.SetInputConnection(stripper.GetOutputPort())
    brain_mapper.ScalarVisibilityOff()
    brain_mapper.Update()
    return brain_mapper


def create_property(opacity, color):
    prop = vtk.vtkProperty()
    prop.SetColor(color[0], color[1], color[2])
    prop.SetOpacity(opacity)
    return prop

def create_actor(mapper, prop):
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.SetProperty(prop)
    return actor

def add_surface_rendering(nii_object, label_idx, label_value):
    nii_object.labels[label_idx].extractor.SetValue(0, label_value)
    nii_object.labels[label_idx].extractor.Update()

    # if the cell size is 0 then there is no label_idx data
    if nii_object.labels[label_idx].extractor.GetOutput().GetMaxCellSize():
        reducer = create_polygon_reducer(nii_object.labels[label_idx].extractor)
        smoother = create_smoother(reducer, nii_object.labels[label_idx].smoothness)
        normals = create_normals(smoother)
        actor_mapper = create_mapper(normals)
        actor_property = create_property(nii_object.labels[label_idx].opacity, nii_object.labels[label_idx].color)
        actor = create_actor(actor_mapper, actor_property)
        nii_object.labels[label_idx].actor = actor
        nii_object.labels[label_idx].smoother = smoother
        nii_object.labels[label_idx].property = actor_property