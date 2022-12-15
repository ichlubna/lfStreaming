bl_info = {
    "name": "Light Field Renderer",
    "author": "ichlubna",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "location": "3D View side panel",
    "description": "Renders and stores views in a LF grid with the active camera in the center. Project render settings are used and all frames in the active range are rendered in case of animation.",
    "warning": "",
    "doc_url": "",
    "category": "Import-Export"
}

import bpy
import copy
import os
import time
import mathutils

class LFPanel(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "LF"
    bl_label = "Light Field Renderer"

    def draw(self, context):
        col = self.layout.column(align=True)
        col.prop(context.scene, "LFStep")
        col.prop(context.scene, "LFDensity")
        col.prop(context.scene, "LFAnimation")
        col.operator("lf.render", text="Render")
        col.operator("lf.preview", text="Preview")

class CameraTrajectory():
    class CameraVectors():
        down = mathutils.Vector((0.0, 0.0, 0.0))
        direction = mathutils.Vector((0.0, 0.0, 0.0))
        right = mathutils.Vector((0.0, 0.0, 0.0))
        position = mathutils.Vector((0.0, 0.0, 0.0))
              
        def offsetPosition(self, step, coords, size):   
            centering = copy.deepcopy(coords)
            centering.x /= size.x-1
            centering.y /= size.y-1
            centering.x -= 0.5
            centering.y -= 0.5
            centering *= 2
            newPosition = copy.deepcopy(self.position)
            newPosition += self.down*step.y*centering.y
            newPosition += self.right*step.x*centering.x
            return newPosition
        
    def getCameraVectors(self):
        vectors = self.CameraVectors()
        camera = bpy.context.scene.camera 
        vectors.position = mathutils.Vector(camera.location)
        vectors.down = camera.matrix_world.to_quaternion() @ mathutils.Vector((0.0, -1.0, 0.0))
        vectors.down = vectors.down.normalized()
        vectors.direction = camera.matrix_world.to_quaternion() @ mathutils.Vector((0.0, 0.0, -1.0))
        vectors.direction = vectors.direction.normalized()
        vectors.right = vectors.down.cross(vectors.direction)
        return vectors
        
    
    def createTrajectory(self, context):
        camera = self.getCameraVectors()   
        print(camera.position)     
        gridSize = mathutils.Vector(context.scene.LFDensity)    
        trajectory = [[copy.deepcopy(camera.position) for x in range(int(gridSize.x))] for y in range(int(gridSize.y))]
        step = mathutils.Vector(context.scene.LFStep)
        for x in range(int(gridSize.x)):
            for y in range(int(gridSize.y)):
                trajectory[x][y] = camera.offsetPosition(step, mathutils.Vector((x,y)), gridSize)
        print(trajectory)
        return trajectory

class LFRender(bpy.types.Operator, CameraTrajectory):
    """ Renders the LF structure """
    bl_idname = "lf.render"
    bl_label = "Render"

    def render(self, context):
        renderInfo = bpy.context.scene.render
        originalPath = copy.deepcopy(renderInfo.filepath)
        trajectory = self.createTrajectory(context)
        camera = bpy.context.scene.camera 
        originalCameraLocation = copy.deepcopy(camera.location)        
        gridSize = mathutils.Vector(context.scene.LFDensity)
        
        frameRange = range(bpy.context.scene.frame_current, bpy.context.scene.frame_current+1)
        if context.scene.LFAnimation:
            frameRange = range(bpy.context.scene.frame_start, bpy.context.scene.frame_end+1)
            
        for frameID in frameRange:
            bpy.context.scene.frame_set(frameID)
            #TODO dangerous
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
            for x in range(int(gridSize.x)):
                for y in range(int(gridSize.y)):    
                    filename = os.path.join(str(frameID), str(y)+"_"+str(x))
                    renderInfo.filepath = os.path.join(originalPath, filename)
                    camera.location = trajectory[x][y]
                    bpy.ops.render.render( write_still=True )  
        
        renderInfo.filepath = originalPath 
        camera.location = originalCameraLocation

    def invoke(self, context, event):
        self.render(context)
        return {"FINISHED"}
    
class LFPreview(bpy.types.Operator, CameraTrajectory):
    """ Animated the camera motion in the grid """
    bl_idname = "lf.preview"
    bl_label = "Render"


    def animateCamera(self, context):
        trajectory = self.createTrajectory(context)
        camera = bpy.context.scene.camera 
        originalCameraLocation = copy.deepcopy(camera.location)
        gridSize = mathutils.Vector(context.scene.LFDensity)
        
        for x in range(int(gridSize.x)):
            for y in range(int(gridSize.y)):    
                camera.location = trajectory[x][y]
                #TODO fix with modal operator?
                bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
                time.sleep(0.2)
        camera.location = originalCameraLocation
        return

    def invoke(self, context, event):
        self.animateCamera(context)
        return {"FINISHED"}

def register():
    bpy.utils.register_class(LFPanel)
    bpy.utils.register_class(LFRender)
    bpy.utils.register_class(LFPreview)
    bpy.types.Scene.LFStep = bpy.props.FloatVectorProperty(name="Step", size=2, description="The distance between cameras", min=0, default=(1.0,1.0), subtype="XYZ")
    bpy.types.Scene.LFDensity = bpy.props.IntVectorProperty(name="Density", size=2, description="The total number of the views", min=1, default=(8,8), subtype="XYZ")
    bpy.types.Scene.LFAnimation = bpy.props.BoolProperty(name="Render animation", description="Will render all active frames as animation", default=False)
    
def unregister():
    bpy.utils.unregister_class(LFPanel)
    bpy.utils.unregister_class(LFRender)
    bpy.utils.unregister_class(LFPreview)
    
if __name__ == "__main__" :
    register()        
