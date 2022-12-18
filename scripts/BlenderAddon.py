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
        col.prop(context.scene, "LFGridSize")
        col.prop(context.scene, "LFAnimation")
        col.operator("lf.render", text="Render")
        col.operator("lf.preview", text="Preview")
        
        if context.scene.camera == None:
            col.label(text="No active camera set!")
        if context.scene.LFRunning:
            col.label(text="Running...")
            row = self.layout.row()
            row.prop(context.scene, "LFProgress")
            row.enabled = False

class ProgressBar(bpy.types.Operator):
    """Prints the progress"""
    bl_idname = "wm.progressbar"
    bl_label = "Progress bar"
    timer = None
    timerPeriod = 0.65
    totalCount = 0
    totalCountInTime = 0
    frameCount = 1
    currentFrame = 0
    borderIDs = [0,0]
    originalCameraLocation = None
    
    def backupCamera(self):
        camera = bpy.context.scene.camera 
        self.originalCameraLocation = copy.deepcopy(camera.location)
    
    def restoreCamera(self):
        camera = bpy.context.scene.camera 
        camera.location = copy.deepcopy(self.originalCameraLocation)
        
    def initVars(self, context): 
        context.scene.LFCurrentView[0] = 0
        context.scene.LFCurrentView[1] = 0
        context.scene.LFCurrentView[2] = 0
        self.totalCount = context.scene.LFGridSize[0]*context.scene.LFGridSize[1]
        self.totalCountInTime = copy.deepcopy(self.totalCount)      
        self.currentFrame = bpy.context.scene.frame_current 
        if context.scene.LFAnimation:
            self.frameCount = bpy.context.scene.frame_end - bpy.context.scene.frame_start + 1
            self.totalCountInTime *= self.frameCount
            print(self.totalCountInTime)
            self.currentFrame = bpy.context.scene.frame_start
    
    def updateFrame(self, context):
        if (context.scene.LFCurrentView[2] % self.totalCount) == 0:
            currentFrame = context.scene.LFCurrentView[2] // self.totalCount
            currentFrame += bpy.context.scene.frame_start
            self.restoreCamera()
            context.scene.frame_set(currentFrame)
            self.backupCamera()
    
    def checkAndUpdateCurrentView(self, context):
        if context.scene.LFAnimation:
            self.updateFrame(context)
        context.scene.LFCurrentView[2] += 1    
        if context.scene.LFCurrentView[2] >= self.totalCountInTime:
            return False
        linear = context.scene.LFCurrentView[2] % self.totalCount
        context.scene.LFCurrentView[0] = linear % context.scene.LFGridSize[0] 
        context.scene.LFCurrentView[1] = linear // context.scene.LFGridSize[0]
        return True    
    
    def updateProgress(self, context):
        context.scene.LFProgress = (context.scene.LFCurrentView[2]/(self.totalCountInTime))*100

    def modal(self, context, event):
        if event.type in {'ESC'} or context.scene.LFShouldEnd:
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':            
            if context.scene.LFCurrentTaskFinished:
                self.restoreCamera()
                if context.scene.LFIsPreview:
                    bpy.ops.lf.preview()
                else:
                    bpy.ops.lf.render()
                notFinished = self.checkAndUpdateCurrentView(context)
                if not notFinished:
                    self.deferredCancel(context)
                    return {'PASS_THROUGH'}
                self.updateProgress(context)

        return {'PASS_THROUGH'}

    def execute(self, context):
        context.scene.LFProgress = 0.0
        context.scene.LFRunning = True
        context.scene.LFShouldEnd = False
        self.backupCamera()
        self.initVars(context)
        wm = context.window_manager
        self.timer = wm.event_timer_add(self.timerPeriod, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def deferredCancel(self, context):
        context.scene.LFShouldEnd = True
        context.scene.LFProgress = 100.0

    def cancel(self, context):
        self.restoreCamera()
        context.scene.LFRunning = False
        wm = context.window_manager
        wm.event_timer_remove(self.timer)

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
    
    def getCurrentCoords(self, context):
        x = context.scene.LFCurrentView[0]
        y = context.scene.LFCurrentView[1]
        return x,y
    
    def currentPosition(self, context):
        x, y = self.getCurrentCoords(context)
        step = mathutils.Vector(context.scene.LFStep)
        gridSize = mathutils.Vector(context.scene.LFGridSize)
        camera = self.getCameraVectors()
        return camera.offsetPosition(step, mathutils.Vector((x,y)), gridSize) 
    
    def createTrajectory(self, context):   
        camera = self.getCameraVectors()
        gridSize = mathutils.Vector(context.scene.LFGridSize)    
        trajectory = [[copy.deepcopy(camera.position) for x in range(int(gridSize.x))] for y in range(int(gridSize.y))]
        step = mathutils.Vector(context.scene.LFStep)
        for x in range(int(gridSize.x)):
            for y in range(int(gridSize.y)):
                trajectory[x][y] = camera.offsetPosition(step, mathutils.Vector((x,y)), gridSize)
        return trajectory

class LFRender(bpy.types.Operator, CameraTrajectory):
    """ Renders the LF structure """
    bl_idname = "lf.render"
    bl_label = "Render"
    
    originalCamera = None

    def duplicateCamera(self, context):
        for obj in bpy.context.selected_objects:
            obj.select_set(False)          
        camera = context.scene.camera
        camera.select_set(True)
        self.originalCamera = camera
        bpy.ops.object.duplicate(linked=False)
        self.originalCamera.select_set(False)
        newCamera = bpy.context.selected_objects[0]
        print(newCamera.name)
        if newCamera.animation_data.action != None:
            newCamera.animation_data.action = None
        context.scene.camera = newCamera
        return newCamera  
    
    def restoreCamera(self, context):
        bpy.ops.object.delete() 
        context.scene.camera = self.originalCamera

    def execute(self, context):
        renderInfo = bpy.context.scene.render
        originalPath = copy.deepcopy(renderInfo.filepath)
        camera = self.duplicateCamera(context)    
        print(camera.name)
        gridSize = mathutils.Vector(context.scene.LFGridSize)    
 
        x, y = self.getCurrentCoords(context)
        frameID = bpy.context.scene.frame_current
        filename = os.path.join(str(frameID), str(y)+"_"+str(x))
        renderInfo.filepath = os.path.join(originalPath, filename)
        camera.location = self.currentPosition(context)
        bpy.ops.render.render(write_still=True)  
        
        self.restoreCamera(context)
        renderInfo.filepath = originalPath 
        return {"FINISHED"}

    def invoke(self, context, event):
        context.scene.LFIsPreview = False
        bpy.ops.wm.progressbar()
        return {"FINISHED"}
    
class LFPreview(bpy.types.Operator, CameraTrajectory):
    """ Animated the camera motion in the grid """
    bl_idname = "lf.preview"
    bl_label = "Render" 

    def execute(self, context):
        context.scene.LFCurrentTaskFinished = False    
        camera = context.scene.camera
        camera.location = self.currentPosition(context)
        context.scene.LFCurrentTaskFinished = True
        return {"FINISHED"}

    def invoke(self, context, event):
        #print(self.createTrajectory(context))
        context.scene.LFIsPreview = True
        bpy.ops.wm.progressbar()        
        return {"FINISHED"}

def register():
    bpy.utils.register_class(ProgressBar)
    bpy.utils.register_class(LFPanel)
    bpy.utils.register_class(LFRender)
    bpy.utils.register_class(LFPreview)
    bpy.types.Scene.LFStep = bpy.props.FloatVectorProperty(name="Step", size=2, description="The distance between cameras", min=0, default=(1.0,1.0), subtype="XYZ")
    bpy.types.Scene.LFGridSize = bpy.props.IntVectorProperty(name="Grid size", size=2, description="The total number of the views", min=2, default=(8,8), subtype="XYZ")
    bpy.types.Scene.LFAnimation = bpy.props.BoolProperty(name="Render animation", description="Will render all active frames as animation", default=False)
    bpy.types.Scene.LFProgress = bpy.props.FloatProperty(name="Progress", description="Progress bar", subtype="PERCENTAGE",soft_min=0, soft_max=100, default=0.0)
    bpy.types.Scene.LFCurrentView = bpy.props.IntVectorProperty(name="Current view", size=3, description="The currenty processed view - XY and third is linear ID", default=(0,0,0))
    bpy.types.Scene.LFCurrentTaskFinished = bpy.props.BoolProperty(name="Current view finished", description="Indicates that the current view was processed", default=True)
    bpy.types.Scene.LFShouldEnd = bpy.props.BoolProperty(name="Deffered ending", description="Is set to true in the last view to show last progress", default=False)
    bpy.types.Scene.LFRunning = bpy.props.BoolProperty(name="Running", description="Indicates that the rendering or previewing is in progress", default=False)
    bpy.types.Scene.LFIsPreview = bpy.props.BoolProperty(name="Is preview", description="Switch between preview and render", default=False)
    
    
def unregister():
    bpy.utils.unregister_class(ProgressBar)
    bpy.utils.unregister_class(LFPanel)
    bpy.utils.unregister_class(LFRender)
    bpy.utils.unregister_class(LFPreview)
    
if __name__ == "__main__" :
    register()        
