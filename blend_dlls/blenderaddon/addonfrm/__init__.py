import bpy
import tempfile
import os
import requests
import threading
import time

bl_info = {
    "name": "RIFE Viewport Streamer",
    "author": "Your Name",
    "version": (1, 4),  # Version 1.4 - Thread-safe screenshot
    "blender": (4, 5, 2),
    "location": "3D View > Sidebar > RIFE",
    "description": "Streams viewport screenshots to RIFE server.",
    "category": "Render",
}

bpy.types.Scene.rife_server_url = bpy.props.StringProperty(
    name="RIFE Server URL",
    default="http://127.0.0.1:8000/stream"
)

streaming_active = False
stream_thread = None

def capture_viewport(filepath: str):
    """Capture the active 3D viewport and save as image"""
    try:
        # Get the current context
        for window in bpy.context.window_manager.windows:
            screen = window.screen
            for area in screen.areas:
                if area.type == 'VIEW_3D':
                    # Use render.opengl to capture viewport
                    override = bpy.context.copy()
                    override['window'] = window
                    override['screen'] = screen
                    override['area'] = area
                    
                    # Set render settings for viewport capture
                    scene = bpy.context.scene
                    render = scene.render
                    
                    # Store original settings
                    original_filepath = render.filepath
                    original_format = render.image_settings.file_format
                    
                    # Set temporary settings
                    render.filepath = filepath
                    render.image_settings.file_format = 'PNG'
                    
                    # Capture viewport
                    with bpy.context.temp_override(**override):
                        bpy.ops.render.opengl(write_still=True)
                    
                    # Restore settings
                    render.filepath = original_filepath
                    render.image_settings.file_format = original_format
                    
                    return True
                    
    except Exception as e:
        print(f"Viewport capture error: {e}")
        return False
    
    return False

def stream_viewport(server_url):
    global streaming_active
    temp_dir = r"F:\New folder"  # Changed to F drive
    os.makedirs(temp_dir, exist_ok=True)

    while streaming_active:
        try:
            temp_file = os.path.join(temp_dir, "rife_viewport.png")
            print(f"Streaming active: {streaming_active}")
            print(f"Temp file path: {temp_file}")
            print(f"Attempting to send to: {server_url}")

            if capture_viewport(temp_file) and os.path.exists(temp_file):
                with open(temp_file, "rb") as f:
                    response = requests.post(server_url, data=f.read(), headers={"Content-Type": "image/png"})
                print(f"Sent frame {temp_file} - Response: {response.status_code}")
            else:
                print("Viewport capture failed")

            time.sleep(0.2)  # ~5 fps
        except Exception as e:
            print(f"Stream error: {e}")
            streaming_active = False
            break

class RIFE_OT_ToggleStream(bpy.types.Operator):
    bl_idname = "rife.toggle_stream"
    bl_label = "Start/Stop RIFE Stream"

    def execute(self, context):
        global streaming_active
        if streaming_active:
            streaming_active = False
            self.report({'INFO'}, "Stopped streaming")
        else:
            streaming_active = True
            # Start timer instead of thread
            bpy.app.timers.register(lambda: stream_frame(), first_interval=0.1)
            self.report({'INFO'}, "Started streaming")
        return {'FINISHED'}

class RIFE_PT_Panel(bpy.types.Panel):
    bl_label = "RIFE Live Stream"
    bl_idname = "RIFE_PT_Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'RIFE'

    def draw(self, context):
        layout = self.layout
        layout.prop(context.scene, "rife_server_url", text="Server URL")
        layout.operator("rife.toggle_stream", icon="PLAY")

        layout.separator()
        # Show version number
        layout.label(text=f"Addon version: {bl_info['version']}")

def register():
    bpy.utils.register_class(RIFE_OT_ToggleStream)
    bpy.utils.register_class(RIFE_PT_Panel)

def unregister():
    bpy.utils.unregister_class(RIFE_OT_ToggleStream)
    bpy.utils.unregister_class(RIFE_PT_Panel)

if __name__ == "__main__":
    register()