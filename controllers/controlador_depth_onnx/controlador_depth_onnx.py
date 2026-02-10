from controller import Robot, Keyboard
import os
import sys
import ctypes
import math

# ==========================================
# 1. HACK GPU (NO TOCAR)
# ==========================================
ENV_PATH = "/home/jd/TFG/tfg_env"
libs_to_load = [
    "libcudart.so.12", "libcublasLt.so.12", "libcublas.so.12", "libcufft.so.11",
    "libcurand.so.10", "libcudnn.so.9", "libcudnn_cnn_infer.so.9", "libcudnn_ops_infer.so.9"
]

def force_load_nvidia_libs(env_root):
    nvidia_dir = os.path.join(env_root, "lib", "python3.13", "site-packages", "nvidia")
    found_libs = {}
    for root, dirs, files in os.walk(nvidia_dir):
        for file in files:
            if ".so" in file:
                found_libs[file] = os.path.join(root, file)
    for lib in libs_to_load:
        path = found_libs.get(lib)
        if not path:
            for k, v in found_libs.items():
                if lib in k:
                    path = v
                    break
        if path:
            try:
                ctypes.CDLL(path)
            except:
                pass

force_load_nvidia_libs(ENV_PATH)

# ==========================================
# 2. IMPORTS IA
# ==========================================
import cv2
import numpy as np
import onnxruntime as ort

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# ==========================================
# 3. INIT ROBOT + DEVICES
# ==========================================
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Cámara (para tu IA)
camera = robot.getDevice("camera")
camera.enable(timestep)

# LEDs (si existen; si no, no pasa nada)
front_left_led = None
front_right_led = None
try:
    front_left_led = robot.getDevice("front left led")
    front_right_led = robot.getDevice("front right led")
except:
    pass

# Sensores de vuelo
imu = robot.getDevice("inertial unit")
imu.enable(timestep)

gps = robot.getDevice("gps")
gps.enable(timestep)

gyro = robot.getDevice("gyro")
gyro.enable(timestep)

# Compass (no lo usamos en este control, pero lo habilito si existe)
try:
    compass = robot.getDevice("compass")
    compass.enable(timestep)
except:
    compass = None

# Teclado
keyboard = Keyboard()
keyboard.enable(timestep)

# Motores de cámara (si existen; si no, no pasa nada)
camera_roll_motor = None
camera_pitch_motor = None
try:
    camera_roll_motor = robot.getDevice("camera roll")
    camera_pitch_motor = robot.getDevice("camera pitch")
except:
    pass

# Motores de hélices
front_left_motor = robot.getDevice("front left propeller")
front_right_motor = robot.getDevice("front right propeller")
rear_left_motor = robot.getDevice("rear left propeller")
rear_right_motor = robot.getDevice("rear right propeller")

motors = [front_left_motor, front_right_motor, rear_left_motor, rear_right_motor]
for m in motors:
    m.setPosition(float("inf"))
    m.setVelocity(1.0)

print("Start the drone...")
# Espera 1s (como el ejemplo C)
while robot.step(timestep) != -1:
    if robot.getTime() > 1.0:
        break

print("You can control the drone with your computer keyboard:")
print("- 'up': move forward.")
print("- 'down': move backward.")
print("- 'right': turn right.")
print("- 'left': turn left.")
print("- 'shift + up': increase the target altitude.")
print("- 'shift + down': decrease the target altitude.")
print("- 'shift + right': strafe right.")
print("- 'shift + left': strafe left.")

# ==========================================
# 4. CONTROL ESTABLE (PORTADO DEL C)
# ==========================================
k_vertical_thrust = 68.5   # con este empuje, despega
k_vertical_offset = 0.6
k_vertical_p = 3.0
k_roll_p = 50.0
k_pitch_p = 30.0

target_altitude = 1.0

# ==========================================
# 5. CARGA IA (DEPTH ANYTHING V2 ONNX) - IGUAL QUE TENÍAS
# ==========================================
MODEL_FILE = "depth_anything_vits14_364.onnx"
script_dir = os.path.dirname(os.path.abspath(__file__))
ai_path = os.path.join(script_dir, MODEL_FILE)

INPUT_SIZE = 364
PASOS_A_SALTAR = 8
FACTOR_METRICO = 12.0

class DepthONNX:
    def __init__(self, model_path):
        try:
            self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
            print("-> IA EN GPU (CUDA) ACTIVADA")
        except:
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            print("-> IA EN CPU")
        self.input_name = self.session.get_inputs()[0].name
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def infer(self, img):
        img_resized = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        img_input = ((img_resized / 255.0 - self.mean) / self.std).transpose(2, 0, 1)[None, :, :, :].astype(np.float32)
        out = self.session.run(None, {self.input_name: img_input})[0][0]
        return cv2.resize(out, (img.shape[1], img.shape[0]))

ai = DepthONNX(ai_path)

contador_pasos = 0

# ==========================================
# 5.1 GRABACIÓN VÍDEO (solo cámara del dron)
# ==========================================
GRABAR_VIDEO = True              # Pon False si no quieres grabar
VIDEO_FPS = 20.0                 # Ajusta FPS a tu gusto
VIDEO_PATH = "drone_camera.mp4"  # Ruta/archivo de salida

video_out = None
if GRABAR_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_out = cv2.VideoWriter(
        VIDEO_PATH,
        fourcc,
        VIDEO_FPS,
        (camera.getWidth(), camera.getHeight())
    )
    if not video_out.isOpened():
        print("ERROR: no se pudo abrir el VideoWriter. Prueba a cambiar el codec o la ruta.")
        video_out = None
    else:
        print(f"Grabando vídeo de cámara en: {VIDEO_PATH}")

# ==========================================
# 6. LOOP PRINCIPAL (CONTROL + IA)
# ==========================================
while robot.step(timestep) != -1:
    time_s = robot.getTime()

    # Sensores
    roll, pitch, yaw = imu.getRollPitchYaw()
    altitude = gps.getValues()[2]
    roll_velocity, pitch_velocity, yaw_velocity = gyro.getValues()

    # LEDs (si existen)
    if front_left_led is not None and front_right_led is not None:
        led_state = int(time_s) % 2
        front_left_led.set(led_state)
        front_right_led.set(1 - led_state)

    # Estabilización de cámara (si existen)
    if camera_roll_motor is not None and camera_pitch_motor is not None:
        camera_roll_motor.setPosition(-0.115 * roll_velocity)
        camera_pitch_motor.setPosition(-0.1 * pitch_velocity)

    # Teclado → disturbances (igual que C)
    roll_disturbance = 0.0
    pitch_disturbance = 0.0
    yaw_disturbance = 0.0

    key = keyboard.getKey()
    while key > 0:
        if key == Keyboard.UP:
            pitch_disturbance = -3.0
        elif key == Keyboard.DOWN:
            pitch_disturbance = 3.0
        elif key == Keyboard.RIGHT:
            yaw_disturbance = -2.0
        elif key == Keyboard.LEFT:
            yaw_disturbance = 2.0
        elif key == (Keyboard.SHIFT + Keyboard.RIGHT):
            roll_disturbance = -1.5
        elif key == (Keyboard.SHIFT + Keyboard.LEFT):
            roll_disturbance = 1.5
        elif key == (Keyboard.SHIFT + Keyboard.UP):
            target_altitude += 0.05
            print(f"target altitude: {target_altitude:.2f} [m]")
        elif key == (Keyboard.SHIFT + Keyboard.DOWN):
            target_altitude -= 0.05
            print(f"target altitude: {target_altitude:.2f} [m]")
        key = keyboard.getKey()

    # Control (igual que C)
    roll_input  = k_roll_p  * clamp(roll,  -1.0, 1.0) + roll_velocity  + roll_disturbance
    pitch_input = k_pitch_p * clamp(pitch, -1.0, 1.0) + pitch_velocity + pitch_disturbance
    yaw_input = yaw_disturbance

    clamped_diff_alt = clamp(target_altitude - altitude + k_vertical_offset, -1.0, 1.0)
    vertical_input = k_vertical_p * (clamped_diff_alt ** 3)

    # Mix (igual que C)
    front_left_motor_input  = k_vertical_thrust + vertical_input - roll_input + pitch_input - yaw_input
    front_right_motor_input = k_vertical_thrust + vertical_input + roll_input + pitch_input + yaw_input
    rear_left_motor_input   = k_vertical_thrust + vertical_input - roll_input - pitch_input + yaw_input
    rear_right_motor_input  = k_vertical_thrust + vertical_input + roll_input - pitch_input - yaw_input

    # IMPORTANTÍSIMO: signos como en el C
    front_left_motor.setVelocity(front_left_motor_input)
    front_right_motor.setVelocity(-front_right_motor_input)
    rear_left_motor.setVelocity(-rear_left_motor_input)
    rear_right_motor.setVelocity(rear_right_motor_input)

    # =============================
    # IA Depth (tu bloque, igual)
    # + grabación de vídeo
    # =============================
    if contador_pasos % PASOS_A_SALTAR == 0:
        img_raw = camera.getImage()
        if img_raw:
            img_np = np.frombuffer(img_raw, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
            img_rgb = img_np[:, :, :3]

            # Guardar frame RGB (misma entrada que la red)
            if video_out is not None:
                video_out.write(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

            depth_raw = ai.infer(img_rgb)

            depth_meters = FACTOR_METRICO / (depth_raw + 1e-6)
            d_norm = (depth_raw - depth_raw.min()) / (depth_raw.max() - depth_raw.min() + 1e-9)
            depth_color = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

            cv2.putText(depth_color, f"Alt: {altitude:.2f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(depth_color, f"Dist: {depth_meters[depth_meters.shape[0]//2, depth_meters.shape[1]//2]:.2f}m",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("IA Depth", depth_color)
            cv2.imshow("Dron", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    contador_pasos += 1

# Cierre limpio
if video_out is not None:
    video_out.release()
    print(f"Vídeo guardado: {VIDEO_PATH}")

cv2.destroyAllWindows()
