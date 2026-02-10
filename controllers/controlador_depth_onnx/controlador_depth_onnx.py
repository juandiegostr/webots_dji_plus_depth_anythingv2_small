from controller import Robot, Keyboard
import os
import sys
import ctypes
import math

# ==========================================
# 1. HACK GPU (NO TOCAR)
# ==========================================
ENV_PATH = "/home/jd/TFG/tfg_env"
libs_to_load = ["libcudart.so.12", "libcublasLt.so.12", "libcublas.so.12", "libcufft.so.11", 
                "libcurand.so.10", "libcudnn.so.9", "libcudnn_cnn_infer.so.9", "libcudnn_ops_infer.so.9"]

def force_load_nvidia_libs(env_root):
    nvidia_dir = os.path.join(env_root, "lib", "python3.13", "site-packages", "nvidia")
    found_libs = {}
    for root, dirs, files in os.walk(nvidia_dir):
        for file in files:
            if ".so" in file: found_libs[file] = os.path.join(root, file)
    for lib in libs_to_load:
        path = found_libs.get(lib)
        if not path:
            for k, v in found_libs.items():
                if lib in k: path = v; break
        if path:
            try: ctypes.CDLL(path)
            except: pass
force_load_nvidia_libs(ENV_PATH)

# ==========================================
# 2. IMPORTS IA
# ==========================================
import cv2
import numpy as np
import onnxruntime as ort

# ==========================================
# 3. CLASE PID (EL CEREBRO DE VUELO)
# ==========================================
class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0
    
    def compute(self, target, current, dt):
        error = target - current
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

# ==========================================
# 4. CONFIGURACIÓN DRON
# ==========================================
robot = Robot()
timestep = int(robot.getBasicTimeStep())
dt = timestep / 1000.0 # Tiempo en segundos

# Sensores de Vuelo (NECESARIOS PARA EL PID)
imu = robot.getDevice("inertial unit")
imu.enable(timestep)
gps = robot.getDevice("gps")
gps.enable(timestep)
gyro = robot.getDevice("gyro")
gyro.enable(timestep)

# Teclado
keyboard = Keyboard()
keyboard.enable(timestep)

# Cámara
camera = robot.getDevice("camera")
camera.enable(timestep)

# Motores
motors = []
motor_names = ["front left propeller", "front right propeller", "rear right propeller", "rear left propeller"]
for name in motor_names:
    motor = robot.getDevice(name)
    motor.setPosition(float('inf'))
    motor.setVelocity(0.0)
    motors.append(motor)

# Cargar IA
MODEL_FILE = "depth_anything_vits14_364.onnx"
script_dir = os.path.dirname(os.path.abspath(__file__))
ai_path = os.path.join(script_dir, MODEL_FILE)
INPUT_SIZE = 364
PASOS_A_SALTAR = 12
FACTOR_METRICO = 12.0

class DepthONNX:
    def __init__(self, model_path):
        try:
            self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
            print("-> IA EN GPU (CUDA) ACTIVADA 🚀")
        except:
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def infer(self, img):
        img_resized = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        img_input = ((img_resized / 255.0 - self.mean) / self.std).transpose(2,0,1)[None,:,:,:].astype(np.float32)
        return cv2.resize(self.session.run(None, {self.input_name: img_input})[0][0], (img.shape[1], img.shape[0]))

ai = DepthONNX(ai_path)

# ==========================================
# 5. CONSTANTES PID (AFINADAS PARA MAVIC 2)
# ==========================================
# PID Vertical (Altura)
pid_v = PID(kp=5.0, ki=0.01, kd=3.0) 
target_altitude = 1.0 # Altura deseada en metros

# PID Roll/Pitch (Estabilidad Horizontal)
pid_roll = PID(kp=2.0, ki=0.0, kd=1.0)
pid_pitch = PID(kp=2.0, ki=0.0, kd=1.0)
target_roll = 0.0
target_pitch = 0.0

# PID Yaw (Giro)
pid_yaw = PID(kp=2.0, ki=0.00, kd=1.0)
target_yaw = 0.0 # Empezamos mirando al frente

contador_pasos = 0

print("\n--- PID ACTIVADO ---")
print(" [W/S]: Mover adelante/atrás")
print(" [A/D]: Mover lados")
print(" [ESPACIO/SHIFT]: Subir/Bajar Altura Objetivo")
print(" [FLECHAS]: Girar")

# ==========================================
# 6. BUCLE PRINCIPAL
# ==========================================
while robot.step(timestep) != -1:
    
    # --- A. LEER SENSORES ---
    roll, pitch, yaw = imu.getRollPitchYaw() # Radianes
    x, y, z = gps.getValues() # Metros (Z es altura en Webots)
    roll_rate, pitch_rate, yaw_rate = gyro.getValues()

    # --- B. LEER TECLADO (Actualiza los objetivos, no los motores) ---
    key = keyboard.getKey()
    
    # Altura objetivo
    if key == ord(' '): target_altitude += 0.05
    elif key == 317 or key == 65505: target_altitude -= 0.05

    # Ángulo objetivo (Roll/Pitch)
    disturb = 0.3 # Cuánto inclinamos el dron (radianes)
    if key == ord('W'): target_pitch = disturb   # Morro abajo
    elif key == ord('S'): target_pitch = -disturb # Morro arriba
    else: target_pitch = 0.0 # Si sueltas, vuelve a horizontal

    if key == ord('D'): target_roll = -disturb 
    elif key == ord('A'): target_roll = disturb
    else: target_roll = 0.0

    # Yaw Objetivo
    if key == Keyboard.LEFT: target_yaw += 0.03
    elif key == Keyboard.RIGHT: target_yaw -= 0.03

    # --- C. CÁLCULO PID ---
    # 1. Altura: Qué potencia base necesito para estar a 'target_altitude'?
    #    Clamp para no apagar motores (40) ni quemarlos (90)
    altitude_input = pid_v.compute(target_altitude, z, dt)
    vertical_thrust = 75 + altitude_input # 68.5 es el valor base para flotar (FeedForward)

    # 2. Estabilidad: Qué corrección necesito para estar en 'target_angle'?
    roll_input = pid_roll.compute(target_roll, roll, dt)
    pitch_input = pid_pitch.compute(target_pitch, pitch, dt)
    yaw_input = pid_yaw.compute(target_yaw, yaw, dt)

    # --- D. MIX DE MOTORES (Mavic 2 X-Config) ---
    # Motores: FrontLeft(+), FrontRight(-), RearRight(+), RearLeft(-)
    m1 = vertical_thrust + roll_input + pitch_input - yaw_input
    m2 = vertical_thrust - roll_input + pitch_input + yaw_input
    m3 = vertical_thrust - roll_input - pitch_input - yaw_input
    m4 = vertical_thrust + roll_input - pitch_input + yaw_input

    # Aplicar al robot
    motors[0].setVelocity(m1)
    motors[1].setVelocity(m2)
    motors[2].setVelocity(m3)
    motors[3].setVelocity(m4)

    # --- E. INTELIGENCIA ARTIFICIAL ---
    if contador_pasos % PASOS_A_SALTAR == 0:
        img_raw = camera.getImage()
        if img_raw:
            img_np = np.frombuffer(img_raw, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
            img_rgb = img_np[:, :, :3]
            depth_raw = ai.infer(img_rgb)
            
            # Visualización
            depth_meters = FACTOR_METRICO / (depth_raw + 1e-6)
            d_norm = (depth_raw - depth_raw.min()) / (depth_raw.max() - depth_raw.min())
            depth_color = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            cv2.putText(depth_color, f"Alt: {z:.2f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(depth_color, f"Dist: {depth_meters[182,182]:.2f}m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow("IA Depth", depth_color)
            cv2.imshow("Dron", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    contador_pasos += 1

cv2.destroyAllWindows()
