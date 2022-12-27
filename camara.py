# Al parecer en versiones más recientes no se utiliza la funcion
#Create de la siguiente forma
# cam_rgb = pipeline.create(depthai.node.ColorCamera)
# ahora se utiliza esta funcion reduciendo el codigo a
#cam_rgb = pipeline.createColorCamera()
# de esta forma actualmente se crean los nodos
# Este trabajo esta dedicado a mis amores, Daniela Quinteros y Dominga

import cv2 # Opencv - muestra la transmisión de video
import numpy as np # Numpy: manipula los datos del paquete devueltos por depthai
import depthai # depthai - acceda a la cámara y sus paquetes de datos
#import adafruit_dht # importa la libreria que controla el sensor dht
#import board
#import RPi.GPIO as gpio
import blobconverter #blobconverter: compila y descarga blobs de la red neuronal MyriadX
from time import sleep

# Se crea un Pipeline vacio
pipeline = depthai.Pipeline()

# Se crean 2 nodos dentro de Pipeline

# Nodo de la camara a color
cam_rgb = pipeline.create(depthai.node.ColorCamera)
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setInterleaved(False)

# Nodo de la red neuronal
detection_nn = pipeline.create(depthai.node.MobileNetDetectionNetwork)
detection_nn.setBlobPath(blobconverter.from_zoo(name = 'mobilenet-ssd', shaves = 6)) #El comando blobconverter.from.zoo() devuelve el Path el argumento debe ser el nombre de la neuronal network, y shaves (no se aun que es esto)
detection_nn.setConfidenceThreshold(0.5)

# Con esto se conecta la vista previa de la camara con la red neuronal
# se enlazan ambas a traves del comando link
# La siguiente linea realiza una conexion entre la salida de la camara a color con la entrada de la red neuronal
cam_rgb.preview.link(detection_nn.input)

# XLinkOut conecta la salida del nodo con la entrada del Host, y transfiere datos desde el nodo al Host

# XLinkOut conecta la salida de la camara a color con la entrada del host, tambien esta encargado de transferir la informacion desde la camara al host
xout_rgb = pipeline.create(depthai.node.XLinkOut)
xout_rgb.setStreamName('rgb')
cam_rgb.preview.link(xout_rgb.input)

#XLinkOut conecta la salida de la red neuronal a la entrada del host, tambien se encarga del transporte de informacion desde la salida de la red neuronal al host
xout_nn = pipeline.create(depthai.node.XLinkOut)
xout_nn.setStreamName('nn')
detection_nn.out.link(xout_nn.input)

with depthai.Device(pipeline) as device:

    q_rgb = device.getOutputQueue('rgb')
    q_nn = device.getOutputQueue('nn')

    frame = None
    detections = []
    persona = 0

    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1] #[::2] en python se le llama rebanadas. [::2] se salta los indices pares
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

# Se consumen los resultados tanto de la cámara a color, como también de la red neuronal
    def camara():
        global frame # devuelve un arreglo de 3x3 el cual indica la posicion y color del recuadro rastreador, debe ser global para que pueda ser utilizada por otros hilos
        global detections # devuelve el codigo del objeto detectado, debe ser global para que pueda ser utilizado por otros hilos
        global persona # devuelve el resultado de la variable detection.label
        in_rgb = q_rgb.tryGet()
        in_nn = q_nn.tryGet()
        if in_rgb is not None:
            frame = in_rgb.getCvFrame()
        if in_nn is not None:
            detections = in_nn.detections
        if frame is not None:
            for detection in detections:
                if detection.label==15:
                    persona = 15 # Se crea la variable persona y se configura como global, ya que 'detection.label' no se puede sacar fuera de este hilo como variable global.
                    bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                elif detection.label != 15:
                    persona = 0 #Si es 0 significa que no detecta a una persona, si es 15 es porque esta detectando a una persona.
            cv2.imshow("preview", frame)

# Se inicia el programa principal y dentro se encuentran los hilos            
    while True:
        camara()
        if cv2.waitKey(1) == ord('q'):
            break
            