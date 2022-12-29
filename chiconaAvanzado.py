# Al parecer en versiones más recientes no se utiliza la funcion
#Create de la siguiente forma
# cam_rgb = pipeline.create(depthai.node.ColorCamera)
# ahora se utiliza esta funcion reduciendo el codigo a
#cam_rgb = pipeline.createColorCamera()
# de esta forma actualmente se crean los nodos
# Este trabajo esta dedicado a mis amores, Daniela Quinteros y Dominga

import cv2 # Opencv - muestra la transmisión de video.
import numpy as np # Numpy: manipula los datos del paquete devueltos por depthai.
import depthai # depthai - acceda a la cámara y sus paquetes de datos.
import adafruit_dht # importa la libreria que controla el sensor dht.
import board # libreria que define los pines correspondientes a los sensores DHT, en BCM.
import RPi.GPIO as gpio #Libreria para controlar los pines de Raspberry pi.
import blobconverter #blobconverter: compila y descarga blobs de la red neuronal MyriadX. Devuelve el PATH del modelo de red neuronal utilizado.
from threading import Thread #Libreria para utilzar Thread (hilos) para relizar programación concurrente.
from time import sleep

# Se crea un Pipeline vacio
pipeline = depthai.Pipeline() #Un pipeline es un entorno el cual se pueden crear nodos dentro de él.

# Se crean 2 nodos dentro de Pipeline
# Cada nodo tiene sus caracteristicas y se pueden encontrar en la documentación de Luxonis. Estos nodos se representan por bloques.
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

# Se utiliza la configuracion BCM, puesto que, la libreria 'board' que define los pines para utilizar el sensor DHT, configura la tarjeta con los pines en BCM, para no crear un conflicto se utiliza la misma configuracion para la libreria RPi.GPIO.
gpio.setmode(gpio.BCM)
# Se definen los pines 3, 4 y 17 como salidas
gpio.setup(3, gpio.OUT)
gpio.setup(4, gpio.OUT)
gpio.setup(17, gpio.OUT)
# Los pines 3, 4 y 17 se inician apagados
gpio.output(3, False)
gpio.output(4, False)
gpio.output(17, False)

#Se inicia el dispositivo (Camara OAK), junto con la configuración de nodos dentro de ella.
with depthai.Device(pipeline) as device:

#Se establecen las Queue, donde el primer dato enviado por el dispositivo, será también el primero en ser leido.
    q_rgb = device.getOutputQueue('rgb')
    q_nn = device.getOutputQueue('nn')

# se definen las siguientes variables.
    frame = None
    detections = []
    aux1 = None
    aux2 = None
# se define una funcion la cual permite a partir de una matriz realizar el recuadro y seguimiento de los objetos.
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1] #[::2] en python se le llama rebanadas. [::2] se salta los indices pares
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

# Se define una funcion 'camara' la cual consume los resultados tanto de la cámara a color, como también de la red neuronal.
    def camara():
        while True:
            global frame # devuelve un arreglo de 3x3 el cual indica la posicion y color del recuadro rastreador, debe ser global para que pueda ser utilizada por otros hilos
            global detections # devuelve el codigo del objeto detectado, debe ser global para que pueda ser utilizado por otros hilos
            global aux1 # Variable auxiliar que permite guardar 
            in_rgb = q_rgb.tryGet() 
            in_nn = q_nn.tryGet()
            if in_rgb is not None:
                frame = in_rgb.getCvFrame()
            if in_nn is not None:
                detections = in_nn.detections
            if frame is not None:
                for detection in detections:
                    if detection.label==15: #detection.label indica el indice de los objetos detectados.
                        gpio.output(4, True)
                        aux1 = True
                        bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                    elif detection.label != 15:
                        gpio.output(4, False)
                        aux1 = False
                cv2.imshow("preview", frame) #muestra una ventana con la camara a color y la red neuronal funcionando.
            if cv2.waitKey(1) == ord('q'): #espera una tecla en este caso la 'q' para pasar a la siguiente linea y finalizar el programa.
                gpio.cleanup()
                break   
# se crea una funcion 'sensor' la cual es capaz se recibir la informacion entregada por el sensor DHT22, devuelve la temperatura en grados celsius, farenheit y humedad.
    def sensor():
        global aux2
        global temperature_c
        global temperature_f
        dhtDevice = adafruit_dht.DHT22(board.D2, use_pulseio=False)
        temperature_c = dhtDevice.temperature
        temperature_f = temperature_c*(9/5)+32
        print('Temp: {:.1f}°C / {:.1f}°F'.format(temperature_c, temperature_f)) # muestra el texto en consola
        if temperature_c >= 20.0:
            gpio.output(3, True)
            aux2 = True
        else:
            gpio.output(3, False)
            aux2 = False
        if aux1 == True & aux2 == True:
            gpio.output(17, True)
        else:
            gpio.output(17, False)
        sleep(60)
        while True:
            try:
                sensor()
            except RuntimeError as error:
                print(error.args[0])
                sleep(2)
                continue
            except Exception as error:
                dhtDevice.exit()
# se crean los hilos para cada funcion y así establecer que el programa funcione de forma concurrente.            
    hilocamara = Thread(target=camara)
    hilosensor = Thread(target=sensor, daemon=True) #Daemon True indica que es un hilo hijo
    hilocamara.start() #inicia el hilo
    hilosensor.start() # inicia el hilo
    hilocamara.join() #finaliza el hilo padre



       
             