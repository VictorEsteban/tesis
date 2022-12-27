import RPi.GPIO as gpio

def accesorios():
    #Se configura los pines de la placa en modo BCM
    gpio.setmode(gpio.BCM)
    #Se establecen los pines 3, 4 y 17 como salidas
    gpio.setup(3, gpio.OUT)
    gpio.setup(4, gpio.OUT)
    gpio.setup(17, gpio.OUT)
    #Se establecen en valor True
    gpio.output(3, True)
    gpio.output(4, True)
    gpio.output(17, True)

while True:
    accesorios()

