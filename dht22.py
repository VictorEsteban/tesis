import adafruit_dht
import board
import time

#Se inicia el sensor de temperatura en el pin indicado en el argumento
# Esta configurado en BCM
dhtDevice = adafruit_dht.DHT22(board.D2)

def sensor():
        temperature_c = dhtDevice.temperature
        temperature_f = temperature_c*(9/5)+32
        print('Temp: {:.1f}°C / {:.1f}°F'.format(temperature_c, temperature_f))

while True:
    try:
        sensor()
    except RuntimeError as error:
        # Errors happen fairly often, DHT's are hard to read, just keep going
        print(error.args[0])
        time.sleep(2.0)
        continue
    except Exception as error:
        dhtDevice.exit()
        raise error
    time.sleep(6.0)