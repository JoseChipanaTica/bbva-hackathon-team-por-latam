# Autentícate con Autenticar(nos)

Proyecto creado por el equipo "Por Latam".
Para el Hackathon BBVA 2022

## Procesamiento para desplegar el proyecto
* Es necesario crear un Máquina Virtual EC2 en AWS
* Recomendamos crear un entorno virtual: 'python3 -m venv bbva-env'
* Activar el entorno: source: 'bbva-env/bin/activate'
* Instalar las librerías requeridas: 'pip install -r requirements.txt'
* Ingresar al proyecto:  'cd bbva-hackathon'
* Iniciar el servicio: 'python3 -m uvicorn main:app'


### Archivo Main.py
* En este archivo se encuentra el microservicio basado en FastApi

### Archivo model_classification
* Es el modelo de clasificación de clientes. 
* El modelo es un SVM con 4097 variables de entrada


### Archivo bbva-hackathon.ipynb
* Es el notebook donde se procesó las imágenes y se implementó el modelo.