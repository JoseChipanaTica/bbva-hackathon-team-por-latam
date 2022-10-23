# Autentícate con Autenticar(nos)

Proyecto creado por el equipo "Por Latam".
Para el Hackathon BBVA 2022

## Proyecto WEB
* El proyecto web fue desarrollado el React y se subió a Amplify AWS.
* Para poder acceder al sistema deberá ingresar con las claves usuario 'admin' y password 'admin'


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

### Model Identifación discapacidad.ipynb
* Se analizaron las variables, se desarrolló un EDA
* Se construyó un Modelo de Kmens para clusterizar a los clientes.

![alt text](https://71f5dd87-471e-4cda-bbd4-ff35f66518ff.s3.amazonaws.com/logo-ultimo.png)
