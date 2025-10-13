# Sistema de Asistencia Automática por Reconocimiento Facial 🎓

> Un sistema inteligente basado en **Deep Learning** para registrar la asistencia de estudiantes en clase, mediante el **reconocimiento simultáneo de rostros** capturados por cámara o video en tiempo real.

---

## 🧠 Descripción General

El **Sistema de Asistencia Automática** detecta y reconoce las caras de los estudiantes presentes en el aula, registrando su asistencia automáticamente en una hoja de cálculo.  
Combina **MTCNN** para la detección de rostros, **FaceNet** para la generación de embeddings únicos por individuo, y un clasificador **SVM** para la identificación final.  
La aplicación web fue construida con **Flask**, permitiendo una interfaz simple y funcional para el docente.

---

## 🧩 Arquitectura del Proyecto

root/
│
├── attendance/
│ ├── facenet/
│ │ ├── dataset/
│ │ │ ├── raw/ # Imágenes originales de los estudiantes
│ │ │ └── aligned/ # Rostros detectados y alineados
│ │ ├── src/
│ │ │ ├── align/ # Scripts para detección y alineación facial (MTCNN)
│ │ │ ├── classifier.py # Entrenamiento SVM y embeddings (FaceNet)
│ │ │ └── 20180402-114759/ # Modelo preentrenado FaceNet
│ │ └── Reports/ # Carpeta de reportes Excel generados
│ └── database/ # Base de datos SQLite
│
├── static/ # Archivos estáticos (CSS, JS, imágenes)
├── templates/ # Vistas HTML (interfaz web Flask)
├── run.py # Archivo principal de ejecución del servidor
├── requirements.txt # Dependencias del proyecto
└── README.md # Este archivo

---

## ⚙️ Tecnologías Principales

- **Python 3.x**
- **TensorFlow** (para embeddings de FaceNet)
- **OpenCV** (procesamiento de video e imágenes)
- **Flask** (backend web)
- **MTCNN** (detección facial)
- **FaceNet** (reconocimiento facial)
- **SVM** (clasificador de rostros)
- **SQLite** (base de datos de estudiantes)
- **XlsxWriter** (generación de reportes en Excel)

---

## 🚀 Instalación y Configuración

### 1. Clonar el Repositorio
git clone https://github.com/tu_usuario/Face-Attendance-System.git
cd Face-Attendance-System

### 2. Clonar el Repositorio

python -m venv venv
source venv/bin/activate  # En Linux/Mac
venv\Scripts\activate     # En Windows

### 3. Instalar Dependencias
pip install -r requirements.txt

📸 Pipeline de Procesamiento
Paso 1: Captura o Carga de Imágenes

Cada estudiante debe tener al menos 10 imágenes de entrenamiento, almacenadas en:
attendance/facenet/dataset/raw/Nombre_Estudiante/

attendance/facenet/dataset/raw/
    ├── Ana/
    ├── Carlos/
    └── Lucía/

Paso 2: Detección y Alineación Facial (MTCNN)
