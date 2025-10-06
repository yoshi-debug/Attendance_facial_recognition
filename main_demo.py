"""
Demo Completo del Sistema de Captura y Detección Facial
Integra todos los componentes en un flujo único
"""

import os
import sys
from pathlib import Path

# Importar todos los módulos
try:
    from capture_dataset import DatasetCapture
    from detection_pipeline import FaceDetectionPipeline
    from preprocessing import FacePreprocessor
    from validation_tool import ValidationTool
except ImportError as e:
    print(f"Error importando módulos: {e}")
    print("Asegúrate de que todos los archivos estén en el mismo directorio")
    sys.exit(1)

class FacialRecognitionDemo:
    def __init__(self):
        """Inicializa el sistema completo"""
        self.base_dir = "demo_output"
        self.dataset_dir = os.path.join(self.base_dir, "dataset")
        self.processed_dir = os.path.join(self.base_dir, "processed")
        self.preprocessed_dir = os.path.join(self.base_dir, "preprocessed")
        
        # Crear directorios
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.preprocessed_dir, exist_ok=True)
        
        # Inicializar componentes
        self.capture = DatasetCapture(self.dataset_dir)
        self.pipeline = FaceDetectionPipeline(min_confidence=0.95)
        self.preprocessor = FacePreprocessor(target_size=(160, 160))
        self.validator = ValidationTool(self.dataset_dir)
        
        print("="*60)
        print("🎯 SISTEMA DE DETECCIÓN FACIAL - DEMO COMPLETO")
        print("="*60)
        print(f"Directorio base: {self.base_dir}")
        print(f"Dataset: {self.dataset_dir}")
        print(f"Procesado: {self.processed_dir}")
        print(f"Preprocesado: {self.preprocessed_dir}")
        print("="*60)
    
    def demo_capture(self, num_students=2, photos_per_student=10):
        """
        Demo de captura de dataset
        
        Args:
            num_students: Número de estudiantes a capturar
            photos_per_student: Fotos por estudiante
        """
        print("\n" + "="*60)
        print("📸 DEMO 1: CAPTURA DE DATASET")
        print("="*60)
        
        for i in range(1, num_students + 1):
            student_id = f"STU{i:03d}"
            print(f"\n➡️ Capturando estudiante: {student_id}")
            print(f"   Objetivo: {photos_per_student} fotos")
            print("\n   Instrucciones:")
            print("   - Presiona ESPACIO cuando el recuadro esté VERDE")
            print("   - Mueve ligeramente la cabeza entre fotos")
            print("   - Presiona 'q' para saltar al siguiente estudiante")
            
            input("\n   Presiona ENTER para iniciar captura...")
            
            self.capture.capture_for_student(student_id, photos_per_student)
            
            print(f"\n   ✅ Captura completada para {student_id}")
        
        # Mostrar estadísticas
        print("\n📊 Estadísticas de Captura:")
        for sid, count in self.capture.students_captured.items():
            print(f"   {sid}: {count} fotos capturadas")
    
    def demo_detection(self):
        """Demo del pipeline de detección"""
        print("\n" + "="*60)
        print("🔍 DEMO 2: PIPELINE DE DETECCIÓN")
        print("="*60)
        
        if not os.path.exists(self.dataset_dir):
            print("⚠️ No hay dataset para procesar. Ejecuta demo_capture primero.")
            return
        
        print("\n➡️ Procesando dataset capturado...")
        print(f"   Entrada: {self.dataset_dir}")
        print(f"   Salida: {self.processed_dir}")
        
        # Procesar cada estudiante
        student_dirs = [d for d in Path(self.dataset_dir).iterdir() if d.is_dir()]
        
        total_images = 0
        total_faces = 0
        
        for student_dir in student_dirs:
            student_id = student_dir.name
            output_student_dir = os.path.join(self.processed_dir, student_id)
            
            print(f"\n   Procesando: {student_id}")
            
            results = self.pipeline.process_batch(
                str(student_dir),
                output_student_dir,
                save_crops=True
            )
            
            total_images += results['processed']
            total_faces += results['faces_found']
            
            print(f"   ✅ {student_id}: {results['faces_found']} rostros detectados")
        
        # Estadísticas del pipeline
        stats = self.pipeline.get_statistics()
        
        print(f"\n📊 Estadísticas de Detección:")
        print(f"   Total de imágenes procesadas: {total_images}")
        print(f"   Total de rostros detectados: {total_faces}")
        print(f"   Tiempo promedio por imagen: {stats.get('avg_processing_time', 0):.3f}s")
        print(f"   Rostros con alta confianza: {stats.get('high_confidence', 0)}")
    
    def demo_preprocessing(self):
        """Demo del preprocesamiento"""
        print("\n" + "="*60)
        print("⚙️ DEMO 3: PREPROCESAMIENTO")
        print("="*60)
        
        if not os.path.exists(self.dataset_dir):
            print("⚠️ No hay dataset para preprocesar. Ejecuta demo_capture primero.")
            return
        
        print("\n➡️ Aplicando pipeline de preprocesamiento...")
        print(f"   Entrada: {self.dataset_dir}")
        print(f"   Salida: {self.preprocessed_dir}")
        print("\n   Pasos:")
        print("   1. Redimensionar a 160x160px")
        print("   2. Aplicar CLAHE (mejora de contraste)")
        print("   3. Filtrar imágenes borrosas")
        
        # Preprocesar dataset completo
        stats = self.preprocessor.create_preprocessed_dataset(
            self.dataset_dir,
            self.preprocessed_dir
        )
        
        print(f"\n📊 Estadísticas de Preprocesamiento:")
        print(f"   Estudiantes procesados: {stats['students']}")
        print(f"   Imágenes totales: {stats['total_images']}")
        print(f"   Imágenes preproc