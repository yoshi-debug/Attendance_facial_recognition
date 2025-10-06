import cv2
import os
from datetime import datetime
from mtcnn import MTCNN
import json

class DatasetCapture:
    def __init__(self, output_dir="dataset"):
        """
        Inicializa el sistema de captura de dataset
        
        Args:
            output_dir: Directorio donde se guardarán las imágenes
        """
        self.output_dir = output_dir
        self.detector = MTCNN()
        self.students_captured = {}
        
        # Crear directorio principal si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Cargar registro de capturas previas
        self.registry_file = os.path.join(output_dir, "registry.json")
        self.load_registry()
    
    def load_registry(self):
        """Carga el registro de estudiantes capturados"""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                self.students_captured = json.load(f)
    
    def save_registry(self):
        """Guarda el registro de estudiantes capturados"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.students_captured, f, indent=2)
    
    def validate_quality(self, face_data, frame):
        """
        Valida la calidad de la captura en tiempo real
        
        Args:
            face_data: Datos de detección de MTCNN
            frame: Frame actual de la cámara
            
        Returns:
            tuple: (is_valid, message)
        """
        if not face_data:
            return False, "No se detectó rostro"
        
        box = face_data['box']
        confidence = face_data['confidence']
        
        # Validar confianza mínima
        if confidence < 0.95:
            return False, f"Confianza baja: {confidence:.2f}"
        
        # Validar tamaño del rostro
        width, height = box[2], box[3]
        if width < 100 or height < 100:
            return False, "Rostro muy pequeño"
        
        # Validar iluminación (brillo promedio en región del rostro)
        x, y, w, h = box
        face_roi = frame[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        brightness = gray_face.mean()
        
        if brightness < 50:
            return False, "Muy oscuro"
        elif brightness > 200:
            return False, "Muy iluminado"
        
        return True, "Calidad OK"
    
    def capture_for_student(self, student_id, target_photos=40):
        """
        Captura fotos para un estudiante específico
        
        Args:
            student_id: ID del estudiante (ej: "STU001")
            target_photos: Número objetivo de fotos a capturar
        """
        # Crear carpeta del estudiante
        student_dir = os.path.join(self.output_dir, student_id)
        os.makedirs(student_dir, exist_ok=True)
        
        # Inicializar contador
        if student_id not in self.students_captured:
            self.students_captured[student_id] = 0
        
        photo_count = self.students_captured[student_id]
        
        # Iniciar captura de video
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print(f"\n=== Capturando para {student_id} ===")
        print(f"Objetivo: {target_photos} fotos")
        print(f"Ya capturadas: {photo_count}")
        print("\nControles:")
        print("  ESPACIO: Capturar foto")
        print("  'q': Salir")
        print("  's': Cambiar de estudiante")
        
        frame_skip = 0
        
        while photo_count < target_photos:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detectar rostro cada 3 frames para mejor rendimiento
            faces = []
            if frame_skip % 3 == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = self.detector.detect_faces(rgb_frame)
            frame_skip += 1
            
            # Crear copia para visualización
            display_frame = frame.copy()
            
            # Validar y dibujar
            quality_ok = False
            message = "Buscando rostro..."
            
            if faces:
                face = faces[0]  # Tomar el primer rostro detectado
                quality_ok, message = self.validate_quality(face, frame)
                
                # Dibujar bounding box
                box = face['box']
                x, y, w, h = box
                color = (0, 255, 0) if quality_ok else (0, 0, 255)
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                
                # Dibujar landmarks
                keypoints = face['keypoints']
                for key, point in keypoints.items():
                    cv2.circle(display_frame, point, 2, (255, 255, 0), -1)
                
                # Mostrar confianza
                conf_text = f"Conf: {face['confidence']:.2f}"
                cv2.putText(display_frame, conf_text, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Información en pantalla
            progress = f"{photo_count}/{target_photos}"
            cv2.putText(display_frame, f"Estudiante: {student_id}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Progreso: {progress}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, message, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0, 255, 0) if quality_ok else (0, 0, 255), 2)
            
            cv2.imshow('Captura de Dataset', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Capturar foto con ESPACIO
            if key == ord(' ') and quality_ok and faces:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{student_id}_{photo_count+1:03d}_{timestamp}.jpg"
                filepath = os.path.join(student_dir, filename)
                
                # Guardar imagen
                cv2.imwrite(filepath, frame)
                
                # Guardar metadata
                metadata = {
                    'timestamp': timestamp,
                    'confidence': float(face['confidence']),
                    'box': box,
                    'keypoints': {k: list(v) for k, v in face['keypoints'].items()},
                    'resolution': frame.shape[:2]
                }
                
                metadata_file = filepath.replace('.jpg', '_metadata.json')
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                photo_count += 1
                self.students_captured[student_id] = photo_count
                self.save_registry()
                
                print(f"✓ Foto {photo_count} capturada: {filename}")
            
            # Salir
            elif key == ord('q'):
                break
            
            # Cambiar estudiante
            elif key == ord('s'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✓ Captura completada para {student_id}")
        print(f"Total de fotos: {photo_count}/{target_photos}")

def main():
    """Función principal para ejecutar el sistema de captura"""
    capture = DatasetCapture()
    
    print("=== Sistema de Captura de Dataset Facial ===\n")
    
    while True:
        print("\nOpciones:")
        print("1. Capturar nuevo estudiante")
        print("2. Continuar captura de estudiante existente")
        print("3. Ver estadísticas")
        print("4. Salir")
        
        choice = input("\nSeleccione una opción: ")
        
        if choice == '1':
            student_id = input("Ingrese ID del estudiante (ej: STU001): ").strip()
            if student_id:
                num_photos = input("Número de fotos objetivo (default 40): ").strip()
                target = int(num_photos) if num_photos else 40
                capture.capture_for_student(student_id, target)
        
        elif choice == '2':
            if not capture.students_captured:
                print("No hay estudiantes en el registro")
                continue
            
            print("\nEstudiantes registrados:")
            for sid, count in capture.students_captured.items():
                print(f"  {sid}: {count} fotos")
            
            student_id = input("\nSeleccione ID de estudiante: ").strip()
            if student_id in capture.students_captured:
                target = int(input("Número de fotos objetivo: "))
                capture.capture_for_student(student_id, target)
        
        elif choice == '3':
            if not capture.students_captured:
                print("No hay datos capturados aún")
            else:
                print("\n=== Estadísticas ===")
                total_photos = sum(capture.students_captured.values())
                print(f"Total de estudiantes: {len(capture.students_captured)}")
                print(f"Total de fotos: {total_photos}")
                print("\nDetalle por estudiante:")
                for sid, count in capture.students_captured.items():
                    print(f"  {sid}: {count} fotos")
        
        elif choice == '4':
            print("Saliendo...")
            break

if __name__ == "__main__":
    main()
