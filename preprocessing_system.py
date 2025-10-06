import cv2
import numpy as np
import os
from pathlib import Path

class FacePreprocessor:
    def __init__(self, target_size=(160, 160)):
        """
        Sistema de preprocesamiento de imágenes faciales
        
        Args:
            target_size: Tamaño objetivo para redimensionar (width, height)
        """
        self.target_size = target_size
        self.preprocessing_stats = {
            'processed': 0,
            'low_confidence_filtered': 0,
            'histogram_equalized': 0
        }
    
    def standardize_resize(self, image):
        """
        Redimensiona la imagen al tamaño estándar
        
        Args:
            image: Imagen a redimensionar
            
        Returns:
            numpy.ndarray: Imagen redimensionada
        """
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
    
    def normalize_pixels(self, image):
        """
        Normaliza los valores de píxeles a rango [0, 1]
        
        Args:
            image: Imagen a normalizar
            
        Returns:
            numpy.ndarray: Imagen normalizada
        """
        # Convertir a float32 y normalizar
        normalized = image.astype('float32')
        normalized = normalized / 255.0
        return normalized
    
    def histogram_equalization(self, image):
        """
        Aplica ecualización de histograma para mejorar contraste
        
        Args:
            image: Imagen BGR
            
        Returns:
            numpy.ndarray: Imagen ecualizada
        """
        # Convertir a YCrCb
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # Ecualizar el canal Y (luminancia)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        
        # Convertir de vuelta a BGR
        equalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        
        self.preprocessing_stats['histogram_equalized'] += 1
        
        return equalized
    
    def adaptive_histogram_equalization(self, image, clip_limit=2.0, tile_size=(8, 8)):
        """
        Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Args:
            image: Imagen BGR
            clip_limit: Límite de contraste
            tile_size: Tamaño de los tiles
            
        Returns:
            numpy.ndarray: Imagen con CLAHE aplicado
        """
        # Convertir a LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Aplicar CLAHE al canal L
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convertir de vuelta a BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def filter_low_confidence(self, image_dir, metadata_dir, confidence_threshold=0.95):
        """
        Filtra imágenes con baja confianza de detección
        
        Args:
            image_dir: Directorio con imágenes
            metadata_dir: Directorio con archivos de metadata
            confidence_threshold: Umbral mínimo de confianza
            
        Returns:
            list: Lista de archivos que pasaron el filtro
        """
        import json
        
        valid_files = []
        
        for metadata_file in Path(metadata_dir).glob("*_metadata.json"):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            if metadata['confidence'] >= confidence_threshold:
                # Obtener nombre del archivo de imagen correspondiente
                image_file = str(metadata_file).replace('_metadata.json', '.jpg')
                if os.path.exists(image_file):
                    valid_files.append(image_file)
            else:
                self.preprocessing_stats['low_confidence_filtered'] += 1
        
        return valid_files
    
    def detect_blur(self, image, threshold=100):
        """
        Detecta si una imagen está borrosa usando el operador Laplaciano
        
        Args:
            image: Imagen a evaluar
            threshold: Umbral para considerar imagen borrosa
            
        Returns:
            tuple: (is_sharp, variance)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        is_sharp = laplacian_var >= threshold
        return is_sharp, laplacian_var
    
    def augment_image(self, image, augmentation_type='brightness'):
        """
        Aplica aumentación de datos a una imagen
        
        Args:
            image: Imagen original
            augmentation_type: Tipo de aumentación a aplicar
            
        Returns:
            numpy.ndarray: Imagen aumentada
        """
        if augmentation_type == 'brightness':
            # Ajuste de brillo aleatorio
            factor = np.random.uniform(0.7, 1.3)
            augmented = cv2.convertScaleAbs(image, alpha=factor, beta=0)
            
        elif augmentation_type == 'flip':
            # Volteo horizontal
            augmented = cv2.flip(image, 1)
            
        elif augmentation_type == 'rotation':
            # Rotación leve
            angle = np.random.uniform(-15, 15)
            center = (image.shape[1] // 2, image.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            augmented = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            
        elif augmentation_type == 'noise':
            # Añadir ruido gaussiano
            noise = np.random.randn(*image.shape) * 10
            augmented = np.clip(image + noise, 0, 255).astype(np.uint8)
            
        else:
            augmented = image.copy()
        
        return augmented
    
    def preprocess_pipeline(self, image, apply_clahe=True, apply_normalization=True):
        """
        Pipeline completo de preprocesamiento
        
        Args:
            image: Imagen a preprocesar
            apply_clahe: Si aplicar CLAHE
            apply_normalization: Si normalizar píxeles
            
        Returns:
            numpy.ndarray: Imagen preprocesada
        """
        # 1. Redimensionar al tamaño estándar
        processed = self.standardize_resize(image)
        
        # 2. Aplicar CLAHE si se requiere
        if apply_clahe:
            processed = self.adaptive_histogram_equalization(processed)
        
        # 3. Normalizar píxeles si se requiere
        if apply_normalization:
            processed = self.normalize_pixels(processed)
        
        self.preprocessing_stats['processed'] += 1
        
        return processed
    
    def batch_preprocess(self, input_dir, output_dir, quality_check=True):
        """
        Preprocesa un lote completo de imágenes
        
        Args:
            input_dir: Directorio con imágenes originales
            output_dir: Directorio donde guardar imágenes procesadas
            quality_check: Si realizar verificación de calidad
            
        Returns:
            dict: Estadísticas del preprocesamiento
        """
        os.makedirs(output_dir, exist_ok=True)
        
        stats = {
            'total_files': 0,
            'processed': 0,
            'rejected_blur': 0,
            'rejected_quality': 0
        }
        
        print(f"\nPreprocesando imágenes de {input_dir}...")
        
        # Obtener todos los archivos de imagen
        image_files = list(Path(input_dir).glob("*.jpg")) + \
                     list(Path(input_dir).glob("*.jpeg")) + \
                     list(Path(input_dir).glob("*.png"))
        
        stats['total_files'] = len(image_files)
        
        for image_path in image_files:
            try:
                # Leer imagen
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                
                # Verificar calidad si se requiere
                if quality_check:
                    is_sharp, blur_score = self.detect_blur(image)
                    if not is_sharp:
                        stats['rejected_blur'] += 1
                        print(f"✗ {image_path.name}: Imagen borrosa (score: {blur_score:.2f})")
                        continue
                
                # Aplicar pipeline de preprocesamiento
                processed = self.preprocess_pipeline(image, 
                                                    apply_clahe=True,
                                                    apply_normalization=False)
                
                # Guardar imagen preprocesada
                output_path = os.path.join(output_dir, image_path.name)
                
                # Convertir de vuelta a uint8 si está normalizada
                if processed.dtype == np.float32:
                    processed = (processed * 255).astype(np.uint8)
                
                cv2.imwrite(output_path, processed)
                stats['processed'] += 1
                
                print(f"✓ {image_path.name}: Preprocesado exitosamente")
                
            except Exception as e:
                print(f"✗ Error procesando {image_path.name}: {str(e)}")
                stats['rejected_quality'] += 1
        
        print(f"\n=== Resumen del Preprocesamiento ===")
        print(f"Total de archivos: {stats['total_files']}")
        print(f"Procesados exitosamente: {stats['processed']}")
        print(f"Rechazados por desenfoque: {stats['rejected_blur']}")
        print(f"Rechazados por otros errores: {stats['rejected_quality']}")
        
        return stats
    
    def create_preprocessed_dataset(self, dataset_dir, output_dir):
        """
        Crea un dataset completo preprocesado manteniendo estructura
        
        Args:
            dataset_dir: Directorio raíz del dataset original
            output_dir: Directorio donde guardar dataset preprocesado
        """
        print(f"\n=== Creando Dataset Preprocesado ===")
        print(f"Origen: {dataset_dir}")
        print(f"Destino: {output_dir}")
        
        total_stats = {
            'students': 0,
            'total_images': 0,
            'processed_images': 0
        }
        
        # Iterar por cada carpeta de estudiante
        for student_dir in Path(dataset_dir).iterdir():
            if not student_dir.is_dir():
                continue
            
            student_id = student_dir.name
            
            # Saltar carpeta de registro
            if student_id == 'registry.json':
                continue
            
            print(f"\nProcesando estudiante: {student_id}")
            
            # Crear carpeta de salida para el estudiante
            student_output_dir = os.path.join(output_dir, student_id)
            
            # Preprocesar imágenes del estudiante
            stats = self.batch_preprocess(str(student_dir), student_output_dir)
            
            total_stats['students'] += 1
            total_stats['total_images'] += stats['total_files']
            total_stats['processed_images'] += stats['processed']
        
        print(f"\n=== Estadísticas Finales ===")
        print(f"Estudiantes procesados: {total_stats['students']}")
        print(f"Imágenes totales: {total_stats['total_images']}")
        print(f"Imágenes preprocesadas: {total_stats['processed_images']}")
        
        return total_stats

def main():
    """Función principal para ejecutar el preprocesamiento"""
    print("=== Sistema de Preprocesamiento Facial ===\n")
    
    preprocessor = FacePreprocessor(target_size=(160, 160))
    
    print("Opciones:")
    print("1. Preprocesar dataset completo")
    print("2. Preprocesar carpeta única")
    print("3. Preprocesar imagen única (demo)")
    print("4. Verificar calidad de imágenes")
    
    choice = input("\nSeleccione opción: ")
    
    if choice == '1':
        dataset_dir = input("Directorio del dataset original: ").strip()
        output_dir = input("Directorio de salida: ").strip()
        
        if os.path.exists(dataset_dir):
            preprocessor.create_preprocessed_dataset(dataset_dir, output_dir)
    
    elif choice == '2':
        input_dir = input("Directorio de entrada: ").strip()
        output_dir = input("Directorio de salida: ").strip()
        
        if os.path.exists(input_dir):
            preprocessor.batch_preprocess(input_dir, output_dir)
    
    elif choice == '3':
        image_path = input("Ruta de la imagen: ").strip()
        
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            
            # Mostrar original
            cv2.imshow('Original', image)
            
            # Preprocesar
            processed = preprocessor.preprocess_pipeline(image, 
                                                        apply_clahe=True,
                                                        apply_normalization=False)
            
            # Mostrar preprocesada
            cv2.imshow('Preprocesada', processed)
            
            print("\nPresione cualquier tecla para cerrar...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    elif choice == '4':
        input_dir = input("Directorio a verificar: ").strip()
        
        if os.path.exists(input_dir):
            print("\nVerificando calidad de imágenes...\n")
            
            blur_count = 0
            total_count = 0
            
            for image_file in Path(input_dir).glob("*.jpg"):
                image = cv2.imread(str(image_file))
                if image is None:
                    continue
                
                total_count += 1
                is_sharp, score = preprocessor.detect_blur(image)
                
                status = "✓ OK" if is_sharp else "✗ BORROSA"
                print(f"{status} - {image_file.name}: {score:.2f}")
                
                if not is_sharp:
                    blur_count += 1
            
            print(f"\n=== Resumen ===")
            print(f"Total: {total_count}")
            print(f"Nítidas: {total_count - blur_count}")
            print(f"Borrosas: {blur_count}")

if __name__ == "__main__":
    main()