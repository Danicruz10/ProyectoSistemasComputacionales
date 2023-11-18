import os
import cv2
import pandas as pd
from controller import cvision
from collections import Counter

# Directorio principal que contiene carpetas de estudiantes
dir_Path_student = 'student/' # ruta de carpeta principal estudiante

# Inicializar un diccionario para almacenar las frecuencias de emociones por estudiante, categoría y pregunta
emotion_Student = {}

frame_count = 0 #contador de frames
size_group = 20 # tamaño del de los frame que procesa los casi 30 o mas fps por segundo

# Recorrer carpetas de estudiantes
list_student = sorted(os.listdir(dir_Path_student))
for Student in list_student:  # list_student se crea para obtener una lista de nombres de carpetas de estudiantes en el directorio principal y luego se iterando sobre los elementos dentro /student directorio principal 
    Path_Student = os.path.join(dir_Path_student, Student)
    
    # Verificar si es un directorio (ignorar archivos en la carpeta principal)

    if os.path.isdir(Path_Student):
        emotion_Student[Student] = {}
        
        # Recorrer carpetas de categorías para el estudiante actual
        list_dir_Student = sorted(os.listdir(Path_Student))
        for skill in list_dir_Student:
            Path_skill = os.path.join(Path_Student, skill)
            
            # Verificar si es un directorio
            if os.path.isdir(Path_skill):
                emotion_Student[Student][skill] = {}
                
                # Recorrer carpetas de preguntas para la categoría actual
                list_dir_skills = sorted(os.listdir(Path_skill))
                for question in list_dir_skills:
                    Path_question = os.path.join(Path_skill, question)
                    
                    # Verificar si es un directorio y si contiene archivos
                    if os.path.isdir(Path_question) and any(os.listdir(Path_question)):
                        emotion_frequency = {'Neutral': 0, 'Happy': 0, 'Sad': 0, 'Fear': 0, 'Disgust': 0, 'Surprise': 0, 'Anger': 0, 'Contempt': 0}

                        # Procesar cada video para la pregunta actual
                        list_Path_question = sorted(os.listdir(Path_question))
                        for video_file in list_Path_question:
                            Path_video = os.path.join(Path_question, video_file)
                            print("\n")
                            print(f"Procesanso video de la pregunta:{question} de la categoria:{skill} del estudiante:{Student}")
                            cap = cv2.VideoCapture(Path_video)
                            
                            if not cap.isOpened():
                                print(f"Error al abrir el video {Path_video}")
                                continue

                            frame_count = 0
                            # Procesar cada frame
                            while True:
                                ret, frame = cap.read()
                                frame_count+=1
                                if not ret:
                                    break
                                if frame_count % size_group == 0: #si la division de ese frame es divisible entre el tamaño del size_group entra a la condicion, en este caso de 5 en 5
                                    # Llamar a la función para reconocer la expresión facial
                                    emotions = cvision.recognize_facial_expression(frame, False, 1, False)

                                    # Contar la frecuencia de cada emoción en el vector de emociones
                                    printList = print(emotions.list_emotion) # vector(lista) de emociones de un frame
                                    emotion_counter = Counter(emotions.list_emotion)

                                    # Actualizar el diccionario de frecuencias
                                    for emotion, count in emotion_counter.items():
                                        emotion_frequency[emotion] += count

                            # Liberar el objeto de captura de video
                            cap.release()

                        # Almacenar las frecuencias de emociones para la combinación actual de estudiante, categoría y pregunta
                        emotion_Student[Student][skill][question] = emotion_frequency


# Imprimir el resultado
for Student, skills in emotion_Student.items():# 
    for skill, questions in skills.items():
        for question, emotions in questions.items(): # tupla pregunta, emociones:que es el valor e itera en pregunta
            print(f"Estudiante: {Student}, Categoría: {skill}, Pregunta: {question}")
            most_frequent_emotion = max(emotions, key=emotions.get)
            print("Emoción más frecuente:", most_frequent_emotion)
            print("Frecuencia de cada emoción:", emotions)
            print("\n")

emotion_id_mapping = {
    'Neutral': 0,
    'Happy': 1,
    'Sad': 2,
    'Fear': 3,
    'Disgust': 4,
    'Surprise': 5,
    'Anger': 6,
    'Contempt': 7
}

Path_csv = '../../DatasetC.csv'
# Cargar el archivo CSV en un DataFrame con la columna 'emocion' configurada como objeto (cadena)
df = pd.read_csv(Path_csv, dtype={'emotion': object})
df['emotion_id'] = None  # Nueva columna para los IDs de emoción
df['emotion'] = None
# Actualizar el DataFrame con las emociones
for index, row in df.iterrows():
    Student = 'student_'+str(row['user_id'])
    skill = 'skill_'+ str(row['skill_id'])
    question = 'question_'+str(row['question_id'])

    if Student in emotion_Student and skill in emotion_Student[Student] and question in emotion_Student[Student][skill]:
        emotions = emotion_Student[Student][skill][question] #valor de emocion
        most_frequent_emotion = max(emotions, key=emotions.get)
        df.loc[index, 'emotion'] = most_frequent_emotion
        df.loc[index, 'emotion_id'] = emotion_id_mapping.get(most_frequent_emotion, None)

# Guardar el DataFrame actualizado en el archivo CSV
df.to_csv(Path_csv, index=False)
