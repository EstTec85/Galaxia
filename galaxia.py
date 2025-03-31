# %% [markdown]
# # **Introducción**
# 

# %% [markdown]
# Este proyecto busca clasificar imágenes de galaxias en dos categorías:
# 
# 
# 
# 1.   **Galaxias con mareas** (estructuras alargadas debido a interacciones gravitacionales).
# 2.   **Galaxias sin mareas** (sin deformaciones gravitacionales).
# 
# 
# Para ello, se utilizan:
# 
# 
# *   **MongoDB** como base de datos para almacenar y procesar los datos vectorizados de las imágenes.
# 
# *  **Vectores** extraídos de las imágenes para representar características numéricas
# 
# *  ** Redes neuronales convolucionales** (CNN) en PyTorch para realizar la clasificación
# 
# 
# 
# 
# 
# 
# 

# %% [markdown]
# # Uso de **MongoDB**

# %% [markdown]
# **¿Por qué MongoDB?**
# 
# MongoDB es una base de datos NoSQL, lo que permite almacenar grandes volúmenes de datos de manera flexible.
# 
# Se elige porque:
# 
# 
# 
# 1.   **Manejo eficiente de imágenes**: Permite almacenar imágenes y sus características en documentos BSON sin esquemas rígidos.
# 
# 2.   **Escalabilidad**: Ideal para almacenar grandes conjuntos de datos, como imágenes de galaxias.
# 
# 1.   **Facilidad de consulta**: Se pueden realizar búsquedas rápidas de imágenes y recuperar vectores asociados.
# 
# El código se conecta a MongoDB Atlas, una versión en la nube de MongoDB:
# 
# 
# 
# 
# 
# 
# 

# %%
import pymongo

client = pymongo.MongoClient("mongodb+srv://usuario:contraseña@universo.mongodb.net/")
db = client["galaxias_db"]
mareas_collection = db["mareas_vectors"]
no_mareas_collection = db["no_mareas_vectors"]


# %% [markdown]
# Se crean dos colecciones en la base de datos:
# 
# 
# *   mareas_vectors: Contiene galaxias con mareas.
# 
# *   no_mareas_vectors: Contiene galaxias sin mareas.
# 
# 

# %% [markdown]
# # Creación y Uso de Vectores

# %% [markdown]
# **¿Por qué convertir imágenes en vectores?**
# 
# Las imágenes no pueden ser procesadas directamente por modelos de Machine Learning tradicionales. Por eso:
# 
# 
# 
# 1.   Se **extraen características numéricas** para representar cada imagen.
# 
# 1.   Se **almacenan los vectores en MongoDB** para facilitar el entrenamiento del modelo.
# 
# 2.   Se permite **reutilizar los datos** sin necesidad de procesar imágenes cada vez
# 
# 
# 
# 
# 
# 

# %% [markdown]
# #Conversión de imágenes a vectores
# 
# Cada imagen es cargada, transformada en escala de grises y normalizada:

# %%
def load_vectors_from_mongodb():
    mareas = list(mareas_collection.find({}, {"vector": 1, "class": 1, "_id": 0}))
    no_mareas = list(no_mareas_collection.find({}, {"vector": 1, "class": 1, "_id": 0}))

    data = mareas + no_mareas
    X = np.array([entry["vector"] for entry in data])
    y = np.array([entry["class"] for entry in data])

    return X, y


# %% [markdown]
# 
# 
# *   **Cada imagen se convierte en un vector** de características numéricas (ej. histogramas de color, bordes, texturas, etc.).
# 
# *   **MongoDB almacena estos vectores**, facilitando su recuperación y procesamiento.
# 
# 

# %% [markdown]
# # Preprocesamiento de Imágenes
# 
# Antes de entrenar el modelo, las imágenes se preprocesan para mejorar su calidad y homogeneidad:
# 
# 
# 
# 1.   **Conversión a escala de grises**
# 
# 
# *   Se eliminan las variaciones de color para enfocarse en las estructuras de la galaxia.
# 
# 
# 
# 2.   **Redimensionamiento a 128x128 píxeles**
# 
# 
# 
# *   Estandariza el tamaño para que el modelo trabaje con entradas consistentes.
# 
# 
# 
# 
# 3.  **Normalización**
# 
# 
# *   Se ajustan los valores de los píxeles a un rango entre 0 y 1 para mejorar el entrenamiento
# 
# 
# 
# 

# %%
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))
    return image.flatten() / 255.0  # Normalización


# %% [markdown]
# # Modelos de Clasificación

# %% [markdown]
# **Uso de Modelos Basados en Vectores**
# 
# Se prueban varios modelos tradicionales de Machine Learning:
# 
# 
# 
# *   **SVM (Máquinas de Soporte Vectorial)**
# 
# *   **Random Forest**
# *   **MLP (Perceptrón Multicapa)**
# 
# 
# Se dividen los datos en entrenamiento y prueba:
# 
# 
# 
# 
# 

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %% [markdown]
# Entrenamiento de los modelos:

# %%
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)


# %% [markdown]
# # **Modelo con CNN (Red Neuronal Convolucional)**

# %% [markdown]
# Las CNN son ideales para clasificación de imágenes. Se implementa una CNN en PyTorch con la siguiente arquitectura:

# %%
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 64 * 64, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


# %% [markdown]
# 
# 
# *   *Conv2d(1, 16, kernel_size=3)*: Detecta bordes y patrones en la imagen.
# 
# *   *MaxPool2d(kernel_size=2)*: Reduce el tamaño de la imagen.
# *   *fc1 = nn.Linear(16 * 64 * 64, 2)*: Clasifica la galaxia en una de dos categorías.
# 
# 
# 
# Entrenamiento del modelo:
# 
# 

# %%
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)

for epoch in range(10):
    optimizer.zero_grad()
    outputs = cnn_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()


# %% [markdown]
# El modelo aprende en 10 épocas, minimizando la pérdida (loss) en cada iteración.

# %% [markdown]
# # **Explicación del método Graph Cut y su aplicación en galaxias**
# 

# %% [markdown]
# **¿Qué es Graph Cut?**
# 
# El **Graph Cut Model** es una técnica de segmentación de imágenes basada en grafos. Se usa para dividir una imagen en diferentes regiones utilizando la teoría de grafos, minimizando una función de energía.
# Se representa la imagen como un grafo en donde:
# Los píxeles son nodos.
# 
# Las conexiones entre píxeles son aristas con pesos que representan la similitud entre píxeles adyacentes.
# El objetivo es encontrar el corte óptimo en el grafo que separe los píxeles de interés (foreground, lo que queremos segmentar) de los que no lo son (background).

# %% [markdown]
# ** Relación con el PDF de Wetlands**
# 
# El artículo sobre **Wetlands usa Graph Cut** para segmentación en imágenes de teledetección.
# Se enfoca en identificar regiones de humedales basándose en:
# Características de textura y color.
# 
# *   Características de textura y color.
# 
# *   Agrupamiento de píxeles en regiones homogéneas.
# 
# *   Minimización de una función de energía para separar regiones relevantes.
# 
# En la segmentación de galaxias, aplicamos una idea similar: **identificar estructuras específicas (mareas)** en imágenes astronómicas utilizando Graph Cut.

# %% [markdown]
# **Aplicación **texto en negrita** de Graph Cut en la segmentación de galaxias**
# 
# En el código que se implemento, Graph Cut se usó para mejorar la segmentación de imágenes astronómicas y resaltar las regiones de interés (mareas).
# 
# 
# 1.   **Preprocesamiento**: Aplico técnicas como histogram equalization y morphological opening para mejorar el contraste y eliminar ruido.
# 
# 2.   **Definir la máscara inicial**: Se creó una máscara que diferencia regiones potencialmente relevantes de las que no lo son.
# 
# 3.  **Aplicación de Graph Cut**: Se ejecutó cv2.grabCut(), que:
# Etiqueta píxeles como foreground (marea) o background (fondo de la galaxia).
# Usa una función de energía para encontrar la mejor separación.
# 
# 
# 4. **Post-procesamiento**: Se refinó la segmentación para extraer solo las estructuras relevantes.  
# 
# 

# %%
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.feature import local_binary_pattern, hog
from skimage.filters import gabor

# Parámetros
TARGET_SIZE = (128, 128)  # Tamaño uniforme para todas las imágenes

# Técnicas de preprocesamiento
def resize_and_normalize(image):
    resized = cv2.resize(image, TARGET_SIZE)
    return (resized * 255).astype(np.uint8)  # Asegurar uint8 después de normalización

def convert_to_grayscale(image):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)  # Convertir a uint8 si es necesario
    if len(image.shape) == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale = image
    return cv2.resize(grayscale, TARGET_SIZE)

def histogram_equalization(image):
    grayscale = convert_to_grayscale(image)
    return cv2.equalizeHist(grayscale)

def gaussian_blur(image):
    image = convert_to_grayscale(image)
    return cv2.GaussianBlur(image, (5, 5), 0)

PREPROCESSING_TECHNIQUES = {
    "resize_normalize": resize_and_normalize,
    "grayscale": convert_to_grayscale,
    "histogram_equalization": histogram_equalization,
    "gaussian_blur": gaussian_blur
}

# Extraer características locales con LBP, HOG y Gabor
def extract_local_features(image):
    grayscale = convert_to_grayscale(image)

    # Local Binary Patterns (LBP)
    lbp = local_binary_pattern(grayscale, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)

    # Histogram of Oriented Gradients (HOG)
    hog_features = hog(grayscale, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)

    # Gabor Filters
    gabor_features = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        filt_real, _ = gabor(grayscale, frequency=0.6, theta=theta)
        gabor_features.append(filt_real.mean())

    # Concatenar todas las características
    features = np.hstack([lbp_hist, hog_features, gabor_features])

    # Evitar NaN o valores infinitos
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features

# Aplicar Graph Cut para segmentación
def graph_cut_segmentation(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[-1] == 1:
        image = cv2.merge([image, image, image])

    image = np.uint8(image)

    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (10, 10, image.shape[1]-10, image.shape[0]-10)
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented_image = image * mask2[:, :, np.newaxis]
    segmented_image = np.nan_to_num(segmented_image, nan=0, posinf=255, neginf=0).astype(np.uint8)  # Normalización
    return segmented_image

# Evaluar imágenes con diferentes técnicas de preprocesamiento
sample_images = df_mareas.sample(13, random_state=42)  # Selección de 13 imágenes
results = []

fig, axes = plt.subplots(len(sample_images), 5, figsize=(18, len(sample_images) * 3))

for i, (_, row) in tqdm(enumerate(sample_images.iterrows()), total=len(sample_images)):
    img_name = row['img_name']
    img_path = os.path.join(image_folder, img_name)

    # Cargar imagen
    image = cv2.imread(img_path)
    if image is None:
        print(f"⚠️ No se encontró la imagen: {img_name}")
        continue

    # Asegurar formato correcto
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    technique_scores = {}
    for name, func in PREPROCESSING_TECHNIQUES.items():
        preprocessed_img = func(image)
        segmented_img = graph_cut_segmentation(preprocessed_img)
        features = extract_local_features(segmented_img)
        score = np.sum(features)
        technique_scores[name] = score

    best_technique = max(technique_scores, key=technique_scores.get)
    best_score = technique_scores[best_technique]

    # Mostrar imágenes
    axes[i, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[i, 0].set_title(f"Original: {img_name}")
    axes[i, 0].axis("off")

    preprocessed_img = PREPROCESSING_TECHNIQUES[best_technique](image)
    axes[i, 1].imshow(preprocessed_img, cmap="gray")
    axes[i, 1].set_title(f"Mejor Preprocesado: {best_technique}")
    axes[i, 1].axis("off")

    segmented_img = graph_cut_segmentation(preprocessed_img)
    axes[i, 2].imshow(segmented_img, cmap="gray")
    axes[i, 2].set_title("Segmentación Graph Cut")
    axes[i, 2].axis("off")

    # Guardar resultados
    results.append({
        "img_name": img_name,
        "best_preprocessing_technique": best_technique,
        "score": best_score
    })

plt.tight_layout()
plt.show()

# Convertir resultados a DataFrame
df_results = pd.DataFrame(results)


# %% [markdown]
# **Posibles causas del error en Graph Cut:**
# 
# El fondo es demasiado oscuro
# 
# Graph Cut funciona mejor cuando hay una diferencia clara entre el objeto de interés y el fondo.
# En imágenes astronómicas, el fondo es negro y puede no proporcionar suficiente información para diferenciar foreground y background.
# Inicialización incorrecta de la máscara
# 
# En cv2.grabCut(), se usa un rectángulo para definir la región de foreground, pero si la selección no es precisa, puede dejar muchas áreas sin procesar.
# Esto puede hacer que el algoritmo clasifique gran parte de la imagen como fondo.
# Ruido y estrellas como foreground falso
# 
# Graph Cut depende de modelos de color para separar foreground y background.
# En las imágenes del telescopio, muchas estrellas aparecen como píxeles brillantes aislados, lo que podría hacer que el algoritmo las considere foreground.
# Distribución no uniforme de los píxeles
# 
# cv2.grabCut() usa modelos de distribución de color, y si la imagen tiene muchos tonos similares en diferentes regiones, puede fallar al identificar la estructura correcta.
# 

# %%
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.feature import local_binary_pattern, hog
from skimage.filters import gabor

# Parámetros
TARGET_SIZE = (128, 128)  # Tamaño uniforme para todas las imágenes

# Técnicas de preprocesamiento
def resize_and_normalize(image):
    resized = cv2.resize(image, TARGET_SIZE)
    return (resized * 255).astype(np.uint8)  # Asegurar uint8 después de normalización

def convert_to_grayscale(image):
    if image is None:
        return None
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)  # Convertir a uint8 si es necesario
    if len(image.shape) == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale = image
    return cv2.resize(grayscale, TARGET_SIZE)

def histogram_equalization(image):
    grayscale = convert_to_grayscale(image)
    if grayscale is None:
        return None
    return cv2.equalizeHist(grayscale)

def gaussian_blur(image):
    image = convert_to_grayscale(image)
    if image is None:
        return None
    return cv2.GaussianBlur(image, (5, 5), 0)

PREPROCESSING_TECHNIQUES = {
    "resize_normalize": resize_and_normalize,
    "grayscale": convert_to_grayscale,
    "histogram_equalization": histogram_equalization,
    "gaussian_blur": gaussian_blur
}

# Extraer características locales con LBP, HOG y Gabor
def extract_local_features(image):
    grayscale = convert_to_grayscale(image)
    if grayscale is None:
        return np.zeros(10)

    # Local Binary Patterns (LBP)
    lbp = local_binary_pattern(grayscale, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)

    # Histogram of Oriented Gradients (HOG)
    hog_features = hog(grayscale, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)

    # Gabor Filters
    gabor_features = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        filt_real, _ = gabor(grayscale, frequency=0.6, theta=theta)
        gabor_features.append(filt_real.mean())

    # Concatenar todas las características
    features = np.hstack([lbp_hist, hog_features, gabor_features])

    # Evitar NaN o valores infinitos
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features

# Aplicar Graph Cut para segmentación
def graph_cut_segmentation(image):
    if image is None:
        return None
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[-1] == 1:
        image = cv2.merge([image, image, image])

    image = np.uint8(image)

    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (10, 10, image.shape[1]-10, image.shape[0]-10)
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented_image = image * mask2[:, :, np.newaxis]
    segmented_image = np.nan_to_num(segmented_image, nan=0, posinf=255, neginf=0).astype(np.uint8)  # Normalización
    return segmented_image

# Función para detectar keypoints
def detect_keypoints(image):
    gray = convert_to_grayscale(image)
    if gray is None:
        return "None", 0

    # Harris Corner Detection
    harris_corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    harris_corners = cv2.dilate(harris_corners, None)
    harris_keypoints = np.sum(harris_corners > 0.01 * harris_corners.max())

    # SIFT Keypoints Detection
    sift = cv2.SIFT_create()
    sift_keypoints = sift.detect(gray, None)
    sift_keypoints_count = len(sift_keypoints)

    # Determinar mejor método
    if sift_keypoints_count > harris_keypoints:
        return "SIFT", sift_keypoints_count
    else:
        return "Harris", harris_keypoints

# Evaluar imágenes con diferentes técnicas de preprocesamiento
sample_images = df_mareas.sample(13, random_state=42)  # Selección de 13 imágenes
results = []

fig, axes = plt.subplots(len(sample_images), 3, figsize=(15, len(sample_images) * 3))

for i, (_, row) in tqdm(enumerate(sample_images.iterrows()), total=len(sample_images)):
    img_name = row['img_name']
    img_path = os.path.join(image_folder, img_name)

    # Cargar imagen
    image = cv2.imread(img_path)
    if image is None:
        print(f"⚠️ No se encontró la imagen: {img_name}")
        continue

    technique_scores = {}
    for name, func in PREPROCESSING_TECHNIQUES.items():
        preprocessed_img = func(image)
        segmented_img = graph_cut_segmentation(preprocessed_img)
        features = extract_local_features(segmented_img)
        keypoint_method, keypoint_count = detect_keypoints(segmented_img)
        score = np.sum(features) + keypoint_count  # Combinar score con cantidad de keypoints
        technique_scores[name] = (score, keypoint_method, keypoint_count)

    best_technique = max(technique_scores, key=lambda k: technique_scores[k][0])
    best_score, best_keypoint_method, best_keypoint_count = technique_scores[best_technique]

    # Mostrar imágenes
    axes[i, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[i, 0].set_title(f"Original: {img_name}")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(preprocessed_img, cmap="gray")
    axes[i, 1].set_title(f"Preprocesado: {best_technique}")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(segmented_img, cmap="gray")
    axes[i, 2].set_title("Segmentación Graph Cut")
    axes[i, 2].axis("off")

plt.tight_layout()
plt.show()

# Convertir resultados a DataFrame
df_results = pd.DataFrame(results)


# %% [markdown]
# 
# 
# 1.   El código crea un pipeline completo para analizar galaxias con mareas.
# 
# 2.   Graph Cut segmenta la imagen para aislar las estructuras principales.
# 
# 3.   Las técnicas de LBP, HOG y Gabor detectan patrones importantes.
# 
# 
# 4.   Harris y SIFT ayudan a encontrar keypoints relevantes.
# 
# 
# 
# 

# %% [markdown]
# **Preprocesamiento Deficiente**
# 
# 
# 
# 1.   Graph Cut funciona mejor con imágenes bien contrastadas y con bordes claros.
# 
# 2.   El preprocesado actual no enfatiza los bordes de las mareas.
# 
# 3.   La ecualización de histograma y el resize pueden distorsionar la imagen.
# 
# 4.  Las imágenes con ruido afectan la segmentación.
# 
# 
# 
# 

# %%
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Parámetros
TARGET_SIZE = (128, 128)  # Tamaño uniforme para todas las imágenes

# Técnicas de preprocesamiento
def resize_and_normalize(image):
    resized = cv2.resize(image, TARGET_SIZE)
    return (resized * 255).astype(np.uint8)

def apply_opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def graph_cut_segmentation(image):
    if image is None:
        return None
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[-1] == 1:
        image = cv2.merge([image, image, image])

    image = np.uint8(image)
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (10, 10, image.shape[1]-10, image.shape[0]-10)
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented_image = image * mask2[:, :, np.newaxis]
    segmented_image = np.nan_to_num(segmented_image, nan=0, posinf=255, neginf=0).astype(np.uint8)
    return segmented_image

def detect_harris_keypoints(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    if gray is None:
        return "None", 0
    gray = cv2.GaussianBlur(gray, (5, 5), 1)
    gray = np.float32(gray)
    harris_corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    harris_corners = cv2.dilate(harris_corners, None)
    harris_keypoints = np.sum(harris_corners > 0.01 * harris_corners.max())
    return "Harris", harris_keypoints, harris_corners

# Combinaciones a evaluar
combinations = [
    ("Resize -> Opening -> Graph Cut", [resize_and_normalize, apply_opening, graph_cut_segmentation]),
    ("Resize -> Graph Cut -> Opening", [resize_and_normalize, graph_cut_segmentation, apply_opening]),
    ("Opening -> Resize -> Graph Cut", [apply_opening, resize_and_normalize, graph_cut_segmentation]),
    ("Opening -> Graph Cut -> Resize", [apply_opening, graph_cut_segmentation, resize_and_normalize]),
    ("Graph Cut -> Resize -> Opening", [graph_cut_segmentation, resize_and_normalize, apply_opening]),
    ("Graph Cut -> Opening -> Resize", [graph_cut_segmentation, apply_opening, resize_and_normalize]),
]

# Evaluar combinaciones
sample_images = df_mareas.sample(13, random_state=42)
results = []

for combo_name, functions in combinations:
    fig, axes = plt.subplots(len(sample_images), 5, figsize=(24, len(sample_images) * 3))

    for i, (_, row) in tqdm(enumerate(sample_images.iterrows()), total=len(sample_images)):
        img_name = row['img_name']
        img_path = os.path.join(image_folder, img_name)

        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ No se encontró la imagen: {img_name}")
            continue

        processed_img = image.copy()
        for func in functions:
            processed_img = func(processed_img)

        keypoint_method, keypoint_count, harris_corners = detect_harris_keypoints(processed_img)

        # Mostrar imágenes
        axes[i, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"Original: {img_name}")
        axes[i, 0].axis("off")

        for j, func in enumerate(functions):
            step_img = func(image.copy())
            axes[i, j + 1].imshow(step_img, cmap="gray")
            axes[i, j + 1].set_title(func.__name__)
            axes[i, j + 1].axis("off")

        keypoint_vis = processed_img.copy()
        keypoint_vis[harris_corners > 0.01 * harris_corners.max()] = [255, 0, 0]
        axes[i, 4].imshow(keypoint_vis, cmap="gray")
        axes[i, 4].set_title(f"Harris Keypoints: {keypoint_count}")
        axes[i, 4].axis("off")

        results.append({
            "img_name": img_name,
            "combination": combo_name,
            "keypoint_count": keypoint_count
        })

    plt.tight_layout()
    plt.show()

# Convertir resultados a DataFrame
df_results = pd.DataFrame(results)
# Mostrar la mejor combinación
top_combo = df_results.groupby("combination")["keypoint_count"].mean().idxmax()
print(f"Mejor combinación: {top_combo}")


# %% [markdown]
# Este código evalúa diferentes combinaciones de preprocesamiento de imágenes para detectar keypoints en galaxias utilizando Graph Cut y el detector de esquinas de Harris. Se buscan las mejores técnicas para resaltar mareas galácticas en imágenes astronómicas.
# 
# Pasos del Código
# 
# **Definir Preprocesamiento**
# 
# Se aplican varias transformaciones a las imágenes:
# 
# resize_and_normalize: Redimensiona y normaliza la imagen.
# apply_opening: Usa una operación morfológica "Opening" para eliminar ruido.
# 
# graph_cut_segmentation: Aplica Graph Cut para segmentar la imagen.
# detect_harris_keypoints: Detecta keypoints con el detector de esquinas de Harris.
# 
# **Definir Combinaciones de Preprocesamiento**
# 
# Se prueban 6 combinaciones de los métodos anteriores en diferentes órdenes.
# 
# **Aplicar Preprocesamiento y Evaluación**
# 
# Se carga cada imagen.
# 
# Se aplica cada combinación de funciones en la imagen.
# Se detectan keypoints con Harris y se cuentan los detectados.
# 
# Se muestra la imagen en cada paso del preprocesamiento.
# 
# **Visualización y Resultados**
# 
# Se generan imágenes con cada paso de la transformación.
# 
# Se pintan los keypoints en rojo.
# 
# Se almacena el número de keypoints detectados en un DataFrame (df_results).
# 
# Se identifica la mejor combinación con el mayor número de keypoints detectados.
# 

# %% [markdown]
# **Enviar a Mongo Atlas los vectores**

# %%
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pymongo import MongoClient



# Definir tamaño de imagen
TARGET_SIZE = (128, 128)

# Función para aplicar la operación morfológica "opening"
def apply_opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# Función para redimensionar y normalizar imagen
def resize_and_normalize(image):
    resized = cv2.resize(image, TARGET_SIZE)
    return (resized * 255).astype(np.uint8)

# Función para aplicar Graph Cut
def graph_cut_segmentation(image):
    if image is None:
        return None
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[-1] == 1:
        image = cv2.merge([image, image, image])

    image = np.uint8(image)
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (10, 10, image.shape[1]-10, image.shape[0]-10)
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented_image = image * mask2[:, :, np.newaxis]
    segmented_image = np.nan_to_num(segmented_image, nan=0, posinf=255, neginf=0).astype(np.uint8)
    return segmented_image

# Función para detectar keypoints con Harris
def detect_harris_keypoints(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    if gray is None:
        return "None", 0
    gray = cv2.GaussianBlur(gray, (5, 5), 1)
    gray = np.float32(gray)
    harris_corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    harris_corners = cv2.dilate(harris_corners, None)
    keypoints = np.argwhere(harris_corners > 0.01 * harris_corners.max())
    return "Harris", len(keypoints), keypoints

# Función para convertir imagen en vector
def image_to_vector(image):
    return image.flatten().tolist()

# Función para procesar imágenes y generar vectores
def process_and_vectorize(df):
    vectors = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_name = row['img_name']
        img_path = os.path.join(image_folder, img_name)

        # Cargar imagen
        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ No se encontró la imagen: {img_name}")
            continue

        # Aplicar preprocesamiento en el orden Opening -> Resize -> Graph Cut
        img_opening = apply_opening(image)
        img_resized = resize_and_normalize(img_opening)
        img_segmented = graph_cut_segmentation(img_resized)

        # Detectar keypoints con Harris
        keypoint_method, keypoints_harris, keypoints = detect_harris_keypoints(img_segmented)

        # Convertir en vector
        img_vector = image_to_vector(img_segmented)

        # Agregar datos al nuevo DataFrame
        row_data = row.to_dict()
        row_data["vector"] = img_vector
        row_data["harris_keypoints"] = keypoints_harris
        vectors.append(row_data)

    return pd.DataFrame(vectors)

# Procesar imágenes de galaxias con mareas
df_mareas_vector_f = process_and_vectorize(df_mareas)
df_mareas_vector_f.to_csv("/content/mareas_vectors_f.csv", index=False)
print("📂 df_mareas_vector_f guardado correctamente.")

# Procesar imágenes de galaxias sin mareas
df_no_mareas_vector_f = process_and_vectorize(df_no_mareas)
df_no_mareas_vector_f.to_csv("/content/no_mareas_vectors_f.csv", index=False)
print("📂 df_no_mareas_vector_f guardado correctamente.")


# %% [markdown]
# 100%|██████████| 46/46 [00:05<00:00,  7.83it/s]
# 📂 df_mareas_vector_f guardado correctamente.
# 100%|██████████| 198/198 [00:23<00:00,  8.43it/s]
# 📂 df_no_mareas_vector_f guardado correctamente.
# 

# %% [markdown]
# Carga de Imágenes:
# 
# Se leen imágenes desde un directorio especificado en df_mareas y df_no_mareas.
# 
# Preprocesamiento de Imágenes:
# 
# Opening: Reduce ruido en la imagen.
# 
# Resize & Normalize: Escala la imagen a (128, 128).
# 
# Graph Cut: Segmenta la imagen en fondo y objeto de interés.
# Extracción de Características:
# 
# Harris Keypoints: Detecta puntos de interés.
# 
# Conversión a Vectores: Se aplana la imagen y se almacena como un vector.
# 
# Almacenamiento:
# 
# Se guardan los vectores y keypoints en archivos CSV (mareas_vectors_f.csv y no_mareas_vectors_f.csv).

# %% [markdown]
# # ¿Qué es un Grafo?
# 
# Un grafo es una estructura de datos que representa relaciones entre objetos. Se compone de:
# 
# * **Nodos (vértices)**: Representan entidades (en este caso, imágenes de galaxias).
# 
# * **Aristas (conexiones)**: Representan relaciones entre los nodos (en este caso, similitud entre las imágenes con base en sus keypoints de Harris).

# %% [markdown]
# # ¿Por qué usar un Grafo en este Proyecto?
# 
# En este proyecto, el objetivo es comprender la estructura de las galaxias y detectar patrones en imágenes de mareas gravitacionales.
# 
# El uso de un grafo permite:
# 
# 1. Modelar Relaciones de Similitud:
#    Conectamos imágenes de galaxias si tienen un número similar de **keypoints de Harris**.
# 
#    Esto permite **identificar patrones comune**s entre galaxias con mareas y sin mareas.
# 
# 2. Visualizar la Conectividad entre Imágenes:
# 
#    Podemos ver cómo se agrupan las imágenes basadas en la **cantidad de keypoints**.
# 
#    **Un grafo bien conectado sugiere similitud en estructuras entre galaxias**.
#    
# 3. Explorar la Separabilidad de las Clases:
# 
#    Si las galaxias con mareas están conectadas entre sí pero no con las sin mareas, esto sugiere que los **keypoints pueden ser un buen criterio de clasificación**.
# 

# %%
pip install "pymongo[srv]"

# %%
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pymongo import MongoClient

# 🔹 Conectar a MongoDB Atlas
MONGO_URI = "mongodb+srv://estebanji85:Ykgh58Bs45fZIFva@universo.wcqix.mongodb.net/?retryWrites=true&w=majority&appName=Universo"

client = MongoClient(MONGO_URI)
db = client["galaxias_db"]

# 🔹 Obtener datos de MongoDB
collection_mareas_final = db["mareas_vectors_f"]
collection_no_mareas_final = db["no_mareas_vectors_f"]

# Obtener documentos
mareas_docs = list(collection_mareas_final.find().limit(10))
no_mareas_docs = list(collection_no_mareas_final.find().limit(10))

# 🔹 Extraer keypoints y buscar coincidencias
def extract_harris_keypoints(docs):
    keypoints_sets = []
    for doc in docs:
        keypoints = doc.get("harris_keypoints", 0)  # Obtener keypoints
        keypoints_sets.append(keypoints)
    return keypoints_sets

# Obtener keypoints de cada grupo
mareas_harris = extract_harris_keypoints(mareas_docs)
no_mareas_harris = extract_harris_keypoints(no_mareas_docs)

# 🔹 Construcción del Grafo
G = nx.Graph()

# Agregar nodos según número de keypoints
for i, harris_value in enumerate(mareas_harris):
    G.add_node(f"Marea_{i}", size=harris_value, color="blue")

for i, harris_value in enumerate(no_mareas_harris):
    G.add_node(f"NoMarea_{i}", size=harris_value, color="green")

# Crear conexiones basadas en similitud
for i in range(len(mareas_harris)):
    for j in range(len(no_mareas_harris)):
        if abs(mareas_harris[i] - no_mareas_harris[j]) < 1000:  # Umbral de similitud
            G.add_edge(f"Marea_{i}", f"NoMarea_{j}")

# Dibujar el grafo
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)
node_colors = [G.nodes[node]['color'] for node in G.nodes]
nx.draw(G, pos, with_labels=True, node_size=500, node_color=node_colors, edge_color="gray")
plt.title("Grafo de Coincidencias entre Harris Keypoints")
plt.show()


# %% [markdown]
# **Evaluación del Preprocesamiento y Graph Cut**
# 
# 
# 1. Aspectos Positivos:
# 
#    Se logró extraer keypoints estructurales con el método de Harris después de aplicar **Opening → Resize → Graph Cut.**
# 
#    **Se encontraron agrupaciones entre imágenes con similitud en keypoint**s.
# 
#   ** Se observan subgrupos bien definidos**, lo que indica que algunas imágenes comparten características comunes.
# 
# 2. Aspectos Problemáticos:
# 
#    **Existen conexiones entre imágenes de galaxias con y sin mareas**, lo que sugiere que los keypoints no separan completamente ambas clases.
# 
#    **Algunas galaxias sin mareas están dentro del mismo grupo de galaxias con mareas**, lo que indica falsos positivos en la clasificación.
# 
#   ** Graph Cut segmenta estructuras brillantes en la imagen, pero podría no distinguir completamente las características relevantes de las mareas.**

# %%
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pymongo import MongoClient
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity

# 🔹 Conectar a MongoDB Atlas
MONGO_URI = "mongodb+srv://estebanji85:Ykgh58Bs45fZIFva@universo.wcqix.mongodb.net/?retryWrites=true&w=majority&appName=Universo"


client = MongoClient(MONGO_URI)
db = client["galaxias_db"]

# 🔹 Obtener datos de MongoDB
collection_mareas_final = db["mareas_vectors_f"]
collection_no_mareas_final = db["no_mareas_vectors_f"]

# Obtener múltiples documentos de cada categoría
mareas_docs = list(collection_mareas_final.find().limit(10))
no_mareas_docs = list(collection_no_mareas_final.find().limit(10))

# 🔹 Construcción de Grafos para vectores
G_mareas = nx.Graph()
G_no_mareas = nx.Graph()

# Función para agregar nodos y conexiones por similitud
def build_vector_graph(G, docs, color):
    vectors = []
    for i, doc in enumerate(docs):
        vector = np.array(doc.get("vector", []))
        if vector.size > 0:
            G.add_node(f"Imagen_{i}", vector=vector, color=color)
            vectors.append(vector)

    # Calcular similitud de coseno entre los vectores
    if len(vectors) > 1:
        similarities = cosine_similarity(vectors)
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                if similarities[i, j] > 0.8:  # Umbral de similitud
                    G.add_edge(f"Imagen_{i}", f"Imagen_{j}")

# Construir grafos para ambos conjuntos
build_vector_graph(G_mareas, mareas_docs, "red")
build_vector_graph(G_no_mareas, no_mareas_docs, "blue")

# 🔹 Visualización
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Grafo de imágenes con marea
axes[0].set_title("Grafo de Similitud - Galaxias con Marea")
pos_mareas = nx.spring_layout(G_mareas, seed=42)
nx.draw(G_mareas, pos=pos_mareas, with_labels=True, node_size=500, node_color=[G_mareas.nodes[n]['color'] for n in G_mareas.nodes], edge_color="gray", ax=axes[0])

# Grafo de imágenes sin marea
axes[1].set_title("Grafo de Similitud - Galaxias sin Marea")
pos_no_mareas = nx.spring_layout(G_no_mareas, seed=42)
nx.draw(G_no_mareas, pos=pos_no_mareas, with_labels=True, node_size=500, node_color=[G_no_mareas.nodes[n]['color'] for n in G_no_mareas.nodes], edge_color="gray", ax=axes[1])

plt.show()

# 🔹 Análisis de similitud entre vectores
mareas_vectors = [np.array(doc.get("vector", [])) for doc in mareas_docs if len(doc.get("vector", [])) > 0]
no_mareas_vectors = [np.array(doc.get("vector", [])) for doc in no_mareas_docs if len(doc.get("vector", [])) > 0]

# Calcular matrices de similitud
similarity_matrix_mareas = cosine_similarity(mareas_vectors)
similarity_matrix_no_mareas = cosine_similarity(no_mareas_vectors)

# Visualización de la similitud con mapas de calor
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

im1 = axes[0].imshow(similarity_matrix_mareas, cmap="coolwarm", interpolation="nearest")
axes[0].set_title("Similitud entre Galaxias con Marea")
fig.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(similarity_matrix_no_mareas, cmap="coolwarm", interpolation="nearest")
axes[1].set_title("Similitud entre Galaxias sin Marea")
fig.colorbar(im2, ax=axes[1])

plt.show()



# %% [markdown]
# Análisis de Resultados
# 
# 1. Mapa de la Izquierda (Galaxias con Mareas)
# 
#    Se observan agrupaciones rojizas en algunas zonas, lo que indica que ciertas imágenes de mareas tienen una estructura similar. **También hay zonas azules, lo que significa que algunas galaxias con mareas tienen estructuras completamente diferentes**.
# 
# 2. Mapa de la Derecha (Galaxias sin Mareas)
# 
#   Similar al caso anterior, pero aquí las **similitudes son menos marcadas**. La gran franja azul indica que hay imágenes con estructuras muy diferentes dentro de este grupo.
# 
# 
# 3. Este gráfico muestra dos grafos separados:
# 
#    Izquierda (Rojo): Galaxias con mareas.
# 
#    Derecha (Azul): Galaxias sin mareas.
# 
#   Cada nodo representa una imagen y la distancia entre nodos refleja la similitud entre sus vectores de características. En este caso, **no hay conexiones visibles**, lo que sugiere que las galaxias dentro de cada grupo son muy diferentes entre sí.
# 
# 
#   El hecho de que no haya conexiones entre imágenes dentro de un mismo grupo indica baja similitud interna en cada categoría.
#   Esto puede deberse a: **Las galaxias con mareas pueden tener formas muy diferentes y estructuras caóticas**.
# 
#   Las galaxias sin mareas pueden variar en brillo y tamaño sin una característica unificadora.
# 
#   **El preprocesamiento puede haber fallado en capturar patrones comunes, causando baja similitud entre imágenes.**

# %% [markdown]
#  **Resumen del Código de Clasificación de Galaxias**
# 
#  Este código implementa un sistema de clasificación de imágenes de galaxias con y sin mareas utilizando modelos tradicionales de Machine Learning y modelos de Deep Learning. A continuación, se desglosan sus principales componentes:
# 
# 

# %% [markdown]
# Carga de Datos desde MongoDB Atlas
# 
# * Se conecta a MongoDB Atlas y obtiene los vectores de características de imágenes de galaxias con y sin mareas.
# 
# * Se extraen los datos y se construyen los arreglos X (vectores de características) y y (clases: 1 = con mareas, 0 = sin mareas).

# %% [markdown]
# Entrenamiento de Modelos Tradicionales
# 
# Se utilizan tres modelos clásicos de Machine Learning para la clasificación basada en vectores:
# 
# * **SVM (Support Vector Machine)**
# 
# * **Random Forest**
# 
# * **MLP (Multilayer Perceptron - Red Neuronal)**
# 
# Los datos son normalizados usando StandardScaler.
# Se dividen en entrenamiento (80%) y prueba (20%).
# Cada modelo se entrena y se guarda en un archivo .pkl.

# %% [markdown]
# Modelos de Visión Profunda (Deep Learning)
# 
# Se utilizan tres modelos preentrenados para clasificación de imágenes:
# 
# * **Vision Transformer (ViT)**
# 
# * **ResNet-18**
# 
# * **VGG16**
# 
# Se cargan modelos con pesos preentrenados en ImageNet.
# 
# Se modifica la última capa de cada modelo para clasificar en 2 categorías (con/sin mareas).
# 
# Se mueven los modelos a GPU.

# %% [markdown]
# Preprocesamiento de Imágenes
# 
# Antes de la predicción, cada imagen se somete a los siguientes pasos:
# 
# 1. Conversión a RGB (para mantener consistencia en modelos).
# 
# 2. Mejora del contraste usando EqualizeHist en LAB color space.
# 
# 3. Reducción de ruido con fastNlMeansDenoisingColored().
# Redimensionamiento a 224x224 píxeles.
# 
# 4. Guardado de la imagen procesada.

# %% [markdown]
# Predicción con Modelos de Deep Learning
# 
# * La imagen preprocesada se convierte en tensor para su evaluación en modelos de visión.
# 
# * Cada modelo genera una predicción (0 = Sin Mareas, 1 = Con Mareas) y su probabilidad de confianza.
# 
# * El modelo con mayor confianza es seleccionado como el mejor.

# %% [markdown]
# Evaluación de los Modelos
# 
# Para los modelos tradicionales, se calculan:
# 
#  *  Accuracy
# 
#  *  ROC AUC Score
# 
#  * Precision, Recall y F1-Score (usando classification_report).
# 
# Para los modelos de visión:
# 
#  * Se comparan las probabilidades de los modelos (ViT, ResNet y VGG) y se selecciona el que mayor confianza tenga.

# %% [markdown]
# **Predicción Final**
# 
# * Se ejecuta la función predict_galaxy(image_path), que evalúa una imagen de galaxia y determina si tiene mareas o no.
# 
# * Se imprime la predicción con la mejor confianza entre los modelos utilizados.

# %%
import os
import pymongo
import cv2
import numpy as np
import joblib
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# 📌 **1. Conectar a MongoDB Atlas**
client = pymongo.MongoClient("mongodb+srv://estebanji85:Ykgh58Bs45fZIFva@universo.wcqix.mongodb.net/?retryWrites=true&w=majority&appName=Universo")
db = client["galaxias_db"]
mareas_collection = db["mareas_vectors"]
no_mareas_collection = db["no_mareas_vectors"]

# 📌 **2. Extraer datos desde MongoDB**
def load_vectors_from_mongodb():
    mareas = list(mareas_collection.find({}, {"vector": 1, "class": 1, "_id": 0}))
    no_mareas = list(no_mareas_collection.find({}, {"vector": 1, "class": 1, "_id": 0}))

    data = mareas + no_mareas
    X = np.array([entry["vector"] for entry in data])
    y = np.array([entry["class"] for entry in data])

    return X, y

X, y = load_vectors_from_mongodb()

# 📌 **3. Normalizar datos y dividir en entrenamiento/test**
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 **4. Modelos tradicionales**
models_dict = {
    "SVM": SVC(probability=True),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
}

for name, model in models_dict.items():
    print(f"Entrenando modelo: {name}")
    model.fit(X_train, y_train)
    joblib.dump(model, f"{name}_model.pkl")  # Guardar modelo

joblib.dump(scaler, "scaler.pkl")  # Guardar normalizador

# 📌 **5. Cargar modelos de visión**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vit_model = models.vision_transformer.vit_b_16(weights="IMAGENET1K_V1")
vit_model.heads.head = nn.Linear(vit_model.heads.head.in_features, 2)
vit_model.to(device)

resnet_model = models.resnet18(weights="IMAGENET1K_V1")
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 2)
resnet_model.to(device)

vgg_model = models.vgg16(weights="IMAGENET1K_V1")
vgg_model.classifier[6] = nn.Linear(4096, 2)
vgg_model.to(device)

# 📌 **6. Transformación para modelos de visión**
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 📌 **7. Función para preprocesar imágenes**
def preprocess_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"⚠️ No se pudo cargar la imagen en {image_path}")

    # 📌 **1. Convertir a RGB**
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 📌 **2. Mejorar contraste**
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # 📌 **3. Reducción de ruido**
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # 📌 **4. Resize**
    image = cv2.resize(image, (224, 224))

    # Guardar imagen procesada
    processed_img_path = f"/content/{os.path.basename(image_path)}_processed.png"
    cv2.imwrite(processed_img_path, image)

    return processed_img_path

# 📌 **8. Función para predecir una imagen**
def predict_galaxy(image_path):
    img_name = image_path.split("/")[-1]

    # 📌 **Preprocesar la imagen**
    processed_img_path = preprocess_image(image_path)

    # 📌 **Transformar imagen**
    image_tensor = transform(Image.open(processed_img_path)).unsqueeze(0).to(device)

    # 📌 **Predicción con modelos de visión**
    vit_model.eval()
    vit_output = vit_model(image_tensor)
    vit_prediction = int(torch.argmax(vit_output, axis=1).item())
    vit_proba = float(torch.nn.functional.softmax(vit_output, dim=1).max().item())

    resnet_model.eval()
    resnet_output = resnet_model(image_tensor)
    resnet_prediction = int(torch.argmax(resnet_output, axis=1).item())
    resnet_proba = float(torch.nn.functional.softmax(resnet_output, dim=1).max().item())

    vgg_model.eval()
    vgg_output = vgg_model(image_tensor)
    vgg_prediction = int(torch.argmax(vgg_output, axis=1).item())
    vgg_proba = float(torch.nn.functional.softmax(vgg_output, dim=1).max().item())

    # 📌 **Seleccionar mejor modelo**
    predictions = {
        "ViT": (vit_prediction, vit_proba),
        "ResNet": (resnet_prediction, resnet_proba),
        "VGG": (vgg_prediction, vgg_proba),
    }
    best_model_name = max(predictions, key=lambda x: predictions[x][1])
    final_prediction, confidence = predictions[best_model_name]

    final_label = "Galaxia con Mareas" if final_prediction == 1 else "Galaxia sin Mareas"
    print(f"🔍 Mejor modelo: {best_model_name} - Predicción final: {final_label} con {confidence*100:.2f}% de confianza")

    return final_label, confidence

# 📌 **9. Evaluación del Modelo**
for name, model in models_dict.items():
    y_pred = model.predict(X_test)
    print(f"\n📊 **Evaluación de {name}:**")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred):.4f}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# 📌 **10. Realizar Predicción**
image_path = "/content/drive/MyDrive/CNN/mareatest2.jpg"
predict_galaxy(image_path)


# %% [markdown]
# 

# %% [markdown]
# # **Codigo de la Solucion, preprocesamiento & CNN**

# %%
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from google.colab import drive

# Montar Google Drive
drive.mount('/content/drive')

# Ruta del CSV y carpeta de imágenes
csv_path = '/content/drive/MyDrive/CNN/lsb_data/Atkinson_Walmsley_class_validation.csv'
image_folder = '/content/drive/MyDrive/CNN/lsb_data/imagenes'
output_folder = '/content/lsb_data/imagenes_procesadas'

# Crear carpeta para imágenes procesadas si no existe
os.makedirs(output_folder, exist_ok=True)

# Cargar CSV
df = pd.read_csv(csv_path)

# Obtener lista de imágenes en la carpeta
image_files = os.listdir(image_folder)
image_files_lower = [img.lower() for img in image_files]  # Convertir a minúsculas para comparación

# Función para cargar y procesar imágenes
def load_image(image_name):
    image_name_lower = image_name.lower()
    if image_name_lower in image_files_lower:
        original_name = image_files[image_files_lower.index(image_name_lower)]  # Nombre original con mayúsculas
        image_path = os.path.join(image_folder, original_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, original_name
    else:
        return None, None

# Listas para almacenar rutas de imágenes procesadas y verificación
image_data = []
verification_results = []

# Procesar imágenes y guardarlas
for _, row in tqdm(df.iterrows(), total=df.shape[0]):
    image_name = row['img_name']
    image, original_name = load_image(image_name)

    if image is not None:
        output_image_name = f"{os.path.splitext(original_name)[0]}_procesada.png"
        output_image_path = os.path.join(output_folder, output_image_name)
        cv2.imwrite(output_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        image_data.append(output_image_path)

        # Verificación: Comprobar que la imagen guardada corresponde con la original
        verification_results.append(output_image_path.endswith(image_name.replace(".jpg", "_procesada.png")))
    else:
        image_data.append(None)
        verification_results.append(False)  # Si no se encuentra la imagen, la verificación es falsa

# Agregar las columnas al DataFrame
df['imagen_procesada'] = image_data
df['verificacion_correcta'] = verification_results  # Nueva columna con la verificación



# Verificar imágenes guardadas
guardadas = os.listdir(output_folder)
print(f"Imágenes procesadas guardadas: {len(guardadas)}")

# Función para visualizar imágenes guardadas
def ver_imagen_guardada(image_path):
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.title(os.path.basename(image_path))
        plt.axis('off')
        plt.show()
    else:
        print(f"No se encontró la imagen: {image_path}")

# Mostrar las primeras 3 imágenes guardadas
for image_name in guardadas[:3]:
    ver_imagen_guardada(os.path.join(output_folder, image_name))

# Mostrar la verificación
print("Resumen de la verificación:")
print(df[['img_name', 'imagen_procesada', 'verificacion_correcta']].head(10))


# %%
df

# %%
df.shape

# %%
df.to_csv('lsb_data/mareas_galaxy.csv', index=False)

# %%
# Filtrar clases
df_mareas = df[df['class'] == 1]  # Mareas
df_no_mareas = df[df['class'] == 0]  # Sin mareas

# Seleccionar muestras balanceadas
sample_size = min(len(df_mareas), len(df_no_mareas))  # Igualar tamaño de clases
sample_mareas = df_mareas.sample(n=sample_size, random_state=42)
sample_no_mareas = df_no_mareas.sample(n=sample_size, random_state=42)


# %%
df_mareas.shape

# %%
# Seleccionar 5 imágenes de df_mareas
sample_images = df_mareas.sample(5, random_state=42)  # Toma 5 imágenes aleatorias

# Crear una figura para visualizar
fig, axes = plt.subplots(1, 5, figsize=(15, 5))

for i, row in enumerate(sample_images.iterrows()):
    img_name = row[1]['img_name']
    img_path = os.path.join(image_folder, img_name)

    # Cargar imagen
    image = cv2.imread(img_path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB
        axes[i].imshow(image)
        axes[i].set_title(f"{img_name}")
        axes[i].axis("off")  # Ocultar ejes
    else:
        print(f"⚠️ No se encontró la imagen: {img_name}")

plt.show()

# %%
# Seleccionar 5 imágenes de df_mareas
sample_images = df_no_mareas.sample(5, random_state=42)  # Toma 5 imágenes aleatorias

# Crear una figura para visualizar
fig, axes = plt.subplots(1, 5, figsize=(15, 5))

for i, row in enumerate(sample_images.iterrows()):
    img_name = row[1]['img_name']
    img_path = os.path.join(image_folder, img_name)

    # Cargar imagen
    image = cv2.imread(img_path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB
        axes[i].imshow(image)
        axes[i].set_title(f"{img_name}")
        axes[i].axis("off")  # Ocultar ejes
    else:
        print(f"⚠️ No se encontró la imagen: {img_name}")

plt.show()

# %%
df_mareas.shape

# %%
df_no_mareas.shape

# %%
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Parámetros
TARGET_SIZE = (128, 128)  # Tamaño uniforme para todas las imágenes

# Técnicas de preprocesamiento
def resize_and_normalize(image):
    resized = cv2.resize(image, TARGET_SIZE)
    return resized / 255.0

def convert_to_grayscale(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(grayscale, TARGET_SIZE)

def histogram_equalization(image):
    grayscale = convert_to_grayscale(image)
    return cv2.equalizeHist(grayscale)

def gaussian_blur(image):
    return cv2.GaussianBlur(cv2.resize(image, TARGET_SIZE), (5, 5), 0)

PREPROCESSING_TECHNIQUES = {
    "resize_normalize": resize_and_normalize,
    "grayscale": convert_to_grayscale,
    "histogram_equalization": histogram_equalization,
    "gaussian_blur": gaussian_blur
}

# Definir kernels para operaciones morfológicas
kernel = np.ones((5, 5), np.uint8)

# Función para aplicar operaciones morfológicas
def apply_morphology(image):
    morph_operations = {
        "erosion": cv2.erode(image, kernel, iterations=1),
        "dilation": cv2.dilate(image, kernel, iterations=1),
        "opening": cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel),
        "closing": cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    }
    best_morphology = max(morph_operations, key=lambda x: np.var(morph_operations[x]))
    return morph_operations[best_morphology], best_morphology

# Función para evaluar detección de keypoints
def detect_keypoints(image):
    if len(image.shape) == 3:  # Si la imagen tiene 3 canales (RGB/BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:  # Si la imagen ya está en escala de grises
        gray = image

    # Harris Corner Detection
    harris_corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    harris_corners = cv2.dilate(harris_corners, None)
    harris_keypoints = np.sum(harris_corners > 0.01 * harris_corners.max())

    # SIFT Keypoints Detection
    sift = cv2.SIFT_create()
    sift_keypoints = sift.detect(gray, None)
    sift_keypoints_count = len(sift_keypoints)

    # Determinar mejor método
    if sift_keypoints_count > harris_keypoints:
        best_keypoint = "SIFT"
        keypoint_count = sift_keypoints_count
    else:
        best_keypoint = "Harris"
        keypoint_count = harris_keypoints

    return best_keypoint, keypoint_count

# Función para seleccionar el mejor preprocesamiento
def best_preprocessing(image):
    processed_images = {name: func(image) for name, func in PREPROCESSING_TECHNIQUES.items()}
    best_preprocess = max(processed_images, key=lambda x: np.var(processed_images[x]))
    return processed_images[best_preprocess], best_preprocess

# Evaluar imágenes
sample_images = df_mareas.sample(13, random_state=42)  # Selección de 13 imágenes
results = []

fig, axes = plt.subplots(len(sample_images), 4, figsize=(15, len(sample_images) * 3))

for i, (_, row) in tqdm(enumerate(sample_images.iterrows()), total=len(sample_images)):
    img_name = row['img_name']
    img_path = os.path.join(image_folder, img_name)

    # Cargar imagen
    image = cv2.imread(img_path)
    if image is None:
        print(f"⚠️ No se encontró la imagen: {img_name}")
        continue

    # Aplicar el mejor preprocesamiento
    best_preprocess_img, best_preprocess = best_preprocessing(image)

    # Aplicar la mejor operación morfológica
    best_morph_img, best_morphology = apply_morphology(best_preprocess_img)

    # Detectar keypoints
    best_keypoint, keypoint_count = detect_keypoints(best_morph_img)

    # Mostrar imágenes
    axes[i, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[i, 0].set_title(f"Original: {img_name}")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(best_preprocess_img if len(best_preprocess_img.shape) == 2 else cv2.cvtColor(best_preprocess_img, cv2.COLOR_BGR2RGB), cmap="gray")
    axes[i, 1].set_title(f"Mejor Preprocesado: {best_preprocess}")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(best_morph_img if len(best_morph_img.shape) == 2 else cv2.cvtColor(best_morph_img, cv2.COLOR_BGR2RGB), cmap="gray")
    axes[i, 2].set_title(f"Mejor Morfología: {best_morphology}")
    axes[i, 2].axis("off")

    # Dibujar keypoints en la imagen
    keypoint_image = cv2.cvtColor(best_morph_img, cv2.COLOR_GRAY2BGR) if len(best_morph_img.shape) == 2 else best_morph_img.copy()
    if best_keypoint == "SIFT":
        sift = cv2.SIFT_create()
        kp = sift.detect(cv2.cvtColor(keypoint_image, cv2.COLOR_BGR2GRAY), None)
        keypoint_image = cv2.drawKeypoints(keypoint_image, kp, None)
    else:  # Harris
        harris_corners = cv2.cornerHarris(cv2.cvtColor(keypoint_image, cv2.COLOR_BGR2GRAY), 2, 3, 0.04)
        keypoint_image[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]

    axes[i, 3].imshow(keypoint_image if len(keypoint_image.shape) == 3 else cv2.cvtColor(keypoint_image, cv2.COLOR_BGR2RGB), cmap="gray")
    axes[i, 3].set_title(f"Keypoint: {best_keypoint} ({keypoint_count})")
    axes[i, 3].axis("off")

    # Guardar resultados
    results.append({
        "img_name": img_name,
        "best_preprocess": best_preprocess,
        "best_morphology": best_morphology,
        "best_keypoint": best_keypoint,
        "keypoint_count": keypoint_count
    })

plt.tight_layout()
plt.show()

# Convertir resultados a DataFrame y guardarlos
df_results = pd.DataFrame(results)



# %%
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Parámetros
TARGET_SIZE = (128, 128)  # Tamaño uniforme para todas las imágenes

# Técnicas de preprocesamiento
def resize_and_normalize(image):
    resized = cv2.resize(image, TARGET_SIZE)
    return resized / 255.0

def convert_to_grayscale(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(grayscale, TARGET_SIZE)

def histogram_equalization(image):
    grayscale = convert_to_grayscale(image)
    return cv2.equalizeHist(grayscale)

def gaussian_blur(image):
    return cv2.GaussianBlur(cv2.resize(image, TARGET_SIZE), (5, 5), 0)

PREPROCESSING_TECHNIQUES = {
    "resize_normalize": resize_and_normalize,
    "grayscale": convert_to_grayscale,
    "histogram_equalization": histogram_equalization,
    "gaussian_blur": gaussian_blur
}

# Definir kernels para operaciones morfológicas
kernel = np.ones((5, 5), np.uint8)

# Función para aplicar operaciones morfológicas
def apply_morphology(image):
    morph_operations = {
        "erosion": cv2.erode(image, kernel, iterations=1),
        "dilation": cv2.dilate(image, kernel, iterations=1),
        "opening": cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel),
        "closing": cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    }
    best_morphology = max(morph_operations, key=lambda x: np.var(morph_operations[x]))
    return morph_operations[best_morphology], best_morphology

# Función para evaluar detección de keypoints
def detect_keypoints(image):
    if len(image.shape) == 3:  # Si la imagen tiene 3 canales (RGB/BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:  # Si la imagen ya está en escala de grises
        gray = image

    # Harris Corner Detection
    harris_corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    harris_corners = cv2.dilate(harris_corners, None)
    harris_keypoints = np.sum(harris_corners > 0.01 * harris_corners.max())

    # SIFT Keypoints Detection
    sift = cv2.SIFT_create()
    sift_keypoints = sift.detect(gray, None)
    sift_keypoints_count = len(sift_keypoints)

    # Determinar mejor método
    if sift_keypoints_count > harris_keypoints:
        best_keypoint = "SIFT"
        keypoint_count = sift_keypoints_count
    else:
        best_keypoint = "Harris"
        keypoint_count = harris_keypoints

    return best_keypoint, keypoint_count

# Función para seleccionar el mejor preprocesamiento
def best_preprocessing(image):
    processed_images = {name: func(image) for name, func in PREPROCESSING_TECHNIQUES.items()}
    best_preprocess = max(processed_images, key=lambda x: np.var(processed_images[x]))
    return processed_images[best_preprocess], best_preprocess

# Evaluar imágenes
sample_images = df_no_mareas.sample(60, random_state=42)  # Selección de 13 imágenes
results = []

fig, axes = plt.subplots(len(sample_images), 4, figsize=(15, len(sample_images) * 3))

for i, (_, row) in tqdm(enumerate(sample_images.iterrows()), total=len(sample_images)):
    img_name = row['img_name']
    img_path = os.path.join(image_folder, img_name)

    # Cargar imagen
    image = cv2.imread(img_path)
    if image is None:
        print(f"⚠️ No se encontró la imagen: {img_name}")
        continue

    # Aplicar el mejor preprocesamiento
    best_preprocess_img, best_preprocess = best_preprocessing(image)

    # Aplicar la mejor operación morfológica
    best_morph_img, best_morphology = apply_morphology(best_preprocess_img)

    # Detectar keypoints
    best_keypoint, keypoint_count = detect_keypoints(best_morph_img)

    # Mostrar imágenes
    axes[i, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[i, 0].set_title(f"Original: {img_name}")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(best_preprocess_img if len(best_preprocess_img.shape) == 2 else cv2.cvtColor(best_preprocess_img, cv2.COLOR_BGR2RGB), cmap="gray")
    axes[i, 1].set_title(f"Mejor Preprocesado: {best_preprocess}")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(best_morph_img if len(best_morph_img.shape) == 2 else cv2.cvtColor(best_morph_img, cv2.COLOR_BGR2RGB), cmap="gray")
    axes[i, 2].set_title(f"Mejor Morfología: {best_morphology}")
    axes[i, 2].axis("off")

    # Dibujar keypoints en la imagen
    keypoint_image = cv2.cvtColor(best_morph_img, cv2.COLOR_GRAY2BGR) if len(best_morph_img.shape) == 2 else best_morph_img.copy()
    if best_keypoint == "SIFT":
        sift = cv2.SIFT_create()
        kp = sift.detect(cv2.cvtColor(keypoint_image, cv2.COLOR_BGR2GRAY), None)
        keypoint_image = cv2.drawKeypoints(keypoint_image, kp, None)
    else:  # Harris
        harris_corners = cv2.cornerHarris(cv2.cvtColor(keypoint_image, cv2.COLOR_BGR2GRAY), 2, 3, 0.04)
        keypoint_image[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]

    axes[i, 3].imshow(keypoint_image if len(keypoint_image.shape) == 3 else cv2.cvtColor(keypoint_image, cv2.COLOR_BGR2RGB), cmap="gray")
    axes[i, 3].set_title(f"Keypoint: {best_keypoint} ({keypoint_count})")
    axes[i, 3].axis("off")

    # Guardar resultados
    results.append({
        "img_name": img_name,
        "best_preprocess": best_preprocess,
        "best_morphology": best_morphology,
        "best_keypoint": best_keypoint,
        "keypoint_count": keypoint_count
    })

plt.tight_layout()
plt.show()

# Convertir resultados a DataFrame y guardarlos
df_results_nomarea = pd.DataFrame(results)



# %%
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Parámetros
TARGET_SIZE = (128, 128)  # Tamaño uniforme para todas las imágenes

# Técnicas de preprocesamiento
def resize_and_normalize(image):
    resized = cv2.resize(image, TARGET_SIZE)
    return resized / 255.0

def convert_to_grayscale(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(grayscale, TARGET_SIZE)

def histogram_equalization(image):
    grayscale = convert_to_grayscale(image)
    return cv2.equalizeHist(grayscale)

def gaussian_blur(image):
    return cv2.GaussianBlur(cv2.resize(image, TARGET_SIZE), (5, 5), 0)

PREPROCESSING_TECHNIQUES = {
    "resize_normalize": resize_and_normalize,
    "grayscale": convert_to_grayscale,
    "histogram_equalization": histogram_equalization,
    "gaussian_blur": gaussian_blur
}

# Definir kernels para operaciones morfológicas
kernel = np.ones((5, 5), np.uint8)

# Función para aplicar operaciones morfológicas
def apply_morphology(image):
    morph_operations = {
        "erosion": cv2.erode(image, kernel, iterations=1),
        "dilation": cv2.dilate(image, kernel, iterations=1),
        "opening": cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel),
        "closing": cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    }
    best_morphology = max(morph_operations, key=lambda x: np.var(morph_operations[x]))
    return morph_operations[best_morphology], best_morphology

# Función para evaluar keypoints y descriptores
def evaluate_keypoints(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Harris Detector
    harris_corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    harris_corners = cv2.dilate(harris_corners, None)
    harris_keypoints = np.sum(harris_corners > 0.01 * harris_corners.max())

    # SIFT Detector
    sift = cv2.SIFT_create()
    start_sift = time.time()
    sift_keypoints, sift_descriptors = sift.detectAndCompute(gray, None)
    sift_time = time.time() - start_sift

    # ORB Detector
    orb = cv2.ORB_create()
    start_orb = time.time()
    orb_keypoints, orb_descriptors = orb.detectAndCompute(gray, None)
    orb_time = time.time() - start_orb

    # Determinar el mejor método
    best_keypoint = "Harris"
    keypoint_count = harris_keypoints

    if len(sift_keypoints) > keypoint_count:
        best_keypoint = "SIFT"
        keypoint_count = len(sift_keypoints)

    if len(orb_keypoints) > keypoint_count:
        best_keypoint = "ORB"
        keypoint_count = len(orb_keypoints)

    return best_keypoint, keypoint_count, sift_keypoints, orb_keypoints

# Función para seleccionar el mejor preprocesamiento
def best_preprocessing(image):
    processed_images = {name: func(image) for name, func in PREPROCESSING_TECHNIQUES.items()}
    best_preprocess = max(processed_images, key=lambda x: np.var(processed_images[x]))
    return processed_images[best_preprocess], best_preprocess

# Evaluar imágenes de df_mareas
df_mareas_sample = df_mareas.sample(13, random_state=42)  # Selección de 13 imágenes
results = []

fig, axes = plt.subplots(len(df_mareas_sample), 4, figsize=(15, len(df_mareas_sample) * 3))

for i, (_, row) in tqdm(enumerate(df_mareas_sample.iterrows()), total=len(df_mareas_sample)):
    img_name = row['img_name']
    img_path = os.path.join(image_folder, img_name)

    # Cargar imagen
    image = cv2.imread(img_path)
    if image is None:
        print(f"⚠️ No se encontró la imagen: {img_name}")
        continue

    # Aplicar el mejor preprocesamiento
    best_preprocess_img, best_preprocess = best_preprocessing(image)

    # Aplicar la mejor operación morfológica
    best_morph_img, best_morphology = apply_morphology(best_preprocess_img)

    # Detectar keypoints y descriptores
    best_keypoint, keypoint_count, sift_kp, orb_kp = evaluate_keypoints(best_morph_img)

    # Mostrar imágenes
    axes[i, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[i, 0].set_title(f"Original: {img_name}")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(best_preprocess_img if len(best_preprocess_img.shape) == 2 else cv2.cvtColor(best_preprocess_img, cv2.COLOR_BGR2RGB), cmap="gray")
    axes[i, 1].set_title(f"Mejor Preprocesado: {best_preprocess}")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(best_morph_img if len(best_morph_img.shape) == 2 else cv2.cvtColor(best_morph_img, cv2.COLOR_BGR2RGB), cmap="gray")
    axes[i, 2].set_title(f"Mejor Morfología: {best_morphology}")
    axes[i, 2].axis("off")

    # Dibujar keypoints en la imagen
    keypoint_image = cv2.cvtColor(best_morph_img, cv2.COLOR_GRAY2BGR) if len(best_morph_img.shape) == 2 else best_morph_img.copy()
    if best_keypoint == "SIFT":
        sift = cv2.SIFT_create()
        kp = sift.detect(cv2.cvtColor(keypoint_image, cv2.COLOR_BGR2GRAY), None)
        keypoint_image = cv2.drawKeypoints(keypoint_image, kp, None)
    elif best_keypoint == "ORB":
        orb = cv2.ORB_create()
        kp = orb.detect(cv2.cvtColor(keypoint_image, cv2.COLOR_BGR2GRAY), None)
        keypoint_image = cv2.drawKeypoints(keypoint_image, kp, None)
    else:  # Harris
        harris_corners = cv2.cornerHarris(cv2.cvtColor(keypoint_image, cv2.COLOR_BGR2GRAY), 2, 3, 0.04)
        keypoint_image[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]

    axes[i, 3].imshow(keypoint_image if len(keypoint_image.shape) == 3 else cv2.cvtColor(keypoint_image, cv2.COLOR_BGR2RGB), cmap="gray")
    axes[i, 3].set_title(f"Keypoint: {best_keypoint} ({keypoint_count})")
    axes[i, 3].axis("off")

    # Guardar resultados
    results.append({
        "img_name": img_name,
        "best_preprocess": best_preprocess,
        "best_morphology": best_morphology,
        "best_keypoint": best_keypoint,
        "keypoint_count": keypoint_count
    })

plt.tight_layout()
plt.show()

# Convertir resultados a DataFrame y guardarlos
df_results_best_mareas = pd.DataFrame(results)
csv_output = "/content/resultados_best_mareas.csv"
df_results_best_mareas.to_csv(csv_output, index=False)

print(f"Resultados guardados en: {csv_output}")


# %%
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Definir tamaño de imagen
TARGET_SIZE = (128, 128)


# Funciones de preprocesamiento
def convert_to_grayscale(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(grayscale, TARGET_SIZE)

def histogram_equalization(image):
    grayscale = convert_to_grayscale(image)
    return cv2.equalizeHist(grayscale)

# Función para aplicar la operación morfológica "opening"
def apply_opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# Función para detectar keypoints con Harris
def detect_harris_keypoints(image):
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    harris_corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    harris_corners = cv2.dilate(harris_corners, None)
    return np.sum(harris_corners > 0.01 * harris_corners.max())  # Cuenta los puntos detectados

# Función para convertir imagen en vector
def image_to_vector(image):
    return image.flatten().tolist()  # Aplana la imagen y la convierte en lista

# Función para procesar imágenes y generar vectores
def process_and_vectorize(df):
    vectors = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_name = row['img_name']
        img_path = os.path.join(image_folder, img_name)

        # Cargar imagen
        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ No se encontró la imagen: {img_name}")
            continue

        # Aplicar preprocesamiento
        img_histogram = histogram_equalization(image)
        img_opening = apply_opening(img_histogram)

        # Detectar keypoints con Harris
        keypoints_harris = detect_harris_keypoints(img_opening)

        # Convertir en vector
        img_vector = image_to_vector(img_opening)

        # Agregar datos al nuevo DataFrame
        row_data = row.to_dict()
        row_data["vector"] = img_vector
        row_data["harris_keypoints"] = keypoints_harris
        vectors.append(row_data)

    return pd.DataFrame(vectors)  # Devuelve el DataFrame con los vectores

# Procesar imágenes de galaxias con mareas
df_mareas_vector_f = process_and_vectorize(df_mareas)
df_mareas_vector_f.to_csv("/content/mareas_vectors.csv", index=False)
print("📂 df_mareas_vector_f guardado correctamente.")

# Procesar imágenes de galaxias sin mareas
df_no_mareas_vector_f = process_and_vectorize(df_no_mareas)
df_no_mareas_vector_f.to_csv("/content/no_mareas_vectors.csv", index=False)
print("📂 df_no_mareas_vector_f guardado correctamente.")


# %%
df_no_mareas_vector.shape

# %%
df_mareas_vector.shape

# %%
 pip install "pymongo[srv]"

# %%
mongodb+srv://estebanji85:Ykgh58Bs45fZIFva@universo.wcqix.mongodb.net/?retryWrites=true&w=majority&appName=Universo

# %%
from pymongo import MongoClient
import pandas as pd

# 🔹 URI de MongoDB Atlas (Reemplaza con la tuya si cambia)
MONGO_URI = "mongodb+srv://estebanji85:Ykgh58Bs45fZIFva@universo.wcqix.mongodb.net/?retryWrites=true&w=majority&appName=Universo"

# 🔹 Conectar a MongoDB Atlas
try:
    client = MongoClient(MONGO_URI)
    db = client["galaxias_db"]  # Crear o conectar a la base de datos

    # Crear colecciones
    collection_mareas_final = db["mareas_vectors_f"]  # Galaxias con mareas
    collection_no_mareas_final = db["no_mareas_vectors_f"]  # Galaxias sin mareas

    print("✅ Conexión exitosa a MongoDB Atlas")

except Exception as e:
    print(f"❌ Error de conexión: {e}")

# 🔹 Insertar DataFrames en MongoDB

def insert_dataframe_to_mongo(df, collection):
    """
    Inserta un DataFrame en una colección de MongoDB.
    """
    try:
        data_dict = df.to_dict(orient="records")  # Convertir DataFrame a lista de diccionarios
        collection.insert_many(data_dict)  # Insertar todos los documentos
        print(f"✅ {len(data_dict)} registros insertados en '{collection.name}'")

    except Exception as e:
        print(f"❌ Error al insertar en {collection.name}: {e}")

# 🔹 Insertar datos de mareas y no mareas
insert_dataframe_to_mongo(df_mareas_vector, collection_mareas)
insert_dataframe_to_mongo(df_no_mareas_vector, collection_no_mareas)

# 🔹 Verificar que los datos fueron insertados
print(f"📂 Total de documentos en 'mareas_vectors': {collection_mareas.count_documents({})}")
print(f"📂 Total de documentos en 'no_mareas_vectors': {collection_no_mareas.count_documents({})}")


# %%
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
!pip install einops transformers timm


# %%
!pip install torch torchvision torchaudio timm transformers

# %%
!pip uninstall -y torch torchvision torchaudio
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


# %%
import pymongo
import cv2
import numpy as np
import joblib
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# 📌 **1. Conectar a MongoDB Atlas**
client = pymongo.MongoClient("mongodb+srv://estebanji85:Ykgh58Bs45fZIFva@universo.wcqix.mongodb.net/?retryWrites=true&w=majority&appName=Universo")
db = client["galaxias_db"]
mareas_collection = db["mareas_vectors"]
no_mareas_collection = db["no_mareas_vectors"]

# 📌 **2. Extraer datos desde MongoDB**
def load_vectors_from_mongodb():
    mareas = list(mareas_collection.find({}, {"vector": 1, "class": 1, "_id": 0}))
    no_mareas = list(no_mareas_collection.find({}, {"vector": 1, "class": 1, "_id": 0}))

    data = mareas + no_mareas
    X = np.array([entry["vector"] for entry in data])
    y = np.array([entry["class"] for entry in data])

    return X, y

X, y = load_vectors_from_mongodb()

# 📌 **3. Normalizar datos y dividir en entrenamiento/test**
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 **4. Entrenar modelos basados en vectores**
models = {
    "SVM": SVC(probability=True),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
}

for name, model in models.items():
    print(f"Entrenando modelo: {name}")
    model.fit(X_train, y_train)
    joblib.dump(model, f"{name}_model.pkl")  # Guardar modelo

# Guardar el normalizador
joblib.dump(scaler, "scaler.pkl")

# 📌 **5. Definir CNN**
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 64 * 64, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# 📌 **6. Entrenar CNN**
cnn_model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)

# Convertir datos a tensores
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).reshape(-1, 1, 128, 128)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

for epoch in range(10):
    optimizer.zero_grad()
    outputs = cnn_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

torch.save(cnn_model.state_dict(), "cnn_model.pth")  # Guardar CNN

# 📌 **7. Función para predecir una imagen**
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

def predict_galaxy(image_path):
    img_name = image_path.split("/")[-1]

    # Buscar en MongoDB Atlas
    query = {"img_name": img_name}
    data_marea = mareas_collection.find_one(query)
    data_no_marea = no_mareas_collection.find_one(query)

    if data_marea:
        print(f"🔹 Imagen encontrada en la base de datos como 'Galaxia con Mareas'")
        return "Galaxia con Mareas", 1.0
    elif data_no_marea:
        print(f"🔹 Imagen encontrada en la base de datos como 'Galaxia sin Mareas'")
        return "Galaxia sin Mareas", 0.0

    print("⚠️ Imagen no encontrada en la base de datos. Procesando...")

    # Cargar y preprocesar la imagen
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))
    image = cv2.equalizeHist(image)
    kernel = np.ones((5,5), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # Extraer Harris Keypoints
    harris_corners = cv2.cornerHarris(image, 2, 3, 0.04)
    harris_keypoints = np.sum(harris_corners > 0.01 * harris_corners.max())

    processed_img_path = f"/content/{img_name}_processed.png"
    cv2.imwrite(processed_img_path, image)

    # Extraer características para el modelo basado en vectores
    vector = image.flatten()
    vector = np.pad(vector, (0, 16384 - len(vector)), mode='constant')
    vector = scaler.transform([vector])

    # Cargar modelos guardados
    best_model = joblib.load("RandomForest_model.pkl")  # Seleccionar mejor modelo
    cnn_model.load_state_dict(torch.load("cnn_model.pth"))
    cnn_model.eval()

    # Predicción con modelo basado en vectores
    vector_prediction = best_model.predict(vector)[0]
    vector_proba = max(best_model.predict_proba(vector)[0])

    # Predicción con CNN
    image_tensor = transform(Image.open(processed_img_path)).unsqueeze(0)
    cnn_output = cnn_model(image_tensor)
    cnn_prediction = torch.argmax(cnn_output, axis=1).item()
    cnn_proba = torch.nn.functional.softmax(cnn_output, dim=1).max().item()

    # Determinar predicción final
    if vector_prediction == cnn_prediction:
        final_prediction = "Galaxia con Mareas" if cnn_prediction == 1 else "Galaxia sin Mareas"
        confidence = max(vector_proba, cnn_proba)
    else:
        final_prediction = "Galaxia con Mareas" if cnn_proba > vector_proba else "Galaxia sin Mareas"
        confidence = max(vector_proba, cnn_proba)

    print(f"🔍 Predicción final: {final_prediction} con {confidence*100:.2f}% de confianza")

    # Guardar en MongoDB si no estaba en la base
    new_entry = {
        "img_name": img_name,
        "imagen_procesada": processed_img_path,
        "vector": vector.tolist()[0],
        "harris_keypoints": int(harris_keypoints),
        "class": 1 if final_prediction == "Galaxia con Mareas" else 0
    }
    if final_prediction == "Galaxia con Mareas":
        mareas_collection.insert_one(new_entry)
    else:
        no_mareas_collection.insert_one(new_entry)

    return final_prediction, confidence

# 📌 **8. Realizar Predicción**
image_path = "/content/drive/MyDrive/CNN/mareatest1.jpg"
predict_galaxy(image_path)





# %%
# 📌 **1. Conectar a MongoDB Atlas**
client = pymongo.MongoClient("mongodb+srv://estebanji85:Ykgh58Bs45fZIFva@universo.wcqix.mongodb.net/?retryWrites=true&w=majority&appName=Universo")
db = client["galaxias_db"]
mareas_collection = db["mareas_vectors"]
no_mareas_collection = db["no_mareas_vectors"]

# 📌 **2. Crear dataset con todos los datos**
def create_dataset():
    dataset = []

    projection = {"img_name": 1, "imagen_procesada": 1, "vector": 1, "harris_keypoints": 1, "class": 1, "_id": 0}

    for doc in mareas_collection.find({}, projection):
        dataset.append(doc)

    for doc in no_mareas_collection.find({}, projection):
        dataset.append(doc)

    return dataset
dataset = create_dataset()

# %%
import pymongo
import pandas as pd

client = pymongo.MongoClient("mongodb+srv://estebanji85:Ykgh58Bs45fZIFva@universo.wcqix.mongodb.net/?retryWrites=true&w=majority&appName=Universo")
db = client["galaxias_db"]
mareas_collection = db["mareas_vectors"]
no_mareas_collection = db["no_mareas_vectors"]

# 📌 **Crear dataset con class entre 1 y 185**
def create_dataset():
    dataset = []

    projection = {"img_name": 1, "imagen_procesada": 1, "vector": 1, "harris_keypoints": 1, "class": 1}

    # 🔹 Extraer datos de `mareas_collection` donde `class` esté entre 1 y 185
    for class_value in range(1, 186):  # Itera de 1 a 185
        for doc in mareas_collection.find({"class": class_value}, projection):
            dataset.append(doc)

    # 🔹 Extraer todos los datos de `no_mareas_collection`
    for doc in no_mareas_collection.find({}, projection):
        dataset.append(doc)

    return dataset

dataset = create_dataset()

df =  pd.DataFrame(dataset)

# Verificar cuántos documentos se extrajeron
print(f"Total de documentos en dataset: {len(dataset)}")


# %%
df.head()

# %%
import pymongo
import pandas as pd

# 📌 **1. Conectar a MongoDB**
client = pymongo.MongoClient("mongodb+srv://estebanji85:Ykgh58Bs45fZIFva@universo.wcqix.mongodb.net/?retryWrites=true&w=majority&appName=Universo")
db = client["galaxias_db"]
no_mareas_collection = db["no_mareas_vectors"]

# 📌 **2. Verificar cuántos documentos hay en la colección**
total_docs = no_mareas_collection.count_documents({})
print(f"📊 Total de documentos en 'no_mareas_vectors': {total_docs}")

if total_docs == 0:
    print("⚠️ No hay datos en 'no_mareas_vectors'. Verifica si la colección está vacía en MongoDB.")
    exit()

# 📌 **3. Verificar valores únicos de 'class' en MongoDB**
unique_classes = no_mareas_collection.distinct("class")
print(f"📌 Valores únicos de 'class' en 'no_mareas_vectors': {unique_classes}")

# 📌 **4. Crear dataset solo con `class: 0.0`**
def create_dataset():
    dataset = []

    projection = {"img_name": 1, "imagen_procesada": 1, "vector": 1, "harris_keypoints": 1, "class": 1}

    # 🔹 Buscar solo documentos donde `class` sea 0.0
    docs = list(no_mareas_collection.find({"class": 0.0}, projection))

    if docs:
        print(f"✅ Encontrados {len(docs)} documentos con class = 0.0")

    dataset.extend(docs)

    return pd.DataFrame(dataset)

# 📌 **5. Cargar el DataFrame desde MongoDB**
df_mongo = create_dataset()

# 📌 **6. Mostrar información del DataFrame**
print(f"📊 Total de registros extraídos: {df_mongo.shape[0]}")
print(df_mongo.head())


# %%
df_mongo.shape

# %%
# 📌 **4. Cargar el DataFrame que ya tienes en memoria (`df`)**
# Asegúrate de que `df` ya esté definido en tu entorno antes de ejecutar este código.

# 📌 **5. Realizar el Join entre `df` y `df_mongo` usando `img_name`**
df_final = pd.merge(df_m, df, on="img_name", how="left")  # Left Join para mantener todas las filas de `df`

# %%
df_final.head()

# %%
# Seleccionar 5 imágenes de df_mareas
sample_images = df_final.sample(5, random_state=42)  # Toma 5 imágenes aleatorias

# Crear una figura para visualizar
fig, axes = plt.subplots(1, 5, figsize=(15, 5))

for i, row in enumerate(sample_images.iterrows()):
    img_name = row[1]['img_name']
    img_path = os.path.join(image_folder, img_name)

    # Cargar imagen
    image = cv2.imread(img_path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB
        axes[i].imshow(image)
        axes[i].set_title(f"{img_name}")
        axes[i].axis("off")  # Ocultar ejes
    else:
        print(f"⚠️ No se encontró la imagen: {img_name}")

plt.show()

# %%
df_final2 = pd.merge(df_mongo, df, on="img_name", how="left")  # Left Join para mantener todas las filas de `df

# %%
df_final2.head()

# %%
# Seleccionar 5 imágenes de df_mareas
sample_images = df_final2.sample(5, random_state=42)  # Toma 5 imágenes aleatorias

# Crear una figura para visualizar
fig, axes = plt.subplots(1, 5, figsize=(15, 5))

for i, row in enumerate(sample_images.iterrows()):
    img_name = row[1]['img_name']
    img_path = os.path.join(image_folder, img_name)

    # Cargar imagen
    image = cv2.imread(img_path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB
        axes[i].imshow(image)
        axes[i].set_title(f"{img_name}")
        axes[i].axis("off")  # Ocultar ejes
    else:
        print(f"⚠️ No se encontró la imagen: {img_name}")

plt.show()

# %%
df_galaxias = pd.concat([df_final, df_final2], ignore_index=True)
df_galaxias.shape

# %%
df_galaxias.head()

# %%
df_galaxias = df_galaxias[["img_name", "vector", "harris_keypoints", "class_x", "imagen_procesada_y"]].rename(
    columns={"class_x": "class", "imagen_procesada_y": "imagen_procesada"}
)


# %%
df_galaxias.head()

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

# 📌 **1. Cargar el dataset**
df = df_galaxias.copy()  # Usamos el DataFrame de la imagen

# 📌 **2. Preprocesamiento**
# Convertir `vector` y `harris_keypoints` en features
df["vector"] = df["vector"].apply(lambda x: np.array(x) if isinstance(x, list) else np.zeros(128))  # Asegurar formato correcto
X = np.stack(df["vector"].values)  # Convertir a matriz NumPy
X = np.hstack((X, df["harris_keypoints"].values.reshape(-1, 1)))  # Añadir `harris_keypoints`
y = df["class"].astype(int)  # Variable objetivo

# 📌 **3. Dividir datos en Training (70%), Testing (20%) y Nuevos Datos (10%)**
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_test, X_new, y_test, y_new = train_test_split(X_temp, y_temp, test_size=1/3, stratify=y_temp, random_state=42)

# 📌 **4. Modelos de Clasificación**
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

# 📌 **5. Entrenar y Evaluar Modelos**
results = {}

for name, model in models.items():
    print(f"🔹 Entrenando {name}...")

    # Validación cruzada (cv=5)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")

    # Entrenar con todos los datos de training
    model.fit(X_train, y_train)

    # Predicciones en test
    y_pred = model.predict(X_test)

    # Evaluación
    report = classification_report(y_test, y_pred, output_dict=True)

    # Guardar métricas
    results[name] = {
        "accuracy": report["accuracy"],
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1_score": report["1"]["f1-score"],
        "cv_f1_mean": np.mean(scores),
        "cv_f1_std": np.std(scores),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    print(f"✅ {name} - F1 Score: {report['1']['f1-score']:.4f} (CV: {np.mean(scores):.4f} ± {np.std(scores):.4f})\n")

# 📌 **6. Seleccionar el Mejor Modelo**
best_model = max(results, key=lambda x: results[x]["f1_score"])
print(f"🏆 Mejor modelo: {best_model} con F1 Score = {results[best_model]['f1_score']:.4f}")

# 📌 **7. Graficar Matriz de Confusión del Mejor Modelo**
plt.figure(figsize=(6, 5))
sns.heatmap(results[best_model]["confusion_matrix"], annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Matriz de Confusión - {best_model}")
plt.show()


# %%
# Seleccionar 5 imágenes de df_mareas
sample_images = df_galaxias.sample(5, random_state=42)  # Toma 5 imágenes aleatorias

# Crear una figura para visualizar
fig, axes = plt.subplots(1, 5, figsize=(15, 5))

for i, row in enumerate(sample_images.iterrows()):
    img_name = row[1]['img_name']
    img_path = os.path.join(image_folder, img_name)
    print(img_path)

    # Cargar imagen
    image = cv2.imread(img_path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB
        axes[i].imshow(image)
        axes[i].set_title(f"{img_name}")
        axes[i].axis("off")  # Ocultar ejes
    else:
        print(f"⚠️ No se encontró la imagen: {img_name}")



# %%
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 📌 **1. Definir Carpeta donde están las imágenes**
image_folder = "/content/drive/MyDrive/CNN/lsb_data/imagenes"

# 📌 **2. Verificar si `image_folder` existe**
if not os.path.exists(image_folder):
    raise ValueError(f"❌ La carpeta '{image_folder}' no existe. Verifica la ruta en Google Drive.")

# 📌 **3. Verificar si hay imágenes en `image_folder`**
image_files = os.listdir(image_folder)
if len(image_files) == 0:
    raise ValueError(f"❌ No hay imágenes en '{image_folder}'. Asegúrate de mover las imágenes antes de continuar.")

print(f"📂 Total de imágenes disponibles: {len(image_files)}")

# 📌 **4. Cargar el dataset**
df = df_galaxias.copy()  # Usamos el DataFrame original

# 📌 **5. Preprocesamiento de imágenes con OpenCV**
IMAGE_SIZE = (128, 128)  # Tamaño de entrada para la CNN
X, y = [], []
missing_images = 0

# 📌 **6. Recorrer el DataFrame y cargar TODAS las imágenes usando img_name**
for i, row in enumerate(df.iterrows()):
    img_name = row[1]["img_name"]
    img_path = os.path.join(image_folder, img_name)

    print(f"🔍 Intentando cargar: {img_path}")  # 👀 Verificar qué imágenes está buscando

    if os.path.exists(img_path):
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Escala de grises
        image = cv2.resize(image, IMAGE_SIZE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # 🔹 Convertir a RGB
        image = image / 255.0  # Normalizar (0-1)

        X.append(image)
        y.append(row[1]["class"])

    else:
        missing_images += 1
        print(f"⚠️ No se encontró la imagen: {img_path}")

print(f"🔹 Total de imágenes faltantes: {missing_images}")

# 📌 **7. Verificar si `X` tiene imágenes antes de `train_test_split`**
if len(X) == 0:
    raise ValueError("❌ No se encontraron imágenes. Revisa las rutas en 'image_folder'.")

# 📌 **8. Convertir a formato NumPy**
X = np.array(X).reshape(-1, 128, 128, 3)  # 🔹 Cambiado a 3 canales (RGB)
y = np.array(y).astype(int)

# 📌 **9. Dividir datos en Training (70%), Testing (20%) y Nuevos Datos (10%)**
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_test, X_new, y_test, y_new = train_test_split(X_temp, y_temp, test_size=1/3, stratify=y_temp, random_state=42)

# 📌 **10. Función para construir modelos**
def build_model(base_model, preprocess_input):
    base = base_model(input_shape=(128, 128, 3), include_top=False, weights="imagenet")  # 🔹 Ahora acepta imágenes RGB
    base.trainable = True  # Permitir entrenamiento completo

    x = Flatten()(base.output)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)  # Clasificación binaria

    model = Model(inputs=base.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model, preprocess_input

# 📌 **11. Modelos a entrenar**
models = {
    "ResNet50": build_model(ResNet50, resnet_preprocess),
    "VGG16": build_model(VGG16, vgg_preprocess),
    "EfficientNet": build_model(EfficientNetB0, efficientnet_preprocess)
}

# 📌 **12. Entrenar y Evaluar Modelos**
history_dict, results = {}, {}

for name, (model, preprocess) in models.items():
    print(f"🔹 Entrenando {name}...")

    # Aplicar preprocesamiento específico del modelo
    X_train_prep = preprocess(X_train)
    X_test_prep = preprocess(X_test)

    # Entrenar modelo
    history = model.fit(X_train_prep, y_train, validation_data=(X_test_prep, y_test),
                        epochs=10, batch_size=32, verbose=1)

    # Guardar historia de entrenamiento
    history_dict[name] = history

    # Evaluar modelo
    y_pred = (model.predict(X_test_prep) > 0.5).astype("int32")

    # Métricas de evaluación
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Guardar métricas
    results[name] = {
        "accuracy": report["accuracy"],
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1_score": report["1"]["f1-score"],
        "confusion_matrix": cm
    }

    print(f"✅ {name} - F1 Score: {report['1']['f1-score']:.4f}\n")

# 📌 **13. Seleccionar el Mejor Modelo**
best_model = max(results, key=lambda x: results[x]["f1_score"])
print(f"🏆 Mejor modelo: {best_model} con F1 Score = {results[best_model]['f1_score']:.4f}")

# 📌 **14. Graficar Métricas**
plt.figure(figsize=(10, 6))
for name, history in history_dict.items():
    plt.plot(history.history["accuracy"], label=f"{name} - Train")
    plt.plot(history.history["val_accuracy"], label=f"{name} - Val")
plt.xlabel("Épocas")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Comparación de Modelos CNN con Preprocesamiento")
plt.show()

# 📌 **15. Matriz de Confusión del Mejor Modelo**
plt.figure(figsize=(6, 5))
sns.heatmap(results[best_model]["confusion_matrix"], annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Matriz de Confusión - {best_model}")
plt.show()



