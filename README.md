# SVM-FFT-Vocal-Detector


Cada etapa transforma progresivamente la información hasta pasar de una onda sonora a una decisión de clase.

---

## 1. Captura y Preprocesamiento de Audio

- Captura de audio **mono a 16 kHz**, frecuencia adecuada para voz humana.
- Procesamiento en ventanas de aproximadamente **600 ms**.
- Eliminación automática de silencios para evitar clasificar ruido o ausencia de señal.
---

## 2. Extracción de Características (Feature Extraction)

La información del usuario se extrae como audio crudo o flujo de bytes, no almacena WAVs. 
El audio crudo no es adecuado para clasificación directa.  
Por ello, cada segmento de voz se transforma en un **vector numérico de 56 características**, que funciona como la **huella acústica** de una vocal.

---

### 2.1 Dominio de Frecuencia (FFT)

Se aplica la **Transformada Rápida de Fourier (FFT)** usando ventana de Hamming para analizar la distribución espectral de la señal.

Características extraídas:

- **Centroide espectral:** indica el “centro de masa” del espectro.
- **Ancho de banda espectral:** mide la dispersión de la energía.
- **Rolloff espectral:** frecuencia donde se acumula el 85% de la energía.
- **Formantes (F1–F5):** principales picos de frecuencia del espectro.

Los **formantes** son esenciales, ya que dependen de la forma del tracto vocal y permiten diferenciar vocales entre sí.

---

### 2.2 MFCC (Mel-Frequency Cepstral Coefficients)

Los **MFCC** representan la envolvente espectral de la señal y están inspirados en la percepción auditiva humana.

- Se calculan **20 coeficientes MFCC**
- Se obtiene su **media y desviación estándar**
- Total: **40 características**

Estas características capturan el **timbre vocal**, lo cual es clave en reconocimiento de voz.

---

### 2.3 Características Temporales

Se incluyen características que describen la señal en el tiempo:

- **Zero Crossing Rate:** cantidad de cambios de signo de la señal.
- **RMS:** energía promedio de la señal.
- **Spectral Flatness:** diferencia entre sonidos tonales y ruidosos.

---

### Resultado de la Extracción

Cada vocal se representa como un **vector de 56 valores numéricos**, que resume sus propiedades acústicas relevantes.
Posteriormente se normaliza el vector de caracteristicas para reducir la dimensionalidad y el costo computacional para el modelo. 

---

## 4. Clasificación con Máquina de Vectores de Soporte (SVM)

El sistema utiliza una **SVM con kernel RBF (Radial Basis Function)**.



- Buen desempeño con **datasets pequeños**.
- Eficaz en **espacios de alta dimensión**.
- Capaz de separar clases con fronteras no lineales. 

El SVM aprende **fronteras de decisión óptimas** que separan las cinco vocales en el espacio de 56 dimensiones, maximizando el margen de separabilidad entre clases.
Aqui se probó su funcionamiento con una muestra pequeña. 
![Matriz de confusión](/confusion_matrix.png)



---

## Observaciones Importantes ### Sensibilidad Acústica 
**Vocal O:** - Requiere pronunciación **muy suave** para ser reconocida correctamente - Razón: Su espectro de frecuencias se traslapa con la U - Solución: Hablar en tono bajo y relajado 
**Vocal I:** - Debe pronunciarse **casi cantada** en registro **agudo** - Razón: Necesita formantes altos (F2 > 2000 Hz) para diferenciarse de E - Solución: Elevar el tono como si preguntaras "¿sí?" 
**Vocales E y U:** - Generalmente bien detectadas con pronunciación natural 
**Vocal A:** - La más fácil de detectar (formantes bien separados)

---

## 6. Datos de Entrenamiento

El modelo se entrena con un conjunto de datos almacenado en `training_data.json`.

```json
{
  "samples": [[feat1, feat2, ..., feat56], ...],
  "labels": ["a", "e", "i", "o", "u", ...]
}



---


