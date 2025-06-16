# ==========================================
# PREDICCIÓN DE DELITOS EN PERÚ CON ML (ACTUALIZADO)
# Variables: Total_Delitos, generico, anio
# Modelos: Random Forest, SVM, KNN
# Autor: Alvaro Maguiña
# ==========================================

# ───────────── LIBRERÍAS ─────────────
paquetes <- c("tidyverse", "caret", "randomForest", "e1071", "class", 
              "MLmetrics", "caretEnsemble", "cowplot")

nuevos <- paquetes[!(paquetes %in% installed.packages()[,"Package"])]
if(length(nuevos)) install.packages(nuevos)
lapply(paquetes, library, character.only = TRUE)

# ───────────── DATOS Y PREPROCESAMIENTO ─────────────
datos <- read_csv("C:/Users/ALVARO CHAN/Desktop/delitos/delitos_2019_2023.csv")

datos <- datos %>%
  mutate(cantidad = as.numeric(cantidad)) %>%
  filter(!is.na(cantidad), !is.na(generico), !is.na(anio)) %>%
  group_by(distrito_fiscal, generico, anio) %>%
  summarise(Total_Delitos = sum(cantidad, na.rm = TRUE)) %>%
  ungroup()

# ───────────── CLASIFICACIÓN MULTINIVEL ─────────────
q1 <- quantile(datos$Total_Delitos, 1/3, na.rm = TRUE)
q2 <- quantile(datos$Total_Delitos, 2/3, na.rm = TRUE)

datos <- datos %>%
  mutate(Incidencia = case_when(
    Total_Delitos <= q1 ~ "Baja",
    Total_Delitos <= q2 ~ "Moderada",
    TRUE ~ "Alta"
  ),
  Incidencia = factor(Incidencia, levels = c("Baja", "Moderada", "Alta")),
  generico = factor(generico),
  anio = as.numeric(anio))

# ───────────── PARTICIÓN ─────────────
set.seed(123)
split <- createDataPartition(datos$Incidencia, p = 0.8, list = FALSE)
train_data <- datos[split, ]
test_data  <- datos[-split, ]

# ───────────── RANDOM FOREST ─────────────
modelo_rf <- randomForest(Incidencia ~ Total_Delitos + generico + anio, data = train_data, ntree = 100)
pred_rf <- predict(modelo_rf, test_data)

# ───────────── SVM ─────────────
modelo_svm <- svm(Incidencia ~ Total_Delitos + generico + anio, data = train_data, kernel = "linear")
pred_svm <- predict(modelo_svm, test_data)

# ───────────── KNN ─────────────
# Normalizar variables numéricas
normalize <- function(x) (x - min(x)) / (max(x) - min(x))
train_knn <- train_data %>%
  mutate(Total_Delitos = normalize(Total_Delitos),
         anio = normalize(anio))
test_knn <- test_data %>%
  mutate(Total_Delitos = normalize(Total_Delitos),
         anio = normalize(anio))

# Codificar variable 'generico'
levels_gen <- levels(train_data$generico)
train_knn$generico <- as.numeric(factor(train_knn$generico, levels = levels_gen))
test_knn$generico <- as.numeric(factor(test_knn$generico, levels = levels_gen))

train_x <- as.matrix(train_knn[, c("Total_Delitos", "generico", "anio")])
test_x <- as.matrix(test_knn[, c("Total_Delitos", "generico", "anio")])
train_y <- train_knn$Incidencia

pred_knn <- knn(train = train_x, test = test_x, cl = train_y, k = 3)

# ───────────── FUNCIÓN MÉTRICAS ─────────────
graficar_resultado <- function(nombre_modelo, predicciones, reales) {
  cm <- confusionMatrix(as.factor(predicciones), reales)
  
  matriz <- as.data.frame(cm$table)
  graf_matriz <- ggplot(matriz, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), size = 6) +
    scale_fill_gradient(low = "white", high = "steelblue") +
    ggtitle(paste("Matriz de Confusión -", nombre_modelo)) +
    theme_minimal()
  
  acc <- round(mean(predicciones == reales), 3)
  f1  <- round(F1_Score(predicciones, reales, positive = "Alta"), 3)
  prec <- round(Precision(predicciones, reales, positive = "Alta"), 3)
  
  texto <- paste0(
    "Modelo: ", nombre_modelo, "\n",
    "Accuracy: ", acc, "\n",
    "F1 Score: ", f1, "\n",
    "Precisión: ", prec
  )
  
  graf_texto <- ggplot() + 
    annotate("text", x = 1, y = 1, label = texto, size = 5, hjust = 0) +
    theme_void() +
    ggtitle(paste("Métricas -", nombre_modelo))
  
  plot_grid(graf_matriz, graf_texto, ncol = 2)
}

# ───────────── VISUALIZACIÓN ─────────────
graficar_resultado("Random Forest", pred_rf, test_data$Incidencia)
graficar_resultado("SVM", pred_svm, test_data$Incidencia)
graficar_resultado("KNN", pred_knn, test_knn$Incidencia)

# ───────────── COMPARACIÓN FINAL ─────────────
f1_macro <- function(pred, real) {
  levels <- levels(real)
  f1_scores <- sapply(levels, function(l) {
    F1_Score(pred, real, positive = l)
  })
  mean(f1_scores, na.rm = TRUE)
}

resultados <- tibble(
  Modelo = c("Random Forest", "SVM", "KNN"),
  Accuracy = c(
    Accuracy(pred_rf, test_data$Incidencia),
    Accuracy(pred_svm, test_data$Incidencia),
    Accuracy(pred_knn, test_knn$Incidencia)
  ),
  F1_Score_Macro = c(
    f1_macro(pred_rf, test_data$Incidencia),
    f1_macro(pred_svm, test_data$Incidencia),
    f1_macro(pred_knn, test_knn$Incidencia)
  )
)

resultados_long <- resultados %>%
  pivot_longer(cols = c(Accuracy, F1_Score_Macro), names_to = "Métrica", values_to = "Valor")

ggplot(resultados_long, aes(x = Modelo, y = Valor, fill = Métrica)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  labs(title = "Comparación de Modelos: Accuracy vs F1 Score Macro", y = "Valor", x = "") +
  theme_minimal()

# ───────────── INCIDENCIA POR DISTRITO ─────────────

# Librerías necesarias
library(tidyverse)

# Leer archivo CSV
datos <- read_csv("C:/Users/ALVARO CHAN/Desktop/delitos/delitos_2019_2023.csv")

# Preprocesamiento
datos <- datos %>%
  mutate(cantidad = as.numeric(cantidad)) %>%
  group_by(distrito_fiscal) %>%
  summarise(Total_Delitos = sum(cantidad, na.rm = TRUE)) %>%
  ungroup()

# Clasificación por terciles
q1 <- quantile(datos$Total_Delitos, 1/3)
q2 <- quantile(datos$Total_Delitos, 2/3)

datos <- datos %>%
  mutate(Incidencia = case_when(
    Total_Delitos <= q1 ~ "Baja",
    Total_Delitos <= q2 ~ "Moderada",
    TRUE ~ "Alta"
  ),
  Incidencia = factor(Incidencia, levels = c("Baja", "Moderada", "Alta")))

# Gráfico
ggplot(datos, aes(x = reorder(distrito_fiscal, Total_Delitos), y = Total_Delitos, fill = Incidencia)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_manual(values = c("forestgreen", "goldenrod", "firebrick")) +
  labs(title = "Delitos totales por distrito fiscal y nivel de incidencia",
       x = "Distrito Fiscal", y = "Total de Delitos") +
  theme_minimal()

