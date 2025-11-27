# Simulación autoescalado kubernetes - Teoría de Control

![Logo UTN](./resources/utn%20cuadrado.jpg)

## Profesor

- Omar Civale

## Integrantes

Apellido, Nombre | Mail
--|--
Putrino, Rodrigo Nicolás | rputrino@frba.utn.edu.ar
Rodriguez, Gonzalo Martin | gorodriguez@frba.utn.edu.ar



## Instrucciones de ejecución

### 1. Requisitos previos
- Python 3.9+ instalado.
- pip.

### 2. Clonar el repositorio
```bash
git clone https://github.com/gona7/tdc-simulacion.git
cd tdc-simulacion
```

### 3. Crear y activar un entorno virtual (opcional pero recomendado)
```bash
python -m venv .venv
source .venv/bin/activate   # En Windows: .venv\Scripts\activate
```

### 4. Instalar dependencias
```bash
pip install --upgrade pip
pip install streamlit plotly numpy pillow
```

### 5. Ejecutar la simulación
Desde la carpeta del proyecto, se debe ejecutar:
```bash
streamlit run main.py
```
También se puede usar `python main.py`, que relanza la app con Streamlit.
