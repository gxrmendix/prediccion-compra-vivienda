#!/bin/sh
# Descarga el dataset de Kaggle (requiere el token en ~/.kaggle/access_token o KAGGLE_API_TOKEN).
mkdir -p datos
kaggle datasets download -d mohankrishnathalla/global-house-purchase-decision-dataset -p datos --unzip
