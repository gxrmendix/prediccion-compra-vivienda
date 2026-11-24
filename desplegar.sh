#!/bin/sh
# Despliega app/ en Azure App Service (plan F1 gratuito) SIN compilar en el servidor.
#
# Por que: el plan F1 tiene una cuota de 60 minutos de CPU al dia. Si Azure instala numpy,
# scipy y scikit-learn en cada despliegue, la cuota se agota y el sitio queda deshabilitado
# hasta el dia siguiente (paso el 24/11/2026). Aqui las librerias se descargan ya compiladas
# para Linux y Python 3.12, se empaquetan con la app y el servidor solo descomprime.
#
# Requisitos: entorno conda "azure" con la herramienta az y sesion iniciada (az login).
# Uso: sh desplegar.sh
set -e
APP=prediccion-compra-vivienda
GRUPO=rg-compra-vivienda
DESTINO=build/azure
PAQUETES=$DESTINO/.python_packages/lib/site-packages

rm -rf build && mkdir -p $PAQUETES
cp -R app/app.py app/caracteristicas.py app/modelos app/templates app/static $DESTINO/
pip install -q --target $PAQUETES --python-version 3.12 --implementation cp --only-binary=:all: \
  --platform manylinux_2_28_x86_64 --platform manylinux_2_27_x86_64 --platform manylinux_2_17_x86_64 \
  --platform manylinux2014_x86_64 -r app/requirements.txt
# Las pruebas internas de scipy y scikit-learn no hacen falta en produccion
find $PAQUETES -type d \( -name tests -o -name __pycache__ \) -prune -exec rm -rf {} +
(cd $DESTINO && zip -q -r ../app.zip . -x "*.pyc" ".DS_Store")
echo "paquete: $(du -h build/app.zip | cut -f1)"

az webapp config appsettings set -n $APP -g $GRUPO -o none --settings \
  SCM_DO_BUILD_DURING_DEPLOYMENT=false ENABLE_ORYX_BUILD=false \
  PYTHONPATH=/home/site/wwwroot/.python_packages/lib/site-packages
az webapp config set -n $APP -g $GRUPO -o none \
  --startup-file "python -m gunicorn --bind=0.0.0.0 --timeout 120 --workers 1 app:app"
az webapp deploy -n $APP -g $GRUPO --src-path build/app.zip --type zip -o none

echo "esperando a que responda..."
for i in $(seq 1 40); do
  if curl -fs https://$APP.azurewebsites.net/salud; then echo; exit 0; fi
  sleep 15
done
echo "la app no respondio a tiempo; revisa: az webapp log tail -n $APP -g $GRUPO" && exit 1
