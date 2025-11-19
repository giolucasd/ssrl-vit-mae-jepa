wget https://github.com/giolucasd/ssrl-vit-mae-jepa/releases/download/v1.0/mae_weights_v1.zip
unzip mae_weights_v1.zip -d assets/weights/
mv assets/weights/mae_weights_v1/* assets/weights/
rm -r assets/weights/mae_weights_v1/
rm mae_weights_v1.zip
