#!/bin/bash
for i in {7..45}
do
    tesseract $i.png stdout -l deu2 -psm 10
done
