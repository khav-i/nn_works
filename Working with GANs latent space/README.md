# Исследование скрытого пространства для сгенерированных изображений

Допустим, мы обучили некоторую GAN сеть (в нашем случае DCGAN) на данных [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) и хотим произвести анализ скрытого пространства.

Все, что нам понадобится — генератор из нашей сети и его обученные веса.

В данной работе мы производим только интерполяцию между изображениями и анализ переноса признаков с помощью простых методов векторной арифметики.
