# Классификация фотографий знаменитостей

В этой работе снова решается классическая задача классификации изображений: на этот раз датасет состоит из фото пяти знаменитостей.

В качестве модели использовался предобученный resnet34, который без заморозки слоев и прочей тонкой настройки смог довольно хорошо справиться с нашей задачей, выдав на валидационной выборке 98%-ю точность через 10 эпох обучения.
